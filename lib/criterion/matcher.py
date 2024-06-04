import torch
import torch.nn.functional as F
import torchshow as ts
from scipy.optimize import linear_sum_assignment
from torch import nn
from torch.cuda.amp import autocast

from ...utils import point_sample


@torch.jit.script
def batch_dice_loss(inputs: torch.Tensor, targets: torch.Tensor):
    """
    Compute the DICE loss, similar to generalized IOU for masks
    Args:
        inputs: A float tensor of arbitrary shape.
                The predictions for each example.
        targets: A float tensor with the same shape as inputs. Stores the binary
                 classification label for each element in inputs
                (0 for the negative class and 1 for the positive class).
    """
    inputs = inputs.sigmoid()
    inputs = inputs.flatten(1)
    numerator = 2 * torch.einsum("nc,mc->nm", inputs, targets)
    denominator = inputs.sum(-1)[:, None] + targets.sum(-1)[None, :]
    loss = 1 - (numerator + 1) / (denominator + 1)
    return loss


@torch.jit.script
def batch_iou_loss(inputs: torch.Tensor, targets: torch.Tensor):
    inputs = inputs.flatten(1).sigmoid().ge(0.5).float()
    targets = targets.flatten(1)

    intersection = torch.einsum("nc,mc->nm", inputs, targets)
    union = inputs.sum(-1)[:, None] + targets.sum(-1)[None, :]
    loss = 1 - (intersection + 1e-6) / (union - intersection + 1e-6)

    return loss


@torch.jit.script
def batch_sigmoid_ce_loss(inputs: torch.Tensor, targets: torch.Tensor):
    """
    Args:
        inputs: A float tensor of arbitrary shape.
                The predictions for each example.
        targets: A float tensor with the same shape as inputs. Stores the binary
                 classification label for each element in inputs
                (0 for the negative class and 1 for the positive class).
    Returns:
        Loss tensor
    """
    hw = inputs.shape[1]

    pos = F.binary_cross_entropy_with_logits(inputs, torch.ones_like(inputs), reduction="none")
    neg = F.binary_cross_entropy_with_logits(inputs, torch.zeros_like(inputs), reduction="none")

    loss = torch.einsum("nc,mc->nm", pos, targets) + torch.einsum("nc,mc->nm", neg, (1 - targets))

    return loss / hw


class Many2ManyHungarianMatcher(nn.Module):
    """This class computes an assignment between the targets and the predictions of the network

    For efficiency reasons, the targets don't include the no_object. Because of this, in general,
    there are more predictions than targets. In this case, we do a 1-to-1 matching of the best predictions,
    while the others are un-matched (and thus treated as non-objects).
    """

    def __init__(
        self,
        cost_mask: float = 1,
        cost_dice: float = 1,
        num_points: int = 5000,
    ):
        """Creates the matcher

        Params:
            cost_class: This is the relative weight of the classification error in the matching cost
            cost_mask: This is the relative weight of the focal loss of the binary mask in the matching cost
            cost_dice: This is the relative weight of the dice loss of the binary mask in the matching cost
        """
        super().__init__()
        self.cost_mask = cost_mask
        self.cost_dice = cost_dice
        assert cost_mask != 0 or cost_dice != 0, "all costs cant be 0"

        self.num_points = num_points

    def memory_efficient_forward(self, outputs, targets):
        """More memory-friendly matching"""
        # bs, num_queries = outputs["pred_logits"].shape[:2]
        bs, num_points, num_masks = outputs["pred_masks"].shape[:3]

        indices = []

        # Iterate through batch size
        for b in range(bs):
            out_mask = outputs["pred_masks"][b]  # [num_points, num_masks, H_pred, W_pred]
            # gt masks are already padded when preparing target
            tgt_mask = targets[b]["masks"].to(out_mask)  # [num_instance, H_gt, W_gt]
            num_instance = tgt_mask.shape[0]

            tgt_points = targets[b]["points"].long()  # [num_points, 2]
            # NOTE: tgt_mask is H, W
            indicator = tgt_mask[
                :, tgt_points[:, 1], tgt_points[:, 0]  # this indexing is correct.
            ].T  # [num_instance, num_points] -> [num_points, num_instance]
            points_notin_masks = (indicator == 0).repeat_interleave(
                num_masks, dim=0
            )  # [num_points * num_masks, num_instance]

            tgt_mask = tgt_mask.unsqueeze(0)
            out_mask = out_mask.flatten(0, 1).unsqueeze(0)
            # # shared point coords
            # # all masks share the same set of points for efficient matching!
            # point_coords = torch.rand(1, self.num_points, 2, device=out_mask.device)
            # # get gt labels
            # tgt_mask = point_sample(
            #     tgt_mask,
            #     point_coords,
            #     align_corners=False,
            # ).squeeze(0)
            # out_mask = point_sample(
            #     out_mask,
            #     point_coords,
            #     align_corners=False,
            # ).squeeze(0)
            out_mask = F.interpolate(out_mask, scale_factor=1 / 2, mode="bilinear")
            tgt_mask = F.interpolate(tgt_mask, scale_factor=1 / 8, mode="nearest")
            out_mask = out_mask.squeeze(0).flatten(1)
            tgt_mask = tgt_mask.squeeze(0).flatten(1)

            with autocast(enabled=False):
                out_mask = out_mask.float()
                tgt_mask = tgt_mask.float()
                # Compute the focal loss between masks
                cost_mask = batch_sigmoid_ce_loss(out_mask, tgt_mask)  # [num_points * num_masks, num_instance]
                # Compute the dice loss betwen masks
                cost_dice = batch_dice_loss(out_mask, tgt_mask)  # [num_points * num_masks, num_instance]
            # Final cost matrix
            C = self.cost_mask * cost_mask + self.cost_dice * cost_dice + 1e6 * points_notin_masks
            C = C.reshape(num_points, num_masks, num_instance).cpu()
            cost_dice = cost_dice.reshape(num_points, num_masks, num_instance).cpu()

            row_indices, col_indices, values, dices = [], [], [], []
            for i, c in enumerate(C):
                row_ind, col_ind = linear_sum_assignment(c)
                row_ind = torch.as_tensor(row_ind, dtype=torch.int64)
                value = c[row_ind, col_ind]
                dice = cost_dice[i, row_ind, col_ind]
                row_ind += i * num_masks
                row_indices.append(torch.as_tensor(row_ind, dtype=torch.int64))
                col_indices.append(torch.as_tensor(col_ind, dtype=torch.int64))
                values.append(value)
                dices.append(dice)
            del C
            row_indices = torch.cat(row_indices)
            col_indices = torch.cat(col_indices)
            values = torch.cat(values)
            dices = torch.cat(dices)

            valid = values < 1e6
            row_indices = row_indices[valid].to(out_mask.device)
            col_indices = col_indices[valid].to(out_mask.device)
            values = values[valid].to(out_mask.device)
            dices = dices[valid].to(out_mask.device)
            indices.append((row_indices, col_indices, dices))
            del out_mask, tgt_mask

        return indices

    @torch.no_grad()
    def forward(self, outputs, targets):
        """Performs the matching

        Params:
            outputs: This is a dict that contains at least these entries:
                 "pred_logits": Tensor of dim [batch_size, num_queries, num_classes] with the classification logits
                 "pred_masks": Tensor of dim [batch_size, num_queries, H_pred, W_pred] with the predicted masks

            targets: This is a list of targets (len(targets) = batch_size), where each target is a dict containing:
                 "labels": Tensor of dim [num_target_boxes] (where num_target_boxes is the number of ground-truth
                           objects in the target) containing the class labels
                 "masks": Tensor of dim [num_target_boxes, H_gt, W_gt] containing the target masks

        Returns:
            A list of size batch_size, containing tuples of (index_i, index_j) where:
                - index_i is the indices of the selected predictions (in order)
                - index_j is the indices of the corresponding selected targets (in order)
            For each batch element, it holds:
                len(index_i) = len(index_j) = min(num_queries, num_target_boxes)
        """
        return self.memory_efficient_forward(outputs, targets)

    def __repr__(self, _repr_indent=4):
        head = "Matcher " + self.__class__.__name__
        body = [
            "cost_mask: {}".format(self.cost_mask),
            "cost_dice: {}".format(self.cost_dice),
            "num_points: {}".format(self.num_points),
        ]
        lines = [head] + [" " * _repr_indent + line for line in body]
        return "\n".join(lines)


class One2OneHungarianMatcher(nn.Module):
    """This class computes an assignment between the targets and the predictions of the network

    For efficiency reasons, the targets don't include the no_object. Because of this, in general,
    there are more predictions than targets. In this case, we do a 1-to-1 matching of the best predictions,
    while the others are un-matched (and thus treated as non-objects).
    """

    def __init__(
        self,
        cost_mask: float = 1,
        cost_dice: float = 1,
        num_points: int = 0,
        thres_pos: float = 0.5,
    ):
        """Creates the matcher

        Params:
            cost_class: This is the relative weight of the classification error in the matching cost
            cost_mask: This is the relative weight of the focal loss of the binary mask in the matching cost
            cost_dice: This is the relative weight of the dice loss of the binary mask in the matching cost
        """
        super().__init__()
        self.cost_mask = cost_mask
        self.cost_dice = cost_dice
        self.num_points = num_points
        self.thres_pos = thres_pos

    @torch.no_grad()
    def memory_efficient_forward(self, outputs, targets):
        """More memory-friendly matching"""
        bs, num_queries = outputs["pred_masks"].shape[:2]

        indices = []

        # Iterate through batch size
        for b in range(bs):
            out_mask = outputs["pred_masks"][b]  # [num_queries, H_pred, W_pred]
            # gt masks are already padded when preparing target
            tgt_mask = targets[b]["masks"].to(out_mask)

            out_mask = out_mask[:, None]
            tgt_mask = tgt_mask[:, None]
            # # all masks share the same set of points for efficient matching!
            # point_coords = torch.rand(1, self.num_points, 2, device=out_mask.device)
            # # get gt labels
            # tgt_mask = point_sample(
            #     tgt_mask,
            #     point_coords.repeat(tgt_mask.shape[0], 1, 1),
            #     align_corners=False,
            # ).squeeze(1)
            # out_mask = point_sample(
            #     out_mask,
            #     point_coords.repeat(out_mask.shape[0], 1, 1),
            #     align_corners=False,
            # ).squeeze(1)
            out_mask = F.interpolate(out_mask, scale_factor=1 / 2, mode="bilinear")
            tgt_mask = F.interpolate(tgt_mask, scale_factor=1 / 8, mode="nearest")
            out_mask = out_mask.flatten(1)
            tgt_mask = tgt_mask.flatten(1)

            with autocast(enabled=False):
                out_mask = out_mask.float()
                tgt_mask = tgt_mask.float()
                # # Compute the focal loss between masks
                # cost_mask = batch_sigmoid_ce_loss(out_mask, tgt_mask)
                # Compute the dice loss betwen masks
                # cost_dice = batch_dice_loss(out_mask, tgt_mask)
                cost_dice = batch_iou_loss(out_mask, tgt_mask)

            # Final cost matrix
            # C = self.cost_mask * cost_mask + self.cost_dice * cost_dice
            C = self.cost_dice * cost_dice
            C = C.reshape(num_queries, -1).cpu()
            row_ind, col_ind = linear_sum_assignment(C)
            row_ind = torch.as_tensor(row_ind, dtype=torch.int64, device=out_mask.device)
            col_ind = torch.as_tensor(col_ind, dtype=torch.int64, device=out_mask.device)
            cost_dice = cost_dice.reshape(num_queries, -1)  # .cpu()
            value = 1.0 - cost_dice[row_ind, col_ind]
            valid = value.ge(self.thres_pos)
            row_ind = row_ind[valid]
            col_ind = col_ind[valid]

            indices.append((row_ind, col_ind))
        return indices

    @torch.no_grad()
    def forward(self, outputs, targets):
        """Performs the matching

        Params:
            outputs: This is a dict that contains at least these entries:
                 "pred_logits": Tensor of dim [batch_size, num_queries, num_classes] with the classification logits
                 "pred_masks": Tensor of dim [batch_size, num_queries, H_pred, W_pred] with the predicted masks

            targets: This is a list of targets (len(targets) = batch_size), where each target is a dict containing:
                 "labels": Tensor of dim [num_target_boxes] (where num_target_boxes is the number of ground-truth
                           objects in the target) containing the class labels
                 "masks": Tensor of dim [num_target_boxes, H_gt, W_gt] containing the target masks

        Returns:
            A list of size batch_size, containing tuples of (index_i, index_j) where:
                - index_i is the indices of the selected predictions (in order)
                - index_j is the indices of the corresponding selected targets (in order)
            For each batch element, it holds:
                len(index_i) = len(index_j) = min(num_queries, num_target_boxes)
        """
        return self.memory_efficient_forward(outputs, targets)

    def __repr__(self, _repr_indent=4):
        head = "Matcher " + self.__class__.__name__
        body = [
            "cost_mask: {}".format(self.cost_mask),
            "cost_dice: {}".format(self.cost_dice),
            "num_points: {}".format(self.num_points),
            "thres_pos: {}".format(self.thres_pos),
        ]
        lines = [head] + [" " * _repr_indent + line for line in body]
        return "\n".join(lines)
