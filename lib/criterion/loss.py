import math
from typing import Dict

import torch
import torch.nn.functional as F
from scipy.optimize import linear_sum_assignment
from torch import nn

from detectron2.config import configurable
from detectron2.utils.comm import get_rank, get_world_size, is_main_process
from detectron2.utils.registry import Registry

from ...utils import (
    all_gather_no_grad,
    calculate_uncertainty,
    get_uncertain_point_coords_with_randomness,
    is_dist_avail_and_initialized,
    nested_tensor_from_tensor_list,
    point_sample,
)
from .matcher import Many2ManyHungarianMatcher, One2OneHungarianMatcher

CRITERION_REGISTRY = Registry("CRITERION")
CRITERION_REGISTRY.__doc__ = """
Registry for criterion in Uni-OVSeg.
"""


def build_criterion(cfg, name):
    """
    Build a instance embedding branch from `cfg.MODEL.INS_EMBED_HEAD.NAME`.
    """
    return CRITERION_REGISTRY.get(name)(cfg)


@torch.jit.script
def dice_loss(
    inputs: torch.Tensor,
    targets: torch.Tensor,
    num_masks: float,
):
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
    numerator = 2 * (inputs * targets).sum(-1)
    denominator = inputs.sum(-1) + targets.sum(-1)
    loss = 1 - (numerator + 1) / (denominator + 1)
    return loss.sum() / num_masks


@torch.jit.script
def dice_value(
    inputs: torch.Tensor,
    targets: torch.Tensor,
):
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
    targets = targets.flatten(1)
    numerator = 2 * (inputs * targets).sum(-1)
    denominator = inputs.sum(-1) + targets.sum(-1)
    dice = (numerator + 1) / (denominator + 1)
    return dice


@torch.jit.script
def sigmoid_ce_loss(
    inputs: torch.Tensor,
    targets: torch.Tensor,
    num_masks: float,
):
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
    loss = F.binary_cross_entropy_with_logits(inputs, targets, reduction="none")

    return loss.mean(1).sum() / num_masks


@CRITERION_REGISTRY.register()
class Many2ManySetCriterion(nn.Module):
    @configurable
    def __init__(
        self,
        matcher: nn.Module,
        weight_dict: Dict,
        num_points: int,
        oversample_ratio: float,
        importance_sample_ratio: float,
    ):
        """Create the criterion.
        Parameters:
            num_classes: number of object categories, omitting the special no-object category
            matcher: module able to compute a matching between targets and proposals
            weight_dict: dict containing as key the names of the losses and as values their relative weight.
            eos_coef: relative classification weight applied to the no-object category
            losses: list of all the losses to be applied. See get_loss for list of available losses.
        """
        super().__init__()
        self.matcher = matcher
        self.weight_dict = weight_dict

        # pointwise mask loss parameters
        self.num_points = num_points
        self.oversample_ratio = oversample_ratio
        self.importance_sample_ratio = importance_sample_ratio

    @classmethod
    def from_config(cls, cfg):
        weight_dict = {
            "loss_mask": cfg.MODEL.OVSEG.MASK_WEIGHT,
            "loss_dice": cfg.MODEL.OVSEG.DICE_WEIGHT,
            "loss_iou": cfg.MODEL.OVSEG.IOU_WEIGHT,
        }

        if cfg.MODEL.OVSEG.DEEP_SUPERVISION:
            dec_layers = cfg.MODEL.OVSEG.DEC_LAYERS
            aux_weight_dict = {}
            for i in range(dec_layers - 1):
                aux_weight_dict.update({k + f"_{i}": v for k, v in weight_dict.items()})
            weight_dict.update(aux_weight_dict)

        matcher = Many2ManyHungarianMatcher(
            cost_mask=cfg.MODEL.OVSEG.MASK_WEIGHT,
            cost_dice=cfg.MODEL.OVSEG.DICE_WEIGHT,
            num_points=cfg.MODEL.OVSEG.MATCHER_NUM_POINTS,
        )
        return {
            "matcher": matcher,
            "weight_dict": weight_dict,
            "num_points": cfg.MODEL.OVSEG.TRAIN_NUM_POINTS,
            "oversample_ratio": cfg.MODEL.OVSEG.OVERSAMPLE_RATIO,
            "importance_sample_ratio": cfg.MODEL.OVSEG.IMPORTANCE_SAMPLE_RATIO,
        }

    def loss_masks_paired(self, outputs, targets, indices, num_masks):
        """Compute the losses related to the masks: the focal loss and the dice loss.
        targets dicts must contain the key "masks" containing a tensor of dim [nb_target_boxes, h, w]
        """
        src_masks = outputs["pred_masks"]  # bs, points, masks, h, w
        src_masks = src_masks.flatten(0, 2)  # bs * points * masks, h, w
        masks = [t["masks"] for t in targets]
        # TODO use valid to mask invalid areas due to padding in loss
        target_masks, valid = nested_tensor_from_tensor_list(masks).decompose()
        target_masks = target_masks.to(src_masks)  # bs, max_instance, h, w
        target_masks = target_masks.flatten(0, 1)  # bs * max_instance, h, w

        # No need to upsample predictions as we are using normalized coordinates :)
        # N x 1 x H x W
        src_masks = src_masks[:, None]
        target_masks = target_masks[:, None]

        with torch.no_grad():
            # sample point_coords
            point_coords = get_uncertain_point_coords_with_randomness(
                src_masks,
                lambda logits: calculate_uncertainty(logits),
                self.num_points,
                self.oversample_ratio,
                self.importance_sample_ratio,
            )
            # get gt labels
            point_labels = point_sample(
                target_masks,
                point_coords,
                align_corners=False,
            ).squeeze(1)

        point_logits = point_sample(
            src_masks,
            point_coords,
            align_corners=False,
        ).squeeze(1)

        losses = {
            "loss_mask": sigmoid_ce_loss(point_logits, point_labels, num_masks),
            "loss_dice": dice_loss(point_logits, point_labels, num_masks),
        }

        del src_masks, target_masks, masks
        return losses

    def loss_ious_paired(self, outputs, targets, indices, num_masks):
        """Quality loss (Smooth L1)
        targets dicts must contain the key "pred_ious" containing a tensor of dim [nb_target_boxes]
        """
        assert "pred_ious" in outputs
        with torch.no_grad():
            src_masks = outputs["pred_masks"]  # bs, points, masks, h, w
            src_masks = src_masks.flatten(0, 2)  # bs * points * masks, h, w
            masks = [t["masks"] for t in targets]
            # TODO use valid to mask invalid areas due to padding in loss
            target_masks, valid = nested_tensor_from_tensor_list(masks).decompose()
            target_masks = target_masks.to(src_masks)  # bs, max_instance, h, w
            target_masks = F.interpolate(target_masks, scale_factor=0.25, mode="nearest")
            target_masks = target_masks.flatten(0, 1)  # bs * max_instance, h, w
            # N x H x W
            target_ious = dice_value(src_masks, target_masks)

        src_ious = outputs["pred_ious"]  # bs, points, masks
        src_ious = src_ious.flatten(0, 2)  # bs * points * masks

        loss_iou = F.smooth_l1_loss(src_ious, target_ious, reduction="sum") / num_masks
        losses = {"loss_iou": loss_iou}
        return losses

    def loss_masks(self, outputs, targets, indices, num_masks):
        """Compute the losses related to the masks: the focal loss and the dice loss.
        targets dicts must contain the key "masks" containing a tensor of dim [nb_target_boxes, h, w]
        """
        src_idx, _ = self._get_src_permutation_idx(indices)
        tgt_idx, _ = self._get_tgt_permutation_idx(indices)
        src_masks = outputs["pred_masks"]  # bs, points, masks, h, w
        src_masks = src_masks.flatten(1, 2)  # bs, points * masks, h, w
        src_masks = src_masks[src_idx]  # samples, h, w
        masks = [t["masks"] for t in targets]
        # TODO use valid to mask invalid areas due to padding in loss
        target_masks, valid = nested_tensor_from_tensor_list(masks).decompose()
        target_masks = target_masks.to(src_masks)  # bs, max_instance, h, w
        target_masks = target_masks[tgt_idx]  # samples, h, w

        # No need to upsample predictions as we are using normalized coordinates :)
        # N x 1 x H x W
        src_masks = src_masks[:, None]
        target_masks = target_masks[:, None]

        with torch.no_grad():
            # sample point_coords
            point_coords = get_uncertain_point_coords_with_randomness(
                src_masks,
                lambda logits: calculate_uncertainty(logits),
                self.num_points,
                self.oversample_ratio,
                self.importance_sample_ratio,
            )
            # get gt labels
            point_labels = point_sample(
                target_masks,
                point_coords,
                align_corners=False,
            ).squeeze(1)

        point_logits = point_sample(
            src_masks,
            point_coords,
            align_corners=False,
        ).squeeze(1)

        losses = {
            "loss_mask": sigmoid_ce_loss(point_logits, point_labels, num_masks),
            "loss_dice": dice_loss(point_logits, point_labels, num_masks),
        }

        del src_masks, target_masks, masks
        return losses

    def loss_ious(self, outputs, targets, indices, num_masks):
        """Quality loss (Smooth L1)
        targets dicts must contain the key "qualities" containing a tensor of dim [nb_target_boxes]
        """
        assert "pred_ious" in outputs

        src_idx, cost = self._get_src_permutation_idx(indices)
        src_ious = outputs["pred_ious"]  # bs, points, masks
        src_ious = src_ious.flatten(1, 2)  # bs, points * masks
        src_ious = src_ious[src_idx]  # samples
        target_ious = 1.0 - cost

        loss_iou = F.smooth_l1_loss(src_ious, target_ious, reduction="sum") / num_masks
        losses = {"loss_iou": loss_iou}
        return losses

    def _get_src_permutation_idx(self, indices):
        # permute predictions following indices
        batch_idx, src_idx, cost = [], [], []
        for i, (src, _, value) in enumerate(indices):
            batch_idx.append(torch.full_like(src, i))
            src_idx.append(src)
            cost.append(value)
        batch_idx = torch.cat(batch_idx)
        src_idx = torch.cat(src_idx)
        cost = torch.cat(cost)
        return (batch_idx, src_idx), cost

    def _get_tgt_permutation_idx(self, indices):
        # permute targets following indices
        batch_idx, tgt_idx, cost = [], [], []
        for i, (_, tgt, value) in enumerate(indices):
            batch_idx.append(torch.full_like(tgt, i))
            tgt_idx.append(tgt)
            cost.append(value)
        batch_idx = torch.cat(batch_idx)
        tgt_idx = torch.cat(tgt_idx)
        cost = torch.cat(cost)

        return (batch_idx, tgt_idx), cost

    def get_loss(self, loss, outputs, targets, indices, num_masks):
        loss_map = {
            "masks": self.loss_masks,
            "masks_paired": self.loss_masks_paired,
            "ious": self.loss_ious,
            "ious_paired": self.loss_ious_paired,
        }
        assert loss in loss_map, f"do you really want to compute {loss} loss?"
        return loss_map[loss](outputs, targets, indices, num_masks)

    def forward(self, outputs, targets, losses_name):
        """This performs the loss computation.
        Parameters:
             outputs: dict of tensors, see the output specification of the model for the format
             targets: list of dicts, such that len(targets) == batch_size.
                      The expected keys in each dict depends on the losses applied, see each loss' doc
        """
        outputs_without_aux = {k: v for k, v in outputs.items() if k != "aux_outputs"}

        # Retrieve the matching between the outputs of the last layer and the targets
        if "masks_paired" not in losses_name:
            indices = self.matcher(outputs_without_aux, targets)
            num_masks = sum(len(ind[0]) for ind in indices)
        else:
            indices = None
            num_masks = sum(len(t["masks"]) for t in targets)

        # Compute the average number of target boxes accross all nodes, for normalization purposes
        num_masks = torch.as_tensor([num_masks], dtype=torch.float, device=next(iter(outputs.values())).device)
        if is_dist_avail_and_initialized():
            torch.distributed.all_reduce(num_masks)
        num_masks = torch.clamp(num_masks / get_world_size(), min=1).item()

        # Compute all the requested losses
        losses = {}
        # for loss in self.losses:
        for loss in losses_name:
            losses.update(self.get_loss(loss, outputs, targets, indices, num_masks))

        # In case of auxiliary losses, we repeat this process with the output of each intermediate layer.
        if "aux_outputs" in outputs:
            for i, aux_outputs in enumerate(outputs["aux_outputs"]):
                # indices = self.matcher(aux_outputs, targets)
                for loss in losses_name:
                    l_dict = self.get_loss(loss, aux_outputs, targets, indices, num_masks)
                    l_dict = {k + f"_{i}": v for k, v in l_dict.items()}
                    losses.update(l_dict)

        return losses

    def __repr__(self):
        head = "Criterion: " + self.__class__.__name__
        body = [
            "matcher: {}".format(self.matcher.__repr__(_repr_indent=8)),
            "weight_dict: {}".format(self.weight_dict),
            "num_points: {}".format(self.num_points),
            "oversample_ratio: {}".format(self.oversample_ratio),
            "importance_sample_ratio: {}".format(self.importance_sample_ratio),
        ]
        _repr_indent = 4
        lines = [head] + [" " * _repr_indent + line for line in body]
        return "\n".join(lines)


@CRITERION_REGISTRY.register()
class MaskTextAlignCriterion_legacy(nn.Module):
    @configurable
    def __init__(self, loss_topk: float = 1.0):
        super().__init__()
        self.loss_topk = loss_topk

    @classmethod
    def from_config(cls, cfg):
        return {"loss_topk": cfg.MODEL.OVSEG.LOSS_TOPK}

    @torch.no_grad()
    def mask_text_match(self, cost):
        cost = 1.0 - cost.cpu()
        row_ind, col_ind = linear_sum_assignment(cost)
        row_ind = torch.as_tensor(row_ind, device=cost.device).long()
        col_ind = torch.as_tensor(col_ind, device=cost.device).long()
        if self.loss_topk < 1.0:
            cost = cost[row_ind, col_ind]
            K = max(int(self.loss_topk * cost.shape[0]), 1)
            cost, idx = cost.topk(K, largest=False)
            row_ind = row_ind[idx]
            col_ind = col_ind[idx]
        return row_ind, col_ind

    def forward(self, dlip_embeds, clip_embeds, txt_embeds, logit_scale, scale=None):
        batch_size = len(dlip_embeds)
        losses = []
        for b in range(batch_size):
            dlip_embed = dlip_embeds[b]
            clip_embed = clip_embeds[b]
            txt_embed = txt_embeds[b]
            # similarity distribution
            dist_clip = torch.einsum("kqc,tc->kqt", [clip_embed, txt_embed])
            dist_clip = (dist_clip * logit_scale).softmax(-1).mean(0)
            # dist_dlip = torch.einsum("qc,tc->qt", [dlip_embed, txt_embed])
            # dist_dlip = (dist_dlip * logit_scale).softmax(-1)
            # dist_en = dist_dlip * 0.5 * scale + dist_clip * 0.5
            # bipartite matching
            row_ind, col_ind = self.mask_text_match(dist_clip)
            dlip_embed = dlip_embed[row_ind]
            txt_embed = txt_embed[col_ind]
            # cosine similarity loss
            loss = 1.0 - (dlip_embed * txt_embed).sum(dim=1)
            losses.append(loss)
        losses = torch.cat(losses).mean()

        return {"loss_align": losses}


# @CRITERION_REGISTRY.register()
# class MaskTextAlignCriterion(nn.Module):
#     @configurable
#     def __init__(self, num_sigma: int = 2, momentum: float = 0.999):
#         super().__init__()
#         self.num_sigma = num_sigma
#         self.momentum = momentum
#         self.register_buffer("thres_conf", None)

#     @classmethod
#     def from_config(cls, cfg):
#         return {
#             "num_sigma": cfg.MODEL.OVSEG.NUM_SIGMA,
#             "momentum": cfg.MODEL.OVSEG.MOMENTUM,
#         }

#     @torch.no_grad()
#     def bipartite_match(self, cos, logit_scale, use_prob=False):
#         if use_prob:
#             cost = (cos * logit_scale).float().softmax(-1)
#             cost = -cost.cpu()
#         else:
#             cost = -cos.float().cpu()
#         row_ind, col_ind = linear_sum_assignment(cost)
#         row_ind = torch.as_tensor(row_ind, device=cost.device).long()
#         col_ind = torch.as_tensor(col_ind, device=cost.device).long()
#         return row_ind, col_ind

#     @torch.no_grad()
#     def greedy_match(self, cos, logit_scale):
#         # cost: [M, N]
#         cost = (cos * logit_scale).float().softmax(-1)
#         col_ind = torch.argmax(cost, dim=-1)  # [M]
#         row_ind = torch.arange(cost.size(0), device=cost.device)  # [M]
#         # update thres_conf
#         if cost.shape[0] > 1:
#             hard_label = F.one_hot(col_ind, cost.shape[1])
#             entropy = -torch.sum(hard_label * (cost + 1e-8).log(), dim=-1)
#             mean_ent = entropy.mean()
#             std_ent = entropy.std()
#             thres_cur = mean_ent + self.num_sigma * std_ent
#         else:
#             thres_cur = self.thres_conf.clone()
#         thres_cur = all_gather_no_grad(thres_cur.reshape(1)).mean()
#         if self.thres_conf is None:
#             self.thres_conf = thres_cur
#         else:
#             self.thres_conf = (
#                 self.momentum * self.thres_conf + (1.0 - self.momentum) * thres_cur
#             )
#         # filter out low confidence
#         if cost.shape[0] > 1:
#             mask = entropy <= self.thres_conf
#             row_ind = row_ind[mask]
#             col_ind = col_ind[mask]

#         return row_ind, col_ind

#     def forward(self, dlip_embeds, clip_embeds, txt_embeds, logit_scale, scale=None):
#         batch_size = len(dlip_embeds)
#         losses = []
#         for b in range(batch_size):
#             dlip_embed = dlip_embeds[b]
#             clip_embed = clip_embeds[b]
#             txt_embed = txt_embeds[b]
#             # similarity distribution
#             cos = torch.einsum("qc,tc->qt", [clip_embed, txt_embed])
#             # mask-text matching
#             row_ind, col_ind = self.bipartite_match(cos, logit_scale, use_prob=True)
#             # row_ind, col_ind = self.greedy_match(cos, logit_scale)
#             dlip_embed = dlip_embed[row_ind]
#             txt_embed = txt_embed[col_ind]
#             # cosine similarity loss
#             loss = 1.0 - (dlip_embed * txt_embed).sum(dim=1)
#             losses.append(loss)
#         losses = torch.cat(losses).mean()

#         return {"loss_align": losses}


@CRITERION_REGISTRY.register()
class MaskTextAlignCriterion(nn.Module):
    @configurable
    def __init__(self, total_step, num_sigma: int = 2, momentum: float = 0.999):
        super().__init__()
        self.total_step = total_step
        self.num_sigma = num_sigma
        self.momentum = momentum
        self.register_buffer("thres_conf", None)
        self.register_buffer("cur_step", torch.zeros(1), persistent=False)

    @classmethod
    def from_config(cls, cfg):
        return {
            "total_step": cfg.SOLVER.MAX_ITER,
            "num_sigma": cfg.MODEL.OVSEG.NUM_SIGMA,
            "momentum": cfg.MODEL.OVSEG.MOMENTUM,
        }

    @torch.no_grad()
    def bipartite_match(self, cos, logit_scale, use_prob=False):
        if use_prob:
            cost = (cos * logit_scale).float().softmax(-1)
            cost = -cost.cpu()
        else:
            cost = -cos.float().cpu()
        row_ind, col_ind = linear_sum_assignment(cost)
        row_ind = torch.as_tensor(row_ind, device=cost.device).long()
        col_ind = torch.as_tensor(col_ind, device=cost.device).long()
        return row_ind, col_ind

    @torch.no_grad()
    def greedy_match(self, cos, logit_scale):
        # cost: [M, N]
        cost = (cos * logit_scale).float().softmax(-1)
        col_ind = torch.argmax(cost, dim=-1)  # [M]
        row_ind = torch.arange(cost.size(0), device=cost.device)  # [M]
        # update thres_conf
        if cost.shape[0] > 1:
            hard_label = F.one_hot(col_ind, cost.shape[1])
            entropy = -torch.sum(hard_label * (cost + 1e-8).log(), dim=-1)
            mean_ent = entropy.mean()
            std_ent = entropy.std()
            thres_cur = mean_ent + self.num_sigma * std_ent
        else:
            thres_cur = self.thres_conf.clone()
        thres_cur = all_gather_no_grad(thres_cur.reshape(1)).mean()
        if self.thres_conf is None:
            self.thres_conf = thres_cur
        else:
            self.thres_conf = self.momentum * self.thres_conf + (1.0 - self.momentum) * thres_cur
        # filter out low confidence
        if cost.shape[0] > 1:
            mask = entropy <= self.thres_conf
            row_ind = row_ind[mask]
            col_ind = col_ind[mask]

        return row_ind, col_ind

    def _linear_schedule(self, cur_step):
        return 0.5 * cur_step.float() / self.total_step

    def _cosine_schedule(self, cur_step):
        return 0.5 * 0.5 * (1.0 - torch.cos(cur_step.float() / self.total_step * math.pi))

    def _poly_decay_scheduler(self, cur_step):
        return 0.5 * (cur_step.float() / self.total_step) ** 0.5

    def forward(self, dlip_embeds, dlip_embeds_target, clip_embeds, txt_embeds, logit_scale, scale=None):
        # get current scaler
        self.cur_step += 1
        # linear
        scale = self._linear_schedule(self.cur_step.clone())
        # scale = self._cosine_schedule(self.cur_step.clone())
        # scale = self._poly_decay_scheduler(self.cur_step.clone())

        batch_size = len(dlip_embeds)
        losses = []
        for b in range(batch_size):
            dlip_embed = dlip_embeds[b]
            dlip_embed_target = dlip_embeds_target[b]
            clip_embed = clip_embeds[b]
            txt_embed = txt_embeds[b]
            # similarity distribution
            clip_embed = (1 - scale) * clip_embed + scale * dlip_embed_target
            clip_embed = F.normalize(clip_embed, dim=-1)
            cos = torch.einsum("qc,tc->qt", [clip_embed, txt_embed])
            # mask-text matching
            row_ind, col_ind = self.bipartite_match(cos, logit_scale, use_prob=True)
            dlip_embed = dlip_embed[row_ind]
            txt_embed = txt_embed[col_ind]
            # cosine similarity loss
            loss = 1.0 - (dlip_embed * txt_embed).sum(dim=1)
            losses.append(loss)
        losses = torch.cat(losses).mean()

        return {"loss_align": losses}


class MaskAlignCriterion(nn.Module):
    def __init__(self, weight_dict) -> None:
        super().__init__()
        self.weight_dict = weight_dict

    def forward(self, online_mask, online_iou, online_aux, target_mask, target_iou, target_aux):
        B, M, H, W = online_mask.shape
        losses = dict()

        # main loss
        losses["loss_dice"] = (
            0.1
            * self.weight_dict["loss_dice"]
            * dice_loss(
                online_mask.reshape(B * M, H * W),
                target_mask.reshape(B * M, H * W).sigmoid().detach(),
                B * M,
            )
        )
        losses["loss_mask"] = (
            0.1
            * self.weight_dict["loss_mask"]
            * sigmoid_ce_loss(
                online_mask.reshape(B * M, H * W),
                target_mask.reshape(B * M, H * W).sigmoid().detach(),
                B * M,
            )
        )
        losses["loss_iou"] = 0.1 * self.weight_dict["loss_iou"] * F.smooth_l1_loss(online_iou, target_iou.detach())
        # aux loss
        for i, (online, target) in enumerate(zip(online_aux, target_aux)):
            losses[f"loss_dice@{i}"] = (
                0.1
                * self.weight_dict["loss_dice"]
                * dice_loss(
                    online["pred_masks"].reshape(B * M, H * W),
                    target["pred_masks"].reshape(B * M, H * W).sigmoid().detach(),
                    B * M,
                )
            )
            losses[f"loss_mask@{i}"] = (
                0.1
                * self.weight_dict["loss_mask"]
                * sigmoid_ce_loss(
                    online["pred_masks"].reshape(B * M, H * W),
                    target["pred_masks"].reshape(B * M, H * W).sigmoid().detach(),
                    B * M,
                )
            )
            losses[f"loss_iou@{i}"] = (
                0.1
                * self.weight_dict["loss_iou"]
                * F.smooth_l1_loss(
                    online["pred_ious"],
                    target["pred_ious"].detach(),
                )
            )

        return losses


@CRITERION_REGISTRY.register()
class MaskClsCriterion(nn.Module):
    @configurable
    def __init__(self, matcher: nn.Module, loss_weight: float):
        super().__init__()
        self.matcher = matcher
        self.loss_weight = loss_weight

    @classmethod
    def from_config(cls, cfg):
        loss_weight = cfg.MODEL.OVSEG.ALIGN_WEIGHT
        matcher = One2OneHungarianMatcher(
            cost_mask=cfg.MODEL.OVSEG.MASK_WEIGHT,
            cost_dice=cfg.MODEL.OVSEG.DICE_WEIGHT,
            num_points=cfg.MODEL.OVSEG.MATCHER_NUM_POINTS,
            thres_pos=cfg.MODEL.OVSEG.MATCHER_THRES_POS,
        )
        return {
            "matcher": matcher,
            "loss_weight": loss_weight,
        }

    def _get_src_permutation_idx(self, indices):
        # permute predictions following indices
        batch_idx = torch.cat([torch.full_like(src, i) for i, (src, _) in enumerate(indices)])
        src_idx = torch.cat([src for (src, _) in indices])
        return batch_idx, src_idx

    def _get_tgt_permutation_idx(self, indices):
        # permute targets following indices
        batch_idx = torch.cat([torch.full_like(tgt, i) for i, (_, tgt) in enumerate(indices)])
        tgt_idx = torch.cat([tgt for (_, tgt) in indices])
        return batch_idx, tgt_idx

    def forward(self, outputs, targets):
        assert len(targets) == 1, "only batchsize=1"
        src_idx, tgt_idx = self.matcher(outputs, targets)[0]

        num_masks = len(tgt_idx)
        num_masks = torch.as_tensor([num_masks], dtype=torch.float, device=next(iter(outputs.values())).device)
        if is_dist_avail_and_initialized():
            torch.distributed.all_reduce(num_masks)
        num_masks = torch.clamp(num_masks / get_world_size(), min=1).item()

        # classification
        src_logits = outputs["pred_logits"][0, src_idx].float()
        tgt_logits = targets[0]["labels"][tgt_idx]
        loss_align = F.cross_entropy(src_logits, tgt_logits, reduction="sum") / num_masks
        return {"loss_align": loss_align * self.loss_weight}
