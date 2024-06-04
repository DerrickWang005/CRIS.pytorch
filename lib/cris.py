import logging
import os
from typing import Dict, List, Tuple

import torch
import torchshow as ts
from torch import nn
from torch.nn import functional as F
from tqdm import trange

from detectron2.config import configurable
from detectron2.data import MetadataCatalog
from detectron2.modeling import META_ARCH_REGISTRY
from detectron2.modeling.backbone import Backbone
from detectron2.structures import ImageList
from detectron2.utils.comm import is_main_process
from detectron2.utils.memory import retry_if_cuda_oom

from .backbone import build_backbone
from .pixel_decoder import build_pixel_decoder
from .mask_decoder import build_transformer_decoder
from .criterion import build_criterion

logger = logging.getLogger(__name__)


@META_ARCH_REGISTRY.register()
class CRIS(nn.Module):
    @configurable
    def __init__(
        self,
        *,
        backbone: Backbone,
        pixel_decoder: nn.Module,
        mask_decoder: nn.Module,
        criterion: nn.Module,
        size_divisibility: int,
        sem_seg_postprocess_before_inference: bool,
        pixel_mean: Tuple[float],
        pixel_std: Tuple[float],
        conf_threshold: float = 0.5,
    ):
        """
        Args:
            backbone: a backbone module, must follow detectron2's backbone interface
            sem_seg_head: a module that predicts semantic segmentation from backbone features
            criterion: a module that defines the loss
            num_queries: int, number of queries
            object_mask_threshold: float, threshold to filter query based on classification score
                for panoptic segmentation inference
            overlap_threshold: overlap threshold used in general inference for panoptic segmentation
            metadata: dataset meta, get `thing` and `stuff` category names for panoptic
                segmentation inference
            size_divisibility: Some backbones require the input height and width to be divisible by a
                specific integer. We can use this to override such requirement.
            sem_seg_postprocess_before_inference: whether to resize the prediction back
                to original input size before semantic segmentation inference or after.
                For high-resolution dataset like Mapillary, resizing predictions before
                inference will cause OOM error.
            pixel_mean, pixel_std: list or tuple with #channels element, representing
                the per-channel mean and std to be used to normalize the input image
            semantic_on: bool, whether to output semantic segmentation prediction
            instance_on: bool, whether to output instance segmentation prediction
            panoptic_on: bool, whether to output panoptic segmentation prediction
            test_topk_per_image: int, instance segmentation parameter, keep topk instances per image
        """
        super().__init__()
        # architecture
        self.backbone = backbone
        self.pixel_decoder = pixel_decoder
        self.mask_decoder = mask_decoder

        # loss function
        self.criterion = criterion

        # utils
        if size_divisibility < 0:
            # use backbone size_divisibility if not set
            size_divisibility = self.backbone.size_divisibility
        self.size_divisibility = size_divisibility

        self.sem_seg_postprocess_before_inference = sem_seg_postprocess_before_inference
        self.conf_threshold = conf_threshold

        # image statistics
        self.register_buffer(
            name="pixel_mean",
            tensor=torch.Tensor(pixel_mean).view(-1, 1, 1),
            persistent=False,
        )
        self.register_buffer(
            name="pixel_std",
            tensor=torch.Tensor(pixel_std).view(-1, 1, 1),
            persistent=False,
        )

    @classmethod
    def from_config(cls, cfg):
        backbone = build_backbone(cfg)
        pixel_decoder = build_pixel_decoder(cfg, backbone.output_shape())
        mask_decoder = build_transformer_decoder(cfg)
        criterion = build_criterion(cfg, cfg.MODEL.CRIS.CRITERION_SEG)
        return {
            "backbone": backbone,
            "pixel_decoder": pixel_decoder,
            "mask_decoder": mask_decoder,
            "criterion": criterion,
            "size_divisibility": cfg.MODEL.OVSEG.SIZE_DIVISIBILITY,
            "sem_seg_postprocess_before_inference": cfg.MODEL.OVSEG.SEM_SEG_POSTPROCESSING_BEFORE_INFERENCE,
            "pixel_mean": cfg.MODEL.PIXEL_MEAN,
            "pixel_std": cfg.MODEL.PIXEL_STD,
            # inference
            "conf_threshold": cfg.MODEL.CRIS.TEST.CONF_THRESHOLD,
        }

    @property
    def device(self):
        return self.pixel_mean.device

    def preprosses_image(self, batched_inputs):
        images = [x["image"].to(self.device) for x in batched_inputs]
        images = [(x - self.pixel_mean) / self.pixel_std for x in images]
        images = ImageList.from_tensors(images, self.size_divisibility)
        return images

    def prepare_refer_targets(self, targets, points):
        h_pad, w_pad = self.input_size
        new_targets = []
        for targets_per_image, point in zip(targets, points):
            # pad gt
            gt_masks = targets_per_image.gt_masks
            padded_masks = torch.zeros(
                (gt_masks.shape[0], h_pad, w_pad),
                dtype=gt_masks.dtype,
                device=gt_masks.device,
            )
            padded_masks[:, : gt_masks.shape[1], : gt_masks.shape[2]] = gt_masks
            new_targets.append(
                {
                    "masks": padded_masks,
                    "points": (point[:, 2:] + point[:, :2]) / 2,
                }
            )
        return new_targets

    def semantic_inference(self, mask_cls, mask_pred):
        mask_pred = mask_pred.sigmoid()
        semseg = torch.einsum("qc,qhw->chw", mask_cls, mask_pred)
        return semseg

    def refer_train_forward(
        self,
        gt_instances: List,
        features: Dict,
    ):
        boxes, boxes_ids = [], []
        for intance in gt_instances:
            if len(intance.gt_boxes) < max_boxes:
                ids = torch.arange(len(intance.gt_boxes), device=self.device)
                ids = torch.cat([ids] * max_boxes, dim=0)[:max_boxes]
            else:
                ids = torch.randperm(len(intance.gt_boxes), device=self.device)[:max_boxes]
            boxes.append(intance.gt_boxes[ids])
            boxes_ids.append(ids)
        targets = self.prepare_box_targets(gt_instances, boxes_ids)
        (mask_features, transformer_encoder_features, multi_scale_features,) = self.sem_seg_head[
            "pixel_decoder"
        ].forward_features(features)
        outputs = self.sem_seg_head["predictor"](
            multi_scale_features,
            mask_features,
            points=None,
            boxes=boxes,
            points_multi=None,
        )
        losses = self.criterion_seg(outputs, targets, ["masks_paired", "ious_paired"])
        for k in list(losses.keys()):
            if k in self.criterion_seg.weight_dict:
                losses[k] *= self.criterion_seg.weight_dict[k]
            else:
                # remove this loss if not specified in `weight_dict`
                losses.pop(k)
        return losses

    def refer_test_forward(
        self,
        features: Dict,
        scaled_sizes: torch.Tensor,
        output_sizes: torch.Tensor,
    ):
        points, offset_x, offset_y = self.prepare_base_points(self.pts_per_side_test, self.device)
        points = self.augment_points(points, (offset_x, offset_y), scaled_sizes, False)
        points = torch.split(points.squeeze(0), 128, dim=0)

        (mask_features, transformer_encoder_features, multi_scale_features,) = self.sem_seg_head[
            "pixel_decoder"
        ].forward_features(features)

        mask_pred_results = []
        iou_pred_results = []
        for point in points:
            output = self.sem_seg_head["predictor"](
                multi_scale_features,
                mask_features,
                points=[point],
                boxes=None,
                points_multi=None,
            )
            mask_pred_results.append(output["pred_masks"].flatten(1, 2))
            iou_pred_results.append(output["pred_ious"].flatten(1, 2))
        del points, point, output
        mask_pred_results = torch.cat(mask_pred_results, dim=1)
        iou_pred_results = torch.cat(iou_pred_results, dim=1)

        # CLIP-driven mask classifier
        clip_embeds = self.backbone.visual_prediction_forward(
            self.mask_pooling(features["clip_vis_dense"], mask_pred_results)
        )  # N, Q', dim
        mask_cls_results = get_classification_logits_fcclip(
            clip_embeds,
            self.test_text_classifier,
            self.backbone.clip_model.logit_scale,
            self.test_num_templates,
        )
        mask_cls_results = mask_cls_results.softmax(-1)  # N, Q', T'

        processed_results = []
        for mask_cls_result, mask_pred_result, iou_pred_result, output_size in zip(
            mask_cls_results,
            mask_pred_results,
            iou_pred_results,
            output_sizes,
        ):
            processed_results.append({})
            if self.sem_seg_postprocess_before_inference:
                mask_pred_result = retry_if_cuda_oom(sem_seg_postprocess)(
                    mask_pred_result, self.input_size, output_size
                )
                mask_cls_result = mask_cls_result.to(mask_pred_result)

            # semantic segmentation inference
            if self.semantic_on:
                r = retry_if_cuda_oom(self.semantic_inference)(mask_cls_result, mask_pred_result)
                if not self.sem_seg_postprocess_before_inference:
                    r = retry_if_cuda_oom(sem_seg_postprocess)(r, self.input_size, output_size)
                processed_results[-1]["sem_seg"] = r

            # panoptic segmentation inference
            if self.panoptic_on:
                panoptic_r = retry_if_cuda_oom(self.panoptic_inference)(
                    mask_cls_result, mask_pred_result, iou_pred_result
                )
                processed_results[-1]["panoptic_seg"] = panoptic_r

        torch.cuda.empty_cache()
        return processed_results

    def forward(self, batched_inputs):
        """
        Args:
            batched_inputs: a list, batched outputs of :class:`DatasetMapper`.
                Each item in the list contains the inputs for one image.
                For now, each item in the list is a dict that contains:
                   * "image": Tensor, image in (C, H, W) format.
                   * "instances": per-region ground truth
                   * Other information that's included in the original dicts, such as:
                     "height", "width" (int): the output resolution of the model (may be different
                     from input resolution), used in inference.
        Returns:
            list[dict]:
                each dict has the results for one image. The dict contains the following keys:

                * "sem_seg":
                    A Tensor that represents the
                    per-pixel segmentation prediced by the head.
                    The prediction has shape KxHxW that represents the logits of
                    each class for each pixel.
                * "panoptic_seg":
                    A tuple that represent panoptic output
                    panoptic_seg (Tensor): of shape (height, width) where the values are ids for each segment.
                    segments_info (list[dict]): Describe each segment in `panoptic_seg`.
                        Each dict contains keys "id", "category_id", "isthing".
        """
        images = self.preprosses_image(batched_inputs)

        features = self.backbone(images.tensor)
        if self.training:
            gt_instances = [x["instances"].to(self.device) for x in batched_inputs]
            losses = self.box_train_forward(gt_instances, features)
            return losses
        else:
            processed_results = self.ov_test_forward(features)
            torch.cuda.empty_cache()
            return processed_results
