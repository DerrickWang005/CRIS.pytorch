import logging
from typing import Dict, List, Tuple

import torch
from torch import nn
from torch.nn import functional as F

# import torchshow as ts
from detectron2.config import configurable
from detectron2.modeling import META_ARCH_REGISTRY
from detectron2.modeling.backbone import Backbone
from detectron2.structures import ImageList
from detectron2.utils.memory import retry_if_cuda_oom

from .backbone import build_backbone
from .criterion import build_criterion
from .mask_decoder import build_transformer_decoder
from .pixel_decoder import build_pixel_decoder

logger = logging.getLogger(__name__)


def sem_seg_postprocess(result, output_height, output_width):
    max_side = max(output_height, output_width)
    result = F.interpolate(
        result.unsqueeze(0),
        size=(max_side, max_side),
        mode="bilinear",
        align_corners=False,
    )[0]
    result = result[:, :output_height, :output_width]
    return result


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
        pixel_mean: Tuple[float],
        pixel_std: Tuple[float],
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
        criterion = build_criterion(cfg, cfg.MODEL.CRIS.CRITERION)
        return {
            "backbone": backbone,
            "pixel_decoder": pixel_decoder,
            "mask_decoder": mask_decoder,
            "criterion": criterion,
            "pixel_mean": cfg.MODEL.PIXEL_MEAN,
            "pixel_std": cfg.MODEL.PIXEL_STD,
        }

    @property
    def device(self):
        return self.pixel_mean.device

    def preprosses_image(self, batched_inputs):
        images = [x["image"].to(self.device) for x in batched_inputs]
        images = [(x - self.pixel_mean) / self.pixel_std for x in images]
        images = torch.stack(images, dim=0)
        return images

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
        # gt_instances = [x["instances"] for x in batched_inputs]
        features = self.backbone.extract_visual_features(images)
        if self.training:
            losses = self.refer_train_forward(batched_inputs, features)
            return losses
        else:
            processed_results = self.refer_test_forward(images, batched_inputs, features)
            return processed_results

    def refer_train_forward(
        self,
        batched_inputs: List,
        features: Dict,
    ):
        targets, sentences = [], []
        for batch in batched_inputs:
            targets.append(
                {
                    "masks": batch["gt_mask"].to(self.device),
                    "labels": batch["gt_class"].to(self.device),
                }
            )
            sentences.append(batch["sentence"])

        # encode sentences
        y_sent, y_word, y_pad_mask = self.backbone.extract_text_features(sentences, self.device)

        # encode pixel features
        (
            mask_features,
            transformer_encoder_features,
            multi_scale_features,
        ) = self.pixel_decoder.forward_features(features, y_word, y_pad_mask)
        outputs = self.mask_decoder(
            multi_scale_features,
            y_word,
            y_sent,
            y_pad_mask,
            mask_features,
        )
        losses = self.criterion(outputs, targets)
        for k in list(losses.keys()):
            if k in self.criterion.weight_dict:
                losses[k] *= self.criterion.weight_dict[k]
            else:
                # remove this loss if not specified in `weight_dict`
                losses.pop(k)
        return losses

    def refer_test_forward(
        self,
        images: torch.Tensor,
        batched_inputs: List,
        features: Dict,
    ):
        targets, sentences, output_shapes = [], [], []
        for batch in batched_inputs:
            targets.append(
                {
                    "masks_ori": batch["gt_mask_ori"].to(self.device),  # for evaluation
                    "labels": batch["gt_class"].to(self.device),
                }
            )
            sentences.append(batch["sentence"])
            output_shapes.append(batch["image_shape"])

        # encode sentences
        y_sent, y_word, y_pad_mask = self.backbone.extract_text_features(sentences, self.device)

        # encode pixel features
        (
            mask_features,
            transformer_encoder_features,
            multi_scale_features,
        ) = self.pixel_decoder.forward_features(features, y_word, y_pad_mask)
        outputs = self.mask_decoder(
            multi_scale_features,
            y_word,
            y_sent,
            y_pad_mask,
            mask_features,
        )
        mask_cls_results = outputs["pred_logits"]
        mask_pred_results = outputs["pred_masks"]

        # upsample masks
        mask_pred_results = F.interpolate(
            mask_pred_results,
            size=images.shape[-2:],
            mode="bilinear",
            align_corners=False,
        )

        processed_results = []
        for mask_cls_result, mask_pred_result, output_shape in zip(mask_cls_results, mask_pred_results, output_shapes):
            processed_results.append({})
            mask_pred_result = retry_if_cuda_oom(sem_seg_postprocess)(
                mask_pred_result, output_shape[0], output_shape[1]
            )
            mask_cls_result = mask_cls_result.to(mask_pred_result)
            r = retry_if_cuda_oom(self.semantic_inference)(mask_cls_result, mask_pred_result)
            processed_results[-1]["refer_seg"] = r

        torch.cuda.empty_cache()
        return processed_results

    def semantic_inference(self, mask_cls, mask_pred):
        mask_cls = mask_cls.sigmoid().ge(0.5)
        mask_pred = mask_pred.sigmoid()
        semseg = mask_pred[mask_cls].mean(0).ge(0.5)
        return semseg
