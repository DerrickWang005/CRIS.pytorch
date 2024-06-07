import itertools
import logging
import os
from collections import OrderedDict

import numpy as np
import torch
import torch.nn.functional as F
import torchshow as ts
from torchvision.utils import draw_segmentation_masks, make_grid, save_image

from detectron2.evaluation.evaluator import DatasetEvaluator
from detectron2.utils.comm import all_gather, is_main_process, synchronize
from detectron2.utils.file_io import PathManager


def computeIoU(pred_seg, gd_seg):
    I = np.sum(np.logical_and(pred_seg, gd_seg))
    U = np.sum(np.logical_or(pred_seg, gd_seg))

    return I, U


class ReferEvaluator(DatasetEvaluator):
    def __init__(
        self,
        dataset_name,
        distributed=True,
        output_dir=None,
        save_imgs=False,
    ):
        self._logger = logging.getLogger(__name__)
        self._dataset_name = dataset_name
        self._distributed = distributed
        self._output_dir = output_dir
        self._vis_dir = os.path.join(output_dir, "vis")
        os.makedirs(self._vis_dir, exist_ok=True)
        self._save_imgs = save_imgs

        self._cpu_device = torch.device("cpu")
        # self._available_sources = ["refcoco", "grefcoco"]
        self._available_sources = ["refcoco"]
        self._num_classes = 2

    def reset(self):
        self._predictions = []

    def process(self, inputs, outputs):
        for input, output in zip(inputs, outputs):
            gt_mask = input["gt_mask_ori"][0].to(self._cpu_device)
            gt_mask = np.array(gt_mask, dtype=np.int8)
            pred_mask = output["refer_seg"].to(self._cpu_device)
            pred_mask = np.array(pred_mask, dtype=np.int8)
            image = input["image"]
            max_side = max(gt_mask.shape[-2], gt_mask.shape[-1])
            image = F.interpolate(image.unsqueeze(0), size=(max_side, max_side), mode="bilinear", align_corners=False)
            image = image[0, :, : gt_mask.shape[-2], : gt_mask.shape[-1]].to(self._cpu_device)
            image = np.array(image, dtype=np.uint8)

            # save results
            if self._save_imgs:
                rst_pred = draw_segmentation_masks(
                    torch.from_numpy(image), torch.from_numpy(pred_mask).bool(), alpha=0.8, colors="green"
                )
                rst_gt = draw_segmentation_masks(
                    torch.from_numpy(image), torch.from_numpy(gt_mask).bool(), alpha=0.8, colors="blue"
                )
                rst = make_grid([rst_gt, rst_pred], nrow=2, padding=5)
                # import pdb; pdb.set_trace()
                ts.save(rst, os.path.join(self._vis_dir, input["filename"]))

            self._predictions.append(
                {
                    "image": image,
                    "gt_mask": gt_mask,
                    "pred_mask": pred_mask,
                    "sentence": input["sentence"],
                    "source": "refcoco",
                }
            )

    def evaluate(self):
        if self._distributed:
            synchronize()
            predictions = all_gather(self._predictions)
            predictions = list(itertools.chain(*predictions))
            if not is_main_process():
                return
        else:
            predictions = self._predictions

        if self._output_dir and self._save_imgs:
            PathManager.mkdirs(self._output_dir)
            file_path = os.path.join(self._output_dir, "ref_seg_predictions.pth")
            self._logger.info(f"Saving output images to {file_path} ...")
            with PathManager.open(file_path, "wb") as f:
                torch.save(predictions, f)

        accum_I = 0
        accum_U = 0
        accum_IoU = 0
        pr_thres = [0.5, 0.6, 0.7, 0.8, 0.9]
        pr_count = {pr: 0 for pr in pr_thres}
        total_count = 0

        results_dict = []
        for eval_sample in predictions:
            ref_result = {}
            # ref_result['image'] = eval_sample['image']
            # ref_result['pred_mask'] = eval_sample['pred_mask']
            # ref_result['sentence'] = eval_sample['sentence']
            # ref_result['source'] = src
            I, U = computeIoU(eval_sample["pred_mask"], eval_sample["gt_mask"])
            this_iou = float(0) if U == 0 else float(I) / float(U)
            accum_IoU += this_iou
            accum_I += I
            accum_U += U
            total_count += 1
            for thres in pr_thres:
                if this_iou >= thres:
                    pr_count[thres] += 1
            ref_result["I"] = int(I)
            ref_result["U"] = int(U)
            ref_result["IoU"] = float(this_iou)
        results_dict.append(ref_result)

        results = OrderedDict()
        results["gIoU"] = 100.0 * accum_IoU / total_count
        results["cIoU"] = 100.0 * accum_I / accum_U
        for thres in pr_thres:
            pr_name = "Pr@{0:1.1f}".format(thres)
            results[pr_name] = pr_count[thres] * 100.0 / total_count

        self._logger.info(results)
        return results
