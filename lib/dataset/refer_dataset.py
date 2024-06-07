import logging
import os
import os.path as osp
import random

import numpy as np
import torch
from torch.nn import functional as F
from torch.utils.data import Dataset

from detectron2.config import configurable
from detectron2.data import detection_utils as utils
from detectron2.data import transforms as T
from detectron2.projects.point_rend import ColorAugSSDTransform
from detectron2.structures import Instances
from detectron2.utils.registry import Registry

from .refer import REFER

DATASET_REGISTRY = Registry("DATASETS")
DATASET_REGISTRY.__doc__ = """
Registry for datasets.
"""


def build_datasets(cfg):
    """
    Build a instance embedding branch from `cfg.MODEL.INS_EMBED_HEAD.NAME`.
    """
    name = cfg.INPUT.DATASET_NAME
    return DATASET_REGISTRY.get(name)(cfg)


@DATASET_REGISTRY.register()
class ReferDataset(Dataset):
    @configurable
    def __init__(
        self,
        data_root: str,
        dataset_name: str,
        refer_split: str,
        data_split: str,
        augmentations,
        image_format: str,
        ignore_label: int,
        pos_repeat: int,
    ):
        super().__init__()
        logger = logging.getLogger(__name__)
        self.refer = REFER(data_root=data_root, dataset=dataset_name, splitBy=refer_split)
        self.ref_ids = self.refer.getRefIds(split=data_split)
        logger.info(f"There are {len(self.ref_ids)} training referred objects")
        self.data_root = data_root
        self.augmentations = augmentations
        self.image_format = image_format
        self.ignore_label = ignore_label
        self.pos_repeat = pos_repeat
        logger.info(f"Augmentation: {augmentations}")

    @classmethod
    def from_config(cls, cfg):
        # build augmentation
        augs = [
            T.ResizeShortestEdge(short_edge_length=cfg.INPUT.CROP_SIZE, max_size=cfg.INPUT.CROP_SIZE),
            T.FixedSizeCrop(
                crop_size=(cfg.INPUT.CROP_SIZE, cfg.INPUT.CROP_SIZE),
                pad_value=128,
                seg_pad_value=cfg.INPUT.IGNORE_LABEL,
            ),
            ColorAugSSDTransform(img_format=cfg.INPUT.FORMAT),
        ]
        # Assume always applies to the training set.
        data_root = cfg.INPUT.TRAIN_ROOT
        dataset_name = cfg.INPUT.TRAIN_NAME
        refer_split = cfg.INPUT.REFER_SPLIT
        data_split = cfg.INPUT.DATA_SPLIT
        image_format = cfg.INPUT.FORMAT
        ignore_label = cfg.INPUT.IGNORE_LABEL
        pos_repeat = cfg.INPUT.POS_REPEAT

        return {
            "data_root": data_root,
            "dataset_name": dataset_name,
            "refer_split": refer_split,
            "data_split": data_split,
            "augmentations": augs,
            "image_format": image_format,
            "ignore_label": ignore_label,
            "pos_repeat": pos_repeat,
        }

    def __len__(self):
        return len(self.ref_ids)

    def __getitem__(self, index):
        ref_id = self.ref_ids[index]
        ref = self.refer.loadRefs(ref_id)[0]

        # load image
        file_name = self.refer.loadImgs(ref["image_id"])[0]["file_name"]
        file_name = osp.join(self.data_root, "train2014", file_name)
        image = utils.read_image(file_name, format=self.image_format)

        # load mask
        mask_ori = self.refer.getMask(ref)["mask"]

        # load sentence
        sentence = random.choice(ref["sentences"])["raw"]

        # transform image and mask
        aug_input = T.AugInput(image, sem_seg=mask_ori)
        aug_input, transforms = T.apply_transform_gens(self.augmentations, aug_input)
        image = aug_input.image
        mask = aug_input.sem_seg

        image = torch.as_tensor(np.ascontiguousarray(image.transpose(2, 0, 1)))
        mask_ori = torch.as_tensor(mask_ori.astype("long")).unsqueeze(0)
        mask = torch.as_tensor(mask.astype("long")).unsqueeze(0)

        data_dict = dict()
        data_dict["image"] = image
        data_dict["image_shape"] = mask_ori.shape[-2:]  # h, w
        data_dict["gt_mask"] = mask.repeat(self.pos_repeat, 1, 1)
        data_dict["gt_mask_ori"] = mask_ori.repeat(self.pos_repeat, 1, 1)
        data_dict["gt_class"] = torch.tensor([1] * self.pos_repeat, dtype=torch.int64)
        data_dict["sentence"] = sentence
        data_dict["filename"] = os.path.basename(file_name)

        return data_dict
