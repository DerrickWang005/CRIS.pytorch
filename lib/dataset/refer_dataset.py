import logging
from .refer import REFER
from torch.utils.data import Dataset, DataLoader
from detectron2.config import configurable
from detectron2.data import transforms as T


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
        size_divisibility: int,
    ):
        super().__init__()
        logger = logging.getLogger(__name__)
        refer = REFER(data_root=data_root, dataset=dataset_name, splitBy=refer_split)
        self.ref_ids = refer.getRefIds(split=data_split)
        logger.info(f"There are {len(self.ref_ids)} training referred objects")

        self.augmentations = augmentations
        self.image_format = image_format
        self.ignore_label = ignore_label
        self.size_divisibility = size_divisibility
        logger.info("Augmentation: ", augmentations)

    @classmethod
    def from_config(cls, cfg):
        # build augmentation
        augs = [
            T.ResizeShortestEdge(
                short_edge_length=cfg.INPUT.CROP_SIZE,
                max_size=cfg.INPUT.CROP_SIZE
            ),
            T.ColorAugSSDTransform(
                img_format=cfg.INPUT.FORMAT
            )
        ]
        # Assume always applies to the training set.
        data_root = cfg.DATASETS.TRAIN_ROOT
        dataset_name = cfg.DATASETS.TRAIN_NAME
        refer_split = cfg.DATASETS.REFER_SPLIT
        data_split = cfg.DATASETS.DATA_SPLIT
        ignore_label = cfg.DATASETS.IGNORE_LABEL

        return {
            "data_root": data_root,
            "dataset_name": dataset_name,
            "refer_split": refer_split,
            "data_split": data_split,
            "augmentations": augs,
            "image_format": cfg.INPUT.FORMAT,
            "ignore_label": ignore_label,
            "size_divisibility": cfg.INPUT.SIZE_DIVISIBILITY,
        }

    def __getitem__(self, index):
        info = self.ref_ids[index]
