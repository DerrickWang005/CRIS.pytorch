import argparse
import os
import warnings

import cv2
import torch
import torch.nn.parallel
import torch.utils.data
from loguru import logger

import utils.config as config
from engine.engine_single_img import inference
from model import build_segmenter
from utils.dataset import RefDataset
from utils.misc import setup_logger

warnings.filterwarnings("ignore")
cv2.setNumThreads(0)


def get_parser():
    # parser = argparse.ArgumentParser(
    #     description="Pytorch Referring Expression Segmentation"
    # )
    # parser.add_argument(
    #     "--config",
    #     default="/mnt/Enterprise/miccai_2023_CRIS/CRIS.pytorch/config/refcoco/cris_r50.yaml",
    #     type=str,
    #     help="config file",
    # )
    # parser.add_argument(
    #     "--opts",
    #     default=None,
    #     nargs=argparse.REMAINDER,
    #     help="override some settings in the config.",
    # )
    # parser.add_argument("--img_path", default=None, required=True, type=str)
    # parser.add_argument("--txt", default=None, required=True, type=str)

    # args = parser.parse_args()
    # assert args.config is not None
    cfg = config.load_cfg_from_cfg_file(
        "/mnt/Enterprise/miccai_2023_CRIS/CRIS.pytorch/config/cris_r50.yaml"
    )
    # if args.opts is not None:
    #     cfg = config.merge_cfg_from_list(cfg, args.opts)
    return cfg


@logger.catch
def get_inferer(model_path=None):
    args = get_parser()
    args.output_dir = os.path.join(args.output_folder, args.exp_name)
    if args.visualize:
        args.vis_dir = os.path.join(args.output_dir, "vis")
        os.makedirs(args.vis_dir, exist_ok=True)

    # logger
    setup_logger(args.output_dir, distributed_rank=0, filename="test.log", mode="a")
    # logger.info(args)

    # build dataset & dataloader
    # test_data = RefDataset(
    #     lmdb_dir=args.test_lmdb,
    #     mask_dir=args.mask_root,
    #     dataset=args.dataset,
    #     split=args.test_split,
    #     mode="test",
    #     input_size=args.input_size,
    #     word_length=args.word_len,
    # )
    # test_loader = torch.utils.data.DataLoader(
    #     test_data, batch_size=1, shuffle=False, num_workers=1, pin_memory=True
    # )

    # build model
    model, _ = build_segmenter(args)
    model = torch.nn.DataParallel(model)
    # logger.info(model)

    if not model_path == None:
        args.model_dir = model_path
    else:
        args.model_dir = os.path.join(args.output_dir, "best_model.pth")

    if os.path.isfile(args.model_dir):
        logger.info("=> loading checkpoint '{}'".format(args.model_dir))
        checkpoint = torch.load(args.model_dir)
        model.load_state_dict(checkpoint["state_dict"], strict=True)
        logger.info("=> loaded checkpoint '{}'".format(args.model_dir))
    else:
        raise ValueError(
            "=> resume failed! no checkpoint found at '{}'. Please check args.resume again!".format(
                args.model_dir
            )
        )

    model.eval().cuda()

    def infer(img_path, txt):
        return inference(img_path, txt, model, args)

    return infer
