import os
import time

import cv2
import numpy as np
import torch
import torch.cuda.amp as amp
import torch.distributed as dist
import torch.nn.functional as F
import wandb
from loguru import logger
from tqdm import tqdm
from utils.dataset import tokenize
from utils.misc import AverageMeter, ProgressMeter, concat_all_gather, trainMetricGPU


def getTransformMat(input_size, img_size, inverse=False):
    ori_h, ori_w = img_size
    inp_h, inp_w = input_size
    scale = min(inp_h / ori_h, inp_w / ori_w)
    new_h, new_w = ori_h * scale, ori_w * scale
    bias_x, bias_y = (inp_w - new_w) / 2.0, (inp_h - new_h) / 2.0
    src = np.array([[0, 0], [ori_w, 0], [0, ori_h]], np.float32)
    dst = np.array(
        [[bias_x, bias_y], [new_w + bias_x, bias_y], [bias_x, new_h + bias_y]],
        np.float32,
    )
    mat = cv2.getAffineTransform(src, dst)
    if inverse:
        mat_inv = cv2.getAffineTransform(dst, src)
        return mat, mat_inv
    return mat, None


def convert(img, mask=None):
    mean = torch.tensor([0.48145466, 0.4578275, 0.40821073]).reshape(3, 1, 1)
    std = torch.tensor([0.26862954, 0.26130258, 0.27577711]).reshape(3, 1, 1)
    # Image ToTensor & Normalize
    img = torch.from_numpy(img.transpose((2, 0, 1)))
    if not isinstance(img, torch.FloatTensor):
        img = img.float()
    img.div_(255.0).sub_(mean).div_(std)
    # Mask ToTensor
    if mask is not None:
        mask = torch.from_numpy(mask)
        if not isinstance(mask, torch.FloatTensor):
            mask = mask.float()
    return img, mask


@torch.no_grad()
def inference(img_path, txt, model, args):

    model.eval()
    # data
    ori_img = np.array(cv2.imread(img_path))
    img = cv2.cvtColor(ori_img, cv2.COLOR_BGR2RGB)
    send_img = img
    img_size = img.shape[:2]

    mat, mat_inv = getTransformMat((args.input_size, args.input_size), img_size, True)
    img = cv2.warpAffine(
        img,
        mat,
        (args.input_size, args.input_size),
        flags=cv2.INTER_CUBIC,
        borderValue=[0.48145466 * 255, 0.4578275 * 255, 0.40821073 * 255],
    )

    img = convert(img)[0]
    param = {
        "ori_img": np.expand_dims(ori_img, axis=0),
        # 'seg_id': seg_id,
        # 'mask_dir': mask_dir,
        "inverse": np.expand_dims(mat_inv, axis=0),
        "ori_size": np.expand_dims(np.array(img_size), axis=0),
        # 'sents': sents
    }

    # img = np.transpose(img, [2, 0, 1])
    # img = np.expand_dims(img, axis=0)
    # # Convert the image to Tensor
    # img = torch.tensor(img)
    img = torch.unsqueeze(img, dim=0)
    img = img.cuda(non_blocking=True)

    text = tokenize(txt, args.word_len, True)
    text = text.cuda(non_blocking=True)

    # inference
    pred = model(img, text)

    pred = torch.sigmoid(pred)
    if pred.shape[-2:] != img.shape[-2:]:
        pred = F.interpolate(
            pred, size=img.shape[-2:], mode="bicubic", align_corners=True
        ).squeeze()
    # process one sentence
    h, w = param["ori_size"][0]
    mat = param["inverse"][0]
    pred = pred.cpu().numpy()
    pred = cv2.warpAffine(pred, mat, (w, h), flags=cv2.INTER_CUBIC, borderValue=0.0)
    pred = np.array(pred > 0.35)
    # print(img.shape, pred.shape)
    return send_img, txt, pred
    # # iou
    # inter = np.logical_and(pred, mask)
    # union = np.logical_or(pred, mask)
    # iou = np.sum(inter) / (np.sum(union) + 1e-6)
    # iou_list.append(iou)
    # # dump prediction
    # if args.visualize:
    #     pred = np.array(pred * 255, dtype=np.uint8)
    #     sent = "_".join(sent[0].split(" "))
    #     pred_name = "{}-iou={:.2f}-{}.png".format(seg_id, iou * 100, sent)
    #     cv2.imwrite(filename=os.path.join(args.vis_dir, pred_name), img=pred)
    # logger.info("=> Metric Calculation <=")
    # iou_list = np.stack(iou_list)
    # iou_list = torch.from_numpy(iou_list).to(img.device)
    # prec_list = []
    # for thres in torch.arange(0.5, 1.0, 0.1):
    #     tmp = (iou_list > thres).float().mean()
    #     prec_list.append(tmp)
    # iou = iou_list.mean()
    # prec = {}
    # for i, thres in enumerate(range(5, 10)):
    #     key = "Pr@{}".format(thres * 10)
    #     value = prec_list[i].item()
    #     prec[key] = value
    # logger.info("IoU={:.2f}".format(100.0 * iou.item()))
    # for k, v in prec.items():
    #     logger.info("{}: {:.2f}.".format(k, 100.0 * v))

    # return iou.item(), prec
