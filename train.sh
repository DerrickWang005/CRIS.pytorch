#!/usr/bin/env bash
set -x


export TORCH_DISTRIBUTED_DEBUG=INFO
export DETECTRON2_DATASETS=/vepfs/home/wangzhaoqing/uni-ovseg/data/


# Train stage 1
# export CUDA_VISIBLE_DEVICES=4,5,6,7
# python train_net.py \
#     --num-gpus 2 \
#     --num-machines 1 \
#     --machine-rank 0 \
#     --config-file configs/base.yaml \
#     SOLVER.IMS_PER_BATCH 64 \
#     OUTPUT_DIR ./outputs/debug


python train_net.py \
    --num-gpus 8 \
    --num-machines 1 \
    --machine-rank 0 \
    --config-file configs/base.yaml \
    --eval-only \
    SOLVER.IMS_PER_BATCH 64 \
    OUTPUT_DIR ./results/debug \
    # MODEL.WEIGHTS ${CKPT}
