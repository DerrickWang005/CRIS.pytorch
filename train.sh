#!/usr/bin/env bash
set -x


export TORCH_DISTRIBUTED_DEBUG=INFO
export NCCL_DEBUG=INFO

# Train stage 1
# export CUDA_VISIBLE_DEVICES=0,1,2,3
# export CUDA_VISIBLE_DEVICES=4,5,6,7
python train_net.py \
    --num-gpus 1 \
    --num-machines 1 \
    --machine-rank 0 \
    --config-file configs/base.yaml \
    SOLVER.IMS_PER_BATCH 2 \
    OUTPUT_DIR ./outputs/debug


# python train_net.py \
#     --num-gpus 8 \
#     --num-machines 1 \
#     --machine-rank 0 \
#     --config-file configs/base.yaml \
#     --eval-only \
#     SOLVER.IMS_PER_BATCH 64 \
#     OUTPUT_DIR ./results/debug \
#     MODEL.WEIGHTS outputs/debug/model_0009999.pth
