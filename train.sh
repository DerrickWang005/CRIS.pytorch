#!/usr/bin/env bash
set -x

# pip install -e detectron2
# pip install torchshow
# cd lib/pixel_decoder/ops/
# sh make.sh
# cd ../../..

export TORCH_DISTRIBUTED_DEBUG=INFO

# Train stage 1
python train_net.py \
    --num-gpus 8 \
    --num-machines 1 \
    --machine-rank 0 \
    --config-file configs/refcoco-unc_conv-base_100k.yaml


# models=(
#     "output/refcoco-unc_convnext-base_step-100k_bs64_640x640_repeat5_pix-cross/model_0039999.pth"
#     # "output/refcoco-unc_convnext-base_step-100k_bs64_640x640_repeat5/model_0019999.pth"
#     # "output/refcoco-unc_convnext-base_step-100k_bs64_640x640_repeat5/model_0029999.pth"
#     # "output/refcoco-unc_convnext-base_step-100k_bs64_640x640_repeat5/model_0039999.pth"
#     # "output/refcoco-unc_convnext-base_step-100k_bs64_640x640_repeat5/model_0049999.pth"
#     # "output/refcoco-unc_convnext-base_step-100k_bs64_640x640_repeat5/model_0059999.pth"
#     # "output/refcoco-unc_convnext-base_step-100k_bs64_640x640_repeat5/model_0069999.pth"
#     # "output/refcoco-unc_convnext-base_step-100k_bs64_640x640_repeat5/model_0079999.pth"
#     # "output/refcoco-unc_convnext-base_step-100k_bs64_640x640_repeat5/model_0089999.pth"
#     # "output/refcoco-unc_convnext-base_step-100k_bs64_640x640_repeat5/model_0099999.pth"
# )

# # Base command for evaluation
# for model in "${models[@]}"; do
# python train_net.py \
#     --dist-url "tcp://localhost:3681" \
#     --num-gpus 1 \
#     --num-machines 1 \
#     --machine-rank 0 \
#     --config-file configs/refcoco-unc_conv-base_100k.yaml \
#     --eval-only \
#     MODEL.WEIGHTS $model
# done
