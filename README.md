# CRIS: CLIP-Driven Referring Image Segmentation

<p align="center">
  <img src="img/pipeline.png" width="600">
</p>

This is a official PyTorch implementation of the [CRIS paper](https://arxiv.org/pdf/2111.15174):
```
@inproceedings{wang2021cris,
  title={CRIS: CLIP-Driven Referring Image Segmentation},
  author={Wang, Zhaoqing and Lu, Yu and Li, Qiang and Tao, Xunqiang and Guo, Yandong and Gong, Mingming and Liu, Tongliang},
  booktitle={Proceedings of the IEEE/CVF conference on computer vision and pattern recognition},
  year={2022}
}
```

## Preparation

1. Environment
   - [PyTorch](www.pytorch.org) (e.g. 1.10.0)
   - Other dependencies in `requirements.txt`
2. Datasets
   - The detailed instruction is in [prepare_datasets.md](tools/prepare_datasets.md)

## Training & Evaluation

This implementation only supports **multi-gpu**, **DistributedDataParallel** training, which is faster and simpler; single-gpu or DataParallel training is not supported. Besides, the evaluation only supports single-gpu mode.

To do training of CRIS with 8 GPUs, run:

```
# e.g., Evaluation on the val-set of the RefCOCO dataset
python -u train.py --config config/refcoco/cris_r50.yaml
```

To do evaluation of CRIS with 1 GPU, run:
```
# e.g., Evaluation on the val-set of the RefCOCO dataset
CUDA_VISIBLE_DEVICES=0 python -u test.py \
      --config config/refcoco/cris_r50.yaml \
      --opts TEST.test_split val-test \
             TEST.test_lmdb datasets/lmdb/refcocog_g/val.lmdb
```

## License

This project is under the CC-BY-NC 4.0 license. See [LICENSE](LICENSE) for details.
