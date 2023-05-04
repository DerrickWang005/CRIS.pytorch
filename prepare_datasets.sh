#!/bin/bash

dataset_name='kvasir-seg' 

source ./.venv/bin/activate
cd datasets
python ../tools/folder2lmdb.py -j anns/$dataset_name/train.json -i images/$dataset_name/ -m masks/$dataset_name -o lmdb/$dataset_name
python ../tools/folder2lmdb.py -j anns/$dataset_name/val.json -i images/$dataset_name/ -m masks/$dataset_name -o lmdb/$dataset_name
python ../tools/folder2lmdb.py -j anns/$dataset_name/testA.json -i images/$dataset_name/ -m masks/$dataset_name -o lmdb/$dataset_name
python ../tools/folder2lmdb.py -j anns/$dataset_name/testB.json -i images/$dataset_name/ -m masks/$dataset_name -o lmdb/$dataset_name

