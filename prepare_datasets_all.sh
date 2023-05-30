#!/bin/bash

dataset_name='kvasir_polyp_80_10_10'
version='_testxxxxx'

source ./.venv/bin/activate
cd datasets
python /mnt/Enterprise/kanchan/CRIS.pytorch/tools/folder2lmdb.py -j anns_old/kvasir_polyp_80_10_10/train.json -i images/$dataset_name/ -m masks/$dataset_name -o lmdb_old/$dataset_name$version
python /mnt/Enterprise/kanchan/CRIS.pytorch/tools/folder2lmdb.py -j anns_old/kvasir_polyp_80_10_10/val.json -i images/$dataset_name/ -m masks/$dataset_name -o lmdb_old/$dataset_name$version
python /mnt/Enterprise/kanchan/CRIS.pytorch/tools/folder2lmdb.py -j anns_old/kvasir_polyp_80_10_10/testA.json -i images/$dataset_name/ -m masks/$dataset_name -o lmdb_old/$dataset_name$version
python /mnt/Enterprise/kanchan/CRIS.pytorch/tools/folder2lmdb.py -j anns_old/kvasir_polyp_80_10_10/testB.json -i images/$dataset_name/ -m masks/$dataset_name -o lmdb_old/$dataset_name$version

