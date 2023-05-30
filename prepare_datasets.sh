#!/bin/bash

dataset_name=''
version=''

source ./.venv/bin/activate
cd datasets

for dataset_name in 'kvasir_polyp_80_10_10' 'clinicdb_polyp_80_10_10' 'bkai_polyp_80_10_10' 'cvc300_polyp_33_0_67' 'cvccolondb_polyp_51_0_949' 'etis_polyp_10_0_90'
do
	python ../tools/folder2lmdb.py -j anns/$dataset_name$version/train.json -i images/$dataset_name/ -m masks/$dataset_name -o lmdb/$dataset_name$version
	python ../tools/folder2lmdb.py -j anns/$dataset_name$version/val.json -i images/$dataset_name/ -m masks/$dataset_name -o lmdb/$dataset_name$version
	python ../tools/folder2lmdb.py -j anns/$dataset_name$version/testA.json -i images/$dataset_name/ -m masks/$dataset_name -o lmdb/$dataset_name$version
	python ../tools/folder2lmdb.py -j anns/$dataset_name$version/testB.json -i images/$dataset_name/ -m masks/$dataset_name -o lmdb/$dataset_name$version
done
