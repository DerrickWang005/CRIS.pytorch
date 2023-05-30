#!/bin/bash
for (( p=0; p<10; p++ ));
do
	python -u train.py --config config/cris_r50_kvasir_polyp_i.yaml --opts TEST.test_split testA TEST.test_lmdb datasets/lmdb/kvasir_polyp_80_10_10/testA.lmdb TRAIN.prompt_type p$p TRAIN.output_folder exp/kvasir_polyp_80_10_10_0_${p}_fn
	python -u train.py --config config/cris_r50_clinicdb_polyp_i.yaml --opts TEST.test_split testA TEST.test_lmdb datasets/lmdb/clinicdb_polyp_80_10_10/testA.lmdb TRAIN.prompt_type p$p TRAIN.output_folder exp/clinicdb_polyp_80_10_10_0_${p}_fn
	python -u train.py --config config/cris_r50_bkai_polyp_i.yaml --opts TEST.test_split testA TEST.test_lmdb datasets/lmdb/bkai_polyp_80_10_10/testA.lmdb TRAIN.prompt_type p$p TRAIN.output_folder exp/bkai_polyp_80_10_10_0_${p}_fn
	python -u train.py --config config/cris_r50_cvc300_polyp_i.yaml --opts TEST.test_split testA TEST.test_lmdb datasets/lmdb/cvc300_polyp_33_0_67/testA.lmdb TRAIN.prompt_type p$p TRAIN.output_folder exp/cvc300_polyp_33_0_67_0_${p}_fn
	python -u train.py --config config/cris_r50_cvccolondb_polyp_i.yaml --opts TEST.test_split testA TEST.test_lmdb datasets/lmdb/cvccolondb_polyp_51_0_949/testA.lmdb TRAIN.prompt_type p$p TRAIN.output_folder exp/cvccolondb_polyp_51_0_949_0_${p}_fn
	python -u train.py --config config/cris_r50_etis_polyp_i.yaml --opts TEST.test_split testA TEST.test_lmdb datasets/lmdb/etis_polyp_10_0_90/testA.lmdb TRAIN.prompt_type p$p TRAIN.output_folder exp/etis_polyp_10_0_90_0_${p}_fn
done
