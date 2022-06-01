import argparse
import os
import os.path as osp
import lmdb
import pyarrow as pa
import json
from tqdm import tqdm
import warnings
warnings.filterwarnings("ignore")


def loads_pyarrow(buf):
    """
    Args:
        buf: the output of `dumps`.
    """
    return pa.deserialize(buf)


def raw_reader(path):
    with open(path, 'rb') as f:
        bin_data = f.read()
    return bin_data


def dumps_pyarrow(obj):
    """
    Serialize an object.
    Returns:
        Implementation-dependent bytes-like object
    """
    return pa.serialize(obj).to_buffer()


def folder2lmdb(json_data, img_dir, mask_dir, output_dir, split, write_frequency=1000):
    lmdb_path = osp.join(output_dir, "%s.lmdb" % split)
    isdir = os.path.isdir(lmdb_path)

    print("Generate LMDB to %s" % lmdb_path)
    db = lmdb.open(lmdb_path, subdir=isdir,
                   map_size=1099511627776 * 2, readonly=False,
                   meminit=False, map_async=True)

    txn = db.begin(write=True)
    tbar = tqdm(json_data)
    for idx, item in enumerate(tbar):
        img = raw_reader(osp.join(img_dir, item['img_name']))
        mask = raw_reader(osp.join(mask_dir, f"{item['segment_id']}.png"))
        data = {'img': img, 'mask': mask, 'cat': item['cat'],
                'seg_id': item['segment_id'], 'img_name': item['img_name'],
                'num_sents': item['sentences_num'], 'sents': [i['sent'] for i in item['sentences']]}
        txn.put(u'{}'.format(idx).encode('ascii'), dumps_pyarrow(data))
        if idx % write_frequency == 0:
            # print("[%d/%d]" % (idx, len(data_loader)))
            txn.commit()
            txn = db.begin(write=True)

    # finish iterating through dataset
    txn.commit()
    keys = [u'{}'.format(k).encode('ascii') for k in range(idx + 1)]
    with db.begin(write=True) as txn:
        txn.put(b'__keys__', dumps_pyarrow(keys))
        txn.put(b'__len__', dumps_pyarrow(len(keys)))

    print("Flushing database ...")
    db.sync()
    db.close()


def parse_args():
    parser = argparse.ArgumentParser(description='COCO Folder to LMDB.')
    parser.add_argument('-j', '--json-dir', type=str,
                        default='',
                        help='the name of json file.')
    parser.add_argument('-i', '--img-dir', type=str,
                        default='refcoco+',
                        help='the folder of images.')
    parser.add_argument('-m', '--mask-dir', type=str,
                        default='refcoco+',
                        help='the folder of masks.')
    parser.add_argument('-o', '--output-dir', type=str,
                        default='refcoco+',
                        help='the folder of output lmdb file.')
    parser.add_argument('-s', '--split', type=str,
                        default='train',
                        help='the split type.')
    args = parser.parse_args()
    return args


if __name__ == '__main__':
    args = parse_args()
    args.split = osp.basename(args.json_dir).split(".")[0]
    os.makedirs(args.output_dir, exist_ok=True)

    with open(args.json_dir, 'rb') as f:
        json_data = json.load(f)

    folder2lmdb(json_data, args.img_dir, args.mask_dir, args.output_dir, args.split)
