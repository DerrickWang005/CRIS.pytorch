import argparse
import sys
import time
import warnings

sys.path.append('./')
warnings.filterwarnings("ignore")

import torch
import torch.backends.cudnn as cudnn
import utils.config as config
from model import build_segmenter


def get_parser():
    parser = argparse.ArgumentParser(
        description='Pytorch Referring Expression Segmentation')
    parser.add_argument('--config',
                        default='path to xxx.yaml',
                        type=str,
                        help='config file')
    parser.add_argument('--opts',
                        default=None,
                        nargs=argparse.REMAINDER,
                        help='override some settings in the config.')
    args = parser.parse_args()
    assert args.config is not None
    cfg = config.load_cfg_from_cfg_file(args.config)
    if args.opts is not None:
        cfg = config.merge_cfg_from_list(cfg, args.opts)
    return cfg


def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def main():
    # init arguments
    args = get_parser()
    torch.cuda.set_device(0)
    # create model
    model, _ = build_segmenter(args)
    model = model.cuda()
    model.eval()
    # set cudnn state
    cudnn.benchmark = True
    cudnn.deterministic = False
    cudnn.enabled = True
    # init dummy tensor
    image = torch.randn(1, 3, 416, 416).cuda()
    text = torch.randint(4096, size=(1, args.word_len)).long().cuda()
    # init time & memory
    avg_time = 0
    avg_mem = 0
    # record initial gpu memory
    mem = torch.cuda.max_memory_allocated()

    with torch.no_grad():
        for i in range(500):
            start_time = time.time()
            _ = model(image, text)
            torch.cuda.synchronize()
            if (i+1) >= 100:
                avg_time += (time.time() - start_time)
                avg_mem += (torch.cuda.max_memory_allocated() - mem) / 1.073742e9
    params = count_parameters(model) * 1e-6
    print('#########################################')
    print("Average Parameters : {:.2f} M".format(params))
    print("Average FPS: {:.2f}".format(400/avg_time))
    print("Average GPU Memory: {:.2f} GB".format(avg_mem/400))
    print('#########################################')


if __name__ == '__main__':
    main()
