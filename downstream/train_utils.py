import os
from pathlib import Path
import yaml

import numpy as np
import torch
import random


def init_config(args):
    # args.ckpt_save_dir = Path(f"./{args.output_dir}/{args.experiment}/")
    args.ckpt_save_dir = Path(args.output_dir) / args.experiment
    args.ckpt_save_dir.mkdir(parents=True, exist_ok=True)
    args.ckpt_save_dir = args.ckpt_save_dir.resolve()
    config = vars(args).copy()
    config_file = args.ckpt_save_dir / (args.experiment + ".yaml")
    with open(config_file, "w") as file:
        yaml.dump(config, file)
    return args

def set_seed(args):
    r = getattr(args, "rank", 0)
    torch.manual_seed(args.random_seed + r)
    torch.cuda.manual_seed(args.random_seed + r)
    torch.cuda.manual_seed_all(args.random_seed + r) # multi-GPU and sample  different cases in each rank
    
    np.random.seed(args.random_seed + r)
    random.seed(args.random_seed + r)
    os.environ['PYTHONHASHSEED'] = str(args.random_seed + r)

    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    os.environ['CUBLAS_WORKSPACE_CONFIG'] = ':16:8'
    try:
        torch.use_deterministic_algorithms(True, warn_only=True)
    except TypeError:
        # older PyTorch versions don't support warn_only
        torch.use_deterministic_algorithms(False)


class AverageMeter(object):
    """Computes and stores the average and current value."""

    def __init__(self, name, fmt=':f'):
        self.name = name
        self.fmt = fmt
        self.reset()
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count

    def __str__(self):
        fmtstr = '{name} {val' + self.fmt + '} ({avg' + self.fmt + '})'
        return fmtstr.format(**self.__dict__)
