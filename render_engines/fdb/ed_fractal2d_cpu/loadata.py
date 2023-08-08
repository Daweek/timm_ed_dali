import argparse
import torch
import torch.distributed as dist
import torch.nn as nn
import torch.nn.functional as F
from torchvision import datasets, transforms
from torch.nn.parallel import DistributedDataParallel as DDP
import time
import os
# import wandb
#from models import *
import subprocess as sp
## timm
from timm.data import create_dataset, create_loader, resolve_data_config, Mixup, FastCollateMixup, AugMixDataset
from timm.models import create_model, safe_model_name, resume_checkpoint, load_checkpoint, \
    convert_splitbn_model, model_parameters
from timm.utils import *
from timm.loss import LabelSmoothingCrossEntropy, SoftTargetCrossEntropy, JsdCrossEntropy
from timm.optim import create_optimizer_v2, optimizer_kwargs
from timm.scheduler import create_scheduler
from timm.utils import ApexScaler, NativeScaler

from timm.data.transforms import _pil_interp, RandomResizedCropAndInterpolation, ToNumpy, ToTensor
#import matplotlib.pyplot as plt
import numpy as np
import logging
from Fractal2DGenData import Fractal2DGenData


def worker_init_fn(worker_id):
    info = torch.utils.data.get_worker_info()
    seed = info.dataset.patchgen_seed + worker_id
    info.dataset.patchgen_rng = np.random.default_rng(seed)

import cv2

def print0(message):
    if dist.is_initialized():
        # if dist.get_rank() == 0:
        print(message, flush=True)
    else:
        print(message, flush=True)


class AverageMeter(object):
    def __init__(self, name, fmt=':f'):
        self.name = name
        self.fmt = fmt
        self.reset()

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


class ProgressMeter(object):
    def __init__(self, num_batches, meters, prefix="", postfix=""):
        self.batch_fmtstr = self._get_batch_fmtstr(num_batches)
        self.meters = meters
        self.prefix = prefix
        self.postfix = postfix

    def display(self, batch):
        entries = [self.prefix + self.batch_fmtstr.format(batch)]
        entries += [str(meter) for meter in self.meters]
        entries += self.postfix
        print0('\t'.join(entries))

    def _get_batch_fmtstr(self, num_batches):
        num_digits = len(str(num_batches // 1))
        fmt = '{:' + str(num_digits) + 'd}'
        return '[' + fmt + '/' + fmt.format(num_batches) + ']'


def main():
    setup_default_logging()
    parser = argparse.ArgumentParser(description='PyTorch CIFAR10 Example')
    parser.add_argument('--bs', '--batch_size', type=int, default=1, metavar='N',
                        help='input batch size for training (default: 32)')
    parser.add_argument('--epochs', type=int, default=1, metavar='N',
                        help='number of epochs to train (default: 10)')
    parser.add_argument('--lr', '--learning_rate', type=float, default=1.0e-02, metavar='LR',
                        help='learning rate (default: 1.0e-02)')
    ## from Timm
    # Dataset / Model parameters
    # parser.add_argument('data_dir', metavar='DIR',
    #                    help='path to dataset')
    parser.add_argument('--dataset', '-d', metavar='NAME', default='',
                        help='dataset type (default: ImageFolder/ImageTar if empty)')
    parser.add_argument('--train-split', metavar='NAME', default='train',
                        help='dataset train split (default: train)')
    parser.add_argument('--val-split', metavar='NAME', default='validation',
                        help='dataset validation split (default: validation)')
    parser.add_argument('--model', default='deit_tiny_patch16_224', type=str, metavar='MODEL',
                        help='Name of model to train (default: "countception"')
    parser.add_argument('--pretrained', action='store_true', default=False,
                        help='Start with pretrained version of specified network (if avail)')
    parser.add_argument('--initial-checkpoint', default='', type=str, metavar='PATH',
                        help='Initialize model from this checkpoint (default: none)')
    parser.add_argument('--pretrained-path', default='', type=str, metavar='PATH',
                        help='Load from original checkpoint and pretrain (default: none) (with --pretrained)')
    parser.add_argument('--resume', default='', type=str, metavar='PATH',
                        help='Resume full model and optimizer state from checkpoint (default: none)')
    parser.add_argument('--no-resume-opt', action='store_true', default=False,
                        help='prevent resume of optimizer state when resuming model')
    parser.add_argument('--num-classes', type=int, default=None, metavar='N',
                        help='number of label classes (Model default if None)')
    parser.add_argument('--gp', default=None, type=str, metavar='POOL',
                        help='Global pool type, one of (fast, avg, max, avgmax, avgmaxc). Model default if None.')
    parser.add_argument('--img-size', type=int, default=None, metavar='N',
                        help='Image patch size (default: None => model default)')
    parser.add_argument('--input-size', default=None, nargs=3, type=int,
                        metavar='N N N',
                        help='Input all image dimensions (d h w, e.g. --input-size 3 224 224), uses model default if empty')
    parser.add_argument('--crop-pct', default=None, type=float,
                        metavar='N', help='Input image center crop percent (for validation only)')
    parser.add_argument('--mean', type=float, nargs='+', default=None, metavar='MEAN',
                        help='Override mean pixel value of dataset')
    parser.add_argument('--std', type=float, nargs='+', default=None, metavar='STD',
                        help='Override std deviation of of dataset')
    parser.add_argument('--interpolation', default='', type=str, metavar='NAME',
                        help='Image resize interpolation type (overrides model)')
    # parser.add_argument('-b', '--batch-size', type=int, default=32, metavar='N',
    #                    help='input batch size for training (default: 32)')
    parser.add_argument('-vb', '--validation-batch-size-multiplier', type=int, default=1, metavar='N',
                        help='ratio of validation batch size to training batch size (default: 1)')

    parser.add_argument('--trainsize', type=int, default=1281167, help='ImageNet training set size')

    parser.add_argument('-w', '--webdataset', action='store_true', default=False,
                        help='Using webdata to create DataSet from .tar files')

    parser.add_argument('-j', '--workers', default=1, type=int, metavar='N',
                        help='number of data loading workers (default: 4)')

    args = parser.parse_args()

    ip = sp.getoutput('/usr/sbin/ip a show dev bond0 | grep -w inet | cut -d " " -f 6 | cut -d "/" -f 1')
    print(ip)

    device = torch.device('cuda')

    transform_train = transforms.Compose([
        transforms.ToPILImage(),
        transforms.Resize(224, _pil_interp('bilinear')),
        #transforms.CenterCrop(224),
        transforms.ToTensor(),
        #transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
    ])

    
    print("Using Torchvision..\n")
    #train_dataset = datasets.CIFAR10('./data', train=True, download=True, transform=transform_train)
    train_dataset = Fractal2DGenData(param_dir="./data", width = 256, height = 256, npts = 100000, patch_mode = 0, patch_num = 10, patchgen_seed = 100, pointgen_seed = 100, transform=transform_train)
    #sampler_train = torch.utils.data.DistributedSampler(train_dataset, num_replicas=1, rank=0, shuffle=True)
    train_loader = torch.utils.data.DataLoader(dataset=train_dataset, batch_size=64,num_workers=12,persistent_workers=False, drop_last=True,worker_init_fn=worker_init_fn)

    model = create_model(args.model)
    model.to(device)

    if args.num_classes is None:
        assert hasattr(model, 'num_classes'), 'Model must have `num_classes` attr if not set on cmd line/config.'
        args.num_classes = model.num_classes  # FIXME handle model default vs config num_classes more elegantly
        print("Checking the number of classes:{} \n".format(model.num_classes))

    print(f'Model {safe_model_name(args.model)} created, param count:{sum([m.numel() for m in model.parameters()])}')

    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.SGD(model.parameters(), lr=args.lr)

    for epoch in range(args.epochs):
        model.train()
        train_loss, train_acc = train(train_loader, model, criterion, optimizer, epoch, device)

    print("\n.\n.\n.\nALL done...\n")


def train(train_loader, model, criterion, optimizer, epoch, device):
    batch_time = AverageMeter('Time', ':.4f')
    train_loss = AverageMeter('Loss', ':.6f')
    train_acc = AverageMeter('Accuracy', ':.6f')
    progress = ProgressMeter(
        len(train_loader),
        [train_loss, train_acc, batch_time],
        prefix="Epoch: [{}]".format(epoch))
    model.train()

    print("Train loader len:{}".format(len(train_loader)))
    print("\nStart the rendering loop...")
    t  =  t1 = time.perf_counter()

    printfirst = True

    ti = time.perf_counter()
    for batch_idx, (data, target) in enumerate(train_loader):

        t2 = time.perf_counter()

        update = t2 - t
        d = t2 - t1

        t1 = t2
        # print("Batch IDX: {}".format(batch_idx))

        # Save tensor first before sending to cuda

        # data = data.to(device)
        # target = target.to(device)

        # output = model(data)

        # loss = criterion(output, target)

        # train_loss.update(loss.item(), data.size(0))
        # pred = output.data.max(1)[1]
        # acc = 100. * pred.eq(target.data).cpu().sum() / target.size(0)
        # train_acc.update(acc, data.size(0))

        # optimizer.zero_grad()
        # loss.backward()
        # optimizer.step()

        if batch_idx % 1000 == 0:
            batch_time.update(time.perf_counter() - ti)
            ti = time.perf_counter()
            progress.display(batch_idx)

        if (update >= 1.0):  
            print("\tFrames per second: {} ".format(1.0/d))
            print("\tSeconds per frame: {:.9f} ".format(d))
            t = time.perf_counter()
            #print(data)

       

    return train_loss.avg, train_acc.avg


if __name__ == '__main__':
    main()
