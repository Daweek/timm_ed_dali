import torch
import torch.distributed as dist
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
import torch.utils.data.distributed as torch_data_distributed
from torchvision import datasets,transforms
# from torch.nn.parallel import DistributedDataParallel as DDP
import time
import os
from datetime import timedelta
import math
import numpy as np

# from Fractal2D import Fractal2D_cpu, worker_init_fdb_fn
# from customdataloader_egl import MyLoader_egl, worker_init_egl_fn

from nvidia.dali.plugin.pytorch import DALIClassificationIterator, LastBatchPolicy
from nvidia.dali.pipeline import pipeline_def
import nvidia.dali.types as types
import nvidia.dali.fn as fn

import argparse
from termcolor import colored
# from egl_dalicustom_context import ExternalInputIterator_EGL
# from egl_fdb_iterator import ExternalInputIterator_EGL

import webdataset as wds

from typing import List

try: 
    from ffcv.fields import IntField, RGBImageField
    from ffcv.fields.decoders import IntDecoder, SimpleRGBImageDecoder, RandomResizedCropRGBImageDecoder
    from ffcv.loader import Loader, OrderOption
    from ffcv.pipeline.operation import Operation
    from ffcv.transforms import RandomHorizontalFlip, Cutout, \
        RandomTranslate, Convert, ToDevice, ToTensor, ToTorchImage
    from ffcv.transforms.common import Squeeze
    from ffcv.writer import DatasetWriter
except ImportError:
    pass


parser = argparse.ArgumentParser(description='PyTorch fractal make FractalDB')
parser.add_argument('-j', '--workers', type=int, default=-1, metavar='N',
                    help='how many training processes to use (default: 1)')
parser.add_argument('--epochs', type=int, default=3, metavar='N',
                    help='number of epochs to train (default: 200)')
parser.add_argument('-b', '--batch-size', type=int, default=128, metavar='N',
                    help='input batch size for training (default: 32)')
parser.add_argument('--root', default=None, 
                    help='path/URL for CSV path://')
parser.add_argument('--render-type', default="none", type = str, help='{cpu or gpu}')

# WEBDATASETS
parser.add_argument('-w','--wds', action='store_true', default=False,
                    help='Using WebDatasets pipeline')
parser.add_argument('--wds-datasetlen', type=int, default=0, metavar='N',
                    help='total number of images in the dataset')

#DALI pipeline
parser.add_argument('-d','--dali', action='store_true', default=False,
                    help='Using DALI pipeline')
parser.add_argument('--dgpu', action='store_true', default=False,
                    help='Using DALI pipeline')

#FFCV pipeline
parser.add_argument('-f','--ffcv', action='store_true', default=False,
                    help='Using FFCV pipeline')


# CPU render
parser.add_argument('--render-fdb', action='store_true', default=False,
                    help='Using CPU based real-time render to create FDB dataset.')
parser.add_argument('--render-oneinstance', action='store_true', default=False,
                    help='Using CPU based real-time render to create FDB dataset.')                    
parser.add_argument('--render-mvfdb', action='store_true', default=False,
                    help='Using CPU/GPU based real-time render to create MVF dataset.')
parser.add_argument('--csv', default=None, 
                    help='path/URL for CSV path://')
parser.add_argument('--render-size', type=int, default=362, metavar='N',
                    help='Image patch size (default: None => model default)')
parser.add_argument('--render-countpatch', type=int, default=1, metavar='N',
                    help='Image patch size (default: None => model default)')
parser.add_argument('--render-patchmode', type=int, default=0, metavar='N',
                    help='Image patch size (default: None => model default)')
parser.add_argument('--instance', default=10, type = int, help='#instance, 10 => 1000 instance, 100 => 10,000 instance per category')
parser.add_argument('--rotation', default=4, type = int, help='Flip per category')
parser.add_argument('--nweights', default=25, type = int, help='Transformation of each weights. Original DB is 25 from csv files')
parser.add_argument('--ram-batch', action='store_true', default=False,
                    help='Experiment to read from RAM...')
# from mpi4py import MPI
# comm = MPI.COMM_WORLD

def print0(*args):
    if dist.is_initialized():
        if dist.get_rank() == 0:
            print(*args, flush=True)
    else:
        print(*args, flush=True)


@pipeline_def
def create_dali_pipeline(external_render,data_dir, crop, size, shard_id, num_shards, dali_cpu=False, is_training=True, gpu_render=False):
    
    dali_device = 'cpu' if dali_cpu else 'gpu'
    decoder_device = 'cpu' if dali_cpu else 'mixed'
    
    if gpu_render:
        images, labels = fn.external_source(source=external_render, num_outputs=2, dtype=types.UINT8, device=dali_device)
        
    else:
        
        images, labels = fn.readers.file(file_root=data_dir,
                                        shard_id=shard_id,
                                        num_shards=num_shards,
                                        random_shuffle=is_training,
                                        pad_last_batch=True,
                                        name="Reader")
        
        # ask nvJPEG to preallocate memory for the biggest sample in ImageNet for CPU and GPU to avoid reallocations in runtime
        device_memory_padding = 211025920 if decoder_device == 'mixed' else 0
        host_memory_padding = 140544512 if decoder_device == 'mixed' else 0
        # ask HW NVJPEG to allocate memory ahead for the biggest image in the data set to avoid reallocations in runtime
        preallocate_width_hint = 5980 if decoder_device == 'mixed' else 0
        preallocate_height_hint = 6430 if decoder_device == 'mixed' else 0
        
        images = fn.decoders.image_random_crop(images,
                                                device=decoder_device, output_type=types.RGB,
                                                device_memory_padding=device_memory_padding,
                                                host_memory_padding=host_memory_padding,
                                                preallocate_width_hint=preallocate_width_hint,
                                                preallocate_height_hint=preallocate_height_hint,
                                                random_aspect_ratio=[0.8, 1.25],
                                                random_area=[0.1, 1.0],
                                                num_attempts=100)
    
    images = fn.resize(images,
                        device=dali_device,
                        resize_x=crop,
                        resize_y=crop,
                        interp_type=types.INTERP_TRIANGULAR)
    mirror = fn.random.coin_flip(probability=0.5)

    images = fn.crop_mirror_normalize(images,
                                      device=dali_device,
                                      dtype=types.FLOAT,
                                      output_layout="CHW",
                                      crop=(crop, crop),
                                      mean=[0.485 * 255,0.456 * 255,0.406 * 255],
                                      std=[0.229 * 255,0.224 * 255,0.225 * 255],
                                      mirror=mirror)
    labels = labels.gpu()
    return images, labels
    
def train(train_loader,model,criterion,optimizer,epoch,device,world_size,args,terminal_clr):
    # model.train()
    t = t0= time.perf_counter()
    
    # print(dir(train_loader.sampler))
    # print(train_loader.sampler.num_samples)
    
    def debug_tensors(im,la):
        print(colored("  {} \t{} \t{}".format(im.shape,im.device,im.dtype),terminal_clr))
        print(colored("  {} \t\t\t{} \t{}".format(la.shape,la.device,la.dtype),terminal_clr))
        # exit(0)

    if args.wds:
    
        # We measure how many epochs per work and if it is not enough we DROP last Batch
        per_work = args.wds_datasetlen // (world_size * args.workers * args.batch_size)
        last_batch = per_work * args.workers
        
    else:
        last_batch = len(train_loader)
    
    for batch_idx, data in enumerate(train_loader,start=1):
        # print(batch_idx)
        
        if args.dali:
            input = data[0]['data']
            target = data[0]["label"].squeeze(-1).long()
        
        else:
            input,target = data
            input  = input.to(device)
            target = target.to(device)
        # debug_tensors(im,target)
        # data = data.to(device)
        # target = target.to(device)
        # output = model(data)
        # loss.data.item = 0.0
        # loss = criterion(output, target)
        # optimizer.zero_grad()
        # loss.backward()
        # optimizer.step()
        # if True:
        if batch_idx % 200 == 0 or batch_idx % last_batch == 0:
        # if batch_idx % last_batch == 0:
            
            progress  = batch_idx * len(input) * args.world_size
            total_im  = args.wds_datasetlen if args.wds else last_batch * len(input) * args.world_size
            progress_perc =  batch_idx / last_batch
            
            
            print('Epoch: {} [{:,}/{:,} ({:.0%})] Time per Batch:{:.4f} s'.format(
                epoch, progress, total_im, progress_perc , time.perf_counter() - t))
            debug_tensors(input,target)
            t = time.perf_counter()

    tfinal = time.perf_counter()
    # print0("Total time per Epoch [{}]: {} seconds".format(epoch,tfinal - t0))
    total_time = tfinal - t0
    print("\tTotal time per epoch [{}]: {:0.4f} s, {:0>8} ".format(epoch,total_time,str(timedelta(seconds=total_time))))

def main():
    args = parser.parse_args()
    print0("\n\nAll arguments:\n",args)
    print0("\n\n")      
    
    # Choose color for the term
    if args.dali:
        terminal_clr = 'green'
    elif args.wds:
        terminal_clr = 'red'
    elif args.ffcv:
        terminal_clr = 'yellow'
    else:
        terminal_clr = 'blue'
    
    ssd =  os.getenv("SGE_LOCALDIR", default="/tmp")
    # master_addr = os.getenv("MASTER_ADDR", default="localhost")
    # master_port = os.getenv('MASTER_POST', default='8686')
    # method = "tcp://{}:{}".format(master_addr, master_port)
    # rank = int(os.getenv('OMPI_COMM_WORLD_RANK', '0'))
    # world_size = int(os.getenv('OMPI_COMM_WORLD_SIZE', '1'))
    rank = 0
    world_size = 1
    args.world_size = world_size
    
    
    ngpus_per_node = torch.cuda.device_count()
    node = rank // ngpus_per_node
    args.local_rank = device = rank % ngpus_per_node
    torch.cuda.set_device(args.local_rank)
    # dist.init_process_group("nccl", init_method=method, rank=rank, world_size=world_size)
    
    print("WorldSize {}  rank {} -> Ready".format(world_size, rank))
    # dist.barrier()
    
    

    train_transform = transforms.Compose([transforms.RandomCrop(224),
                                          transforms.RandomHorizontalFlip(0.5),
                                          transforms.ToTensor(),
                                          transforms.Normalize(mean=[0.485 * 255,0.456 * 255,0.406 * 255],
                                                               std=[0.229 * 255,0.224 * 255,0.225 * 255]) ,])
    
    if args.render_type != "none" and args.dali is False:
        
        if args.render_type == "cpu":
            print0("===> CPU -- FDB_render ....")
            print0("Loading from:{}".format(args.csv))
            print0("Render-size:{}".format(args.render_size))
            csv_names = os.listdir(args.csv)
            csv_names.sort()            
            nlist = len(csv_names)
            print0(f"\n\nNumber of Classes found in csv files {nlist}")
            
            
            train_dataset = Fractal2D_cpu(root=args.csv, transform = train_transform, width = args.render_size, height = args.render_size, npts = 200000, patch_mode = args.render_patchmode, patch_num = args.instance, patchgen_seed = 100, pointgen_seed = 100,oneinstance = args.render_oneinstance, countpatch =  args.render_countpatch, ram = args.ram_batch,batch=args.batch_size)
            
            train_sampler = None
            
            if (args.workers == -1):
                train_loader = DataLoader(dataset=train_dataset,
                                                    batch_size=args.batch_size,sampler=train_sampler)
            else:
                train_loader = DataLoader(dataset=train_dataset,
                                                    batch_size=args.batch_size,
                                                    sampler=train_sampler,
                                                    num_workers=args.workers,
                                                    worker_init_fn=worker_init_fdb_fn,
                                                    persistent_workers=True)
        # if args.render_type == "gpu":
        #     print0("===> GPU  FDB_render  ....")
        #     print0("Loading from:{}".format(args.root))
        #     print0("Render-size:{}".format(args.render_size))
        #     train_dataset = MyLoader_egl(root=args.root, transform = train_transform, oneinstance=True)
            
        #     train_sampler = torch.utils.data.distributed.DistributedSampler(
        #         train_dataset,
        #         num_replicas=world_size,
        #         rank=rank)
            
        #     if (args.workers == -1):
        #         train_loader = torch.utils.data.DataLoader(dataset=train_dataset,
        #                                             batch_size=args.batch_size,
        #                                             sampler=train_sampler)
        #     else:
        #         train_loader = torch.utils.data.DataLoader(dataset=train_dataset,
        #                                             batch_size=args.batch_size,
        #                                             sampler=train_sampler,
        #                                             num_workers=args.workers,
        #                                             worker_init_fn=worker_init_egl_fn,
        #                                             persistent_workers=True)
        else:
            print0("Not a valid option exit ....")
            exit(0)
    ############################### For Dali
    elif args.dali is True:
        print0(colored("[[[[[[[........Using DALI pipeline.....]]]]]]]",terminal_clr))
        if args.dgpu:
            print0(colored(">>>>>>>>>>>>>>  GPU render <<<<<<<<<<<<<",terminal_clr,attrs=["bold"]))
            print0("Loading from:{}".format(args.root))
            t0 = time.perf_counter()
            
            eii_gpu = ExternalInputIterator_EGL(args.batch_size, device_id=args.local_rank, num_gpus=world_size,root=args.root,oneinstance=False)
            
            pipe = create_dali_pipeline(external_render= eii_gpu, prefetch_queue_depth=4,
                                    batch_size=args.batch_size,
                                    num_threads=args.workers,
                                    py_num_workers=args.workers,
                                    device_id=rank,
                                    seed=12 + rank,
                                    data_dir=args.root,
                                    crop=224,
                                    size=324,
                                    dali_cpu=False,
                                    shard_id=rank,
                                    num_shards=world_size,
                                    is_training=True,
                                    gpu_render=args.dgpu)
            
            r_name = None
            r_size = len(eii_gpu)
        else:
            print0("Loading from:{}".format(args.root))
            t0 = time.perf_counter()
            pipe = create_dali_pipeline(external_render = None, prefetch_queue_depth=4,
                                    batch_size=args.batch_size,
                                    num_threads=args.workers,
                                    device_id=args.local_rank,
                                    seed=12 + rank,
                                    data_dir=args.root,
                                    crop=224,
                                    size=324,
                                    dali_cpu=False,
                                    shard_id=rank,
                                    num_shards=world_size,
                                    is_training=True,
                                    gpu_render=args.dgpu)
            
            r_name = "Reader"
            r_size = -1
    
        # pipe.build()
        train_loader = DALIClassificationIterator(pipe,size=r_size, reader_name=r_name, last_batch_policy=LastBatchPolicy.DROP,prepare_first_batch=True)
        t1 = time.perf_counter()
        
        print0("Dataset length: {:,} total images".format(len(train_loader) * args.batch_size))     
        print0("Number of batches per Rank to be processed: {:,} total batches".format(len(train_loader)))
        print0("Time load DALI:{:.4} seconds\n".format(t1-t0))
    elif args.wds is True:
        print0(colored("wwwwwwww.......Using WebDatasets pipeline.....wwwww",terminal_clr))
        print0("Loading shards:{}".format(args.root))
        
        assert args.wds_datasetlen > 0, "Dataset len should be different than CERO..."
            
        print0("Dataset length wds: {:,} total images".format(args.wds_datasetlen))
               
        batches = args.wds_datasetlen // (world_size * args.workers * args.batch_size)
        
        assert batches >0, "Something wrong with the batch size, ranks and workers ->>>>  args.wds_datasetlen // (world_size * args.workers * args.batch_size)"
        
        print0("Number of batches per epoch considering workers and worldsize: {}".format(batches))
        
        t0 = time.perf_counter()
        train_dataset = (
            wds.WebDataset(args.root,cache_dir=ssd,cache_size=1000)
                .shuffle(args.wds_datasetlen)
                .decode("pil")
                .rename(image="jpg;jpeg;JPEG;png", target="cls")
                .map_dict(image=train_transform)
                .to_tuple("image", "target")
                # .with_epoch(batches)
        )

        train_loader = DataLoader(dataset=train_dataset.batched(args.batch_size),
                                                    batch_size=None,
                                                    # sampler=train_sampler,
                                                    num_workers=args.workers,
                                                    persistent_workers=True,
                                                    prefetch_factor=4,
                                                    )
        t1 = time.perf_counter()
        # print0("Dataset length: {:,} total images".format(len(train_dataset)))
        # print0("Images per Rank to be processed: {:,} total images".format(len(train_loader)))
        print0("Time to load categories:{:.4f} seconds\n".format(t1-t0))
    elif args.ffcv is True:
        print0(colored("fffffffffffff.......Using FFCV pipeline.....fffffffffff",terminal_clr))
        print0("Loading beton file: {}".format(args.root))
        t0 = time.perf_counter()    
        
        CIFAR_MEAN = [125.307, 122.961, 113.8575]
        CIFAR_STD = [51.5865, 50.847, 51.255]
        loaders = {}

        for name in ['train']:
            label_pipeline: List[Operation] = [IntDecoder(), ToTensor(), ToDevice(torch.device('cuda:0')), Squeeze()]
            image_pipeline: List[Operation] = [RandomResizedCropRGBImageDecoder(output_size=[224,224])]
            if name == 'train':
                image_pipeline.extend([
                    RandomHorizontalFlip(),
                    RandomTranslate(padding=2, fill=tuple(map(int, CIFAR_MEAN))),
                    Cutout(4, tuple(map(int, CIFAR_MEAN))),
                ])
            image_pipeline.extend([
                ToTensor(),
                ToDevice(torch.device('cuda:0'), non_blocking=True),
                ToTorchImage(),
                Convert(torch.float32),
                transforms.Normalize(CIFAR_MEAN, CIFAR_STD),
            ])
        
        ordering = OrderOption.QUASI_RANDOM if name == 'train' else OrderOption.SEQUENTIAL

        train_loader = Loader(args.root, batch_size=args.batch_size, num_workers=args.workers,
                               order=ordering, drop_last=(name == 'train'), os_cache=True,
                               pipelines={'image': image_pipeline, 'label': label_pipeline},batches_ahead=4)
       
        t1 = time.perf_counter()
        # print0("Dataset length: {:,} total images".format(len(train_dataset)))
        print0("Batches per Rank to be processed: {:,} ".format(len(train_loader)))
        print0("Time to load categories:{} seconds\n".format(t1-t0))
        
    
    else:
        print0(colored("===<<<< Loading files from PyTorch...",terminal_clr))
        print0("Loading from:{}".format(args.root))
        t0 = time.perf_counter()
        
        train_dataset = datasets.ImageFolder(args.root,transform=train_transform)
        # print(train_dataset.imgs)
        
        train_sampler = torch_data_distributed.DistributedSampler(train_dataset,num_replicas=1,rank=rank%4)

        train_loader = DataLoader(dataset=train_dataset,
                                                    batch_size=args.batch_size,
                                                    sampler=train_sampler,
                                                    num_workers=args.workers,
                                                    persistent_workers=True,
                                                    drop_last=True,
                                                    prefetch_factor=4,
                                                    )
        t1 = time.perf_counter()
        
        print0("Dataset length: {:,} total images".format(len(train_dataset)))
        print0("Batches per Rank to be processed: {:,} total batches".format(len(train_loader)))
        print0("Time to load categories:{} seconds\n".format(t1-t0))

 
    model = None
    criterion = None
    optimizer = None
    # device = None
    # world_size = 1
    # model = CNN().to(device)
    # model = DDP(model, device_ids=[rank % ngpus])
    # criterion = nn.CrossEntropyLoss()
    # optimizer = torch.optim.SGD(model.parameters(), lr=learning_rate)

    t0 = time.perf_counter()
    for epoch in range(args.epochs):
        #model.train()
        train(train_loader,model,criterion,optimizer,epoch,device,world_size,args,terminal_clr)
        # validate(val_loader,model,criterion,device)
    
    # dist.barrier()
    print0("\n\tTotal time for all epochs:: {:0.6f} seconds,{:0>8} ".format(time.perf_counter()-t0,str(timedelta(seconds=time.perf_counter()-t0))))
    print0("Finished...")

    # dist.destroy_process_group()

if __name__ == '__main__':
    main()
