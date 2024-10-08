import sys
import os
import os.path
import random
import argparse

from torchvision import datasets
from tqdm import tqdm
from mpi4py import MPI

from termcolor import colored
import time
from datetime import timedelta

import webdataset as wds

parser = argparse.ArgumentParser("""Generate sharded dataset from original ImageNet data.""")
parser.add_argument("--splits", default="train,val", help="which splits to write")
parser.add_argument("--filekey", action="store_true", help="use file as key (default: index)")
parser.add_argument("--maxsize", type=float, default=1e9)
parser.add_argument("--maxcount", type=float, default=100000)
parser.add_argument("--inputdata-path", default="./cifar100", help="directory containing data distribution suitable for torchvision.datasets")
parser.add_argument("--outshards-path", default="./sharded-dataset", help="directory where shards are written")
parser.add_argument("--base-name", default="cifar10", help="dataset name")
args = parser.parse_args()

assert args.maxsize > 10000000
assert args.maxcount < 1000000

os.makedirs(args.outshards_path, exist_ok=True)
splits = args.splits.split(",")

comm = MPI.COMM_WORLD
mpirank = comm.Get_rank()
mpisize = comm.Get_size()

def print0(message):
   
    if mpisize > 1:
        if mpirank == 0: 
            print(message, flush=True)
    else:
        print(message, flush=True)

def readfile(fname):
    "Read a binary file from disk."
    with open(fname, "rb") as stream:
        return stream.read()

all_keys = set()

def write_dataset(data_dir_path, base, split):
   
    # data_absolute = data_dir_path + "/" + split 
    data_absolute = data_dir_path 
    print0("Reading dataset from {}".format(data_absolute))
    
    t0 = time.perf_counter()
    ds = datasets.ImageFolder(data_absolute)
    
    t1 = time.perf_counter() - t0
    print0(colored("Time to index the dataset using ImageFolder: {:.4f} seconds, {:0>4} ".format(t1,str(timedelta(seconds=t1))),'white'))
    
    comm.Barrier()
    
    nimages = len(ds.imgs)
    print0(f"Total number of images in dataset {nimages}")
    print0("Number of total ranks : {}".format(mpisize))
    
    indexes = list(range(nimages))
    random.seed(2042)
    random.shuffle(indexes)
    
    max_num_imgs = int(args.maxcount)
    tarfile_list = [os.path.join(base, f"{args.base_name}-{split}-{i:06}.tar") for i in range((nimages+max_num_imgs-1)//max_num_imgs)]
    
    nlist = len(tarfile_list)
    print0(f"Number of shards (tars) to produce {nlist}")
        
    nlist_per_rank = (nlist+mpisize-1)//mpisize
    print0(f"Shards procesed per rank: {nlist_per_rank}")
    
    start_list = mpirank*nlist_per_rank
    end_list = min((mpirank+1)*nlist_per_rank, nlist)
    print0(f"Assiganed  tar per rank: {mpirank}, tarfile_list: [{start_list}:{end_list}]")
    
    comm.Barrier()
        
    with tqdm(total=nlist_per_rank) as pbar:
        for ind, target_filename in enumerate(tarfile_list[start_list:end_list]):
            with wds.TarWriter(target_filename) as sink:
                
                start_ind = (ind+start_list)*max_num_imgs
                end_ind = min((ind+start_list+1)*max_num_imgs, nimages)
                for i in indexes[start_ind:end_ind]:
                    fname, cls = ds.imgs[i]
                    assert cls == ds.targets[i]
                    # Read the JPEG-compressed image file contents.
                    image = readfile(fname)
                    # Construct a uniqu keye from the filename.
                    key = os.path.splitext(fname)[0]
                    # Useful check.
                    assert key not in all_keys
                    all_keys.add(key)
                    # Construct a sample.
                    xkey = key if args.filekey else "%07d" % i
                    sample = {"__key__": xkey, "png": image, "cls": cls}
                    # Write the sample to the sharded tar archives.
                    sink.write(sample)
            
            pbar.update(1)

        

def main():
    
    initial_experiment_time = time.perf_counter()
    
    for split in splits:
        print0(f"Start with split {split}")
        write_dataset(data_dir_path=args.inputdata_path, base=args.outshards_path, split=split)
    
    comm.Barrier()
    fina_experiment_time = time.perf_counter() - initial_experiment_time
    time.sleep(0.5) # -> Wait for the console to flush
    
    print0(colored("Total experiment time: {} seconds, {:0>4} ".format(fina_experiment_time,str(timedelta(seconds=fina_experiment_time))),'white'))

if __name__ == '__main__':
    main()
