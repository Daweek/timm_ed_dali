import argparse
import sys

import time
import os
from datetime import timedelta
from termcolor import colored

import torch
from torch.utils.data import Dataset, DataLoader

import ctypes
import multiprocessing as mp
from multiprocessing import shared_memory as sm

import numpy as np




parser = argparse.ArgumentParser(description='To ram test')
parser.add_argument('-n','--nsamples',default=1000, type = int, help='...')
parser.add_argument('-j','--workers', default=10, type = int, help='...')
parser.add_argument('-r','--resolution', default=362, type = int, help='...')
parser.add_argument('-e','--epochs', default=2, type = int, help='...')
parser.add_argument('-f','--freq', default=10, type = int, help='...')
parser.add_argument('-b','--batch-size', default=10, type = int, help='...')


class MyDataset(Dataset):
    def __init__(self, n , c , h , w, freq):
        
        self.n = n     
        self.freq = freq
        self.c = c
        self.h = h
        self.w = w
        byte = sys.getsizeof(np.byte)
        self.ram = sm.SharedMemory(create=True, size= byte * n * c * h * w, name='dataset')
        dataset = self.ram.buf
        
        # Time to reserve memory
        t0 = time.perf_counter()
        print("Start allocating memory...\n")
        
        dataset = mp.Array(ctypes.c_float, n * c * h * w)
        t1 = time.perf_counter()
        
        print("Allocating memory {:,d} bytes".format(n * c * h * w))
        print(colored("Time to allocate memory:{:.6} seconds, {:0>4} \n".format(t1-t0, str(timedelta(seconds=t1-t0))),'blue'))
        
        shared_array = np.ctypeslib.as_array(dataset.get_obj())
        shared_array = shared_array.reshape(n, c, h, w)
        self.shared_array = torch.from_numpy(shared_array)
        self.use_cache = False

    def set_use_cache(self, use_cache):
        self.use_cache = use_cache

    def __getitem__(self, index):
        if not self.use_cache:
            # if index % self.freq == 0:
                # print('Filling cache for index {}'.format(index))
            # Add your loading logic here
            self.shared_array[index] = torch.ones(self.c, self.h, self.w) * index
        x = self.shared_array[index]
        return x

    def __delete__(self):
        self.ram.close()
        self.ram.unlink()

    def __len__(self):
        return self.n



def main():
    args = parser.parse_args()
    # nb_samples, c, h, w = 100000, 3, 362, 362
    freq = args.nsamples // args.freq
    
    dataset = MyDataset(n=args.nsamples,c = 3, h = args.resolution ,w = args.resolution, freq=args.freq)
    loader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        num_workers=0,
        shuffle=False
    )
    
    # Cache the whole dataset in the first run 
    t0 = time.perf_counter()   
    for idx, data in enumerate(loader):
        last_epoch = len(loader) - 1
        if idx % freq == 0 or idx % last_epoch == 0 :
            print('Cache idx {}, data.shape {:,}'.format(idx, data.sum()))
    t1 = time.perf_counter()
    print(colored("Time to cache:{:.6} seconds, {:0>4} \n".format(t1-t0, str(timedelta(seconds=t1-t0))),'magenta'))
    
        
    # Prepare the loaders to run in parallel
    loader.dataset.set_use_cache(use_cache=True)
    loader.num_workers = args.workers
    loader.prefetch_factor = 2
    
    for epoch in range(args.epochs):
        
        t0 = time.perf_counter()   
        last_epoch = len(loader) - 1
        for idx, x in enumerate(loader):
            if idx % freq == 0 or idx % last_epoch == 0 :
                print("Epoch:{}, Read from cache Index {} =\t\t {:>28,} bytes".format(epoch,idx, int(x.sum())))
            else:
                pass
        t1 = time.perf_counter()
        print(colored("Time per epoch:{:.6} seconds, {:0>4} \n".format(t1-t0, str(timedelta(seconds=t1-t0))),'green'))
            
        



if __name__ == '__main__':
    
    main()

    