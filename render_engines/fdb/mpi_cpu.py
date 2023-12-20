from ctypes import LittleEndianStructure
import os
import sys
here = os.getcwd()
sys.path.insert(1,here+"/ed_fractal2d_cpu/build/lib.linux-x86_64-cpython-311")

from tqdm import tqdm
# import moderngl as ModernGL
# import glfw
# from OpenGL.GL import *
import struct
import time
import math
# import glm as glm
import numpy as np
from torch import BufferDict
import random
import cv2
import argparse

from datetime import timedelta
from termcolor import colored
from PIL import Image
import torch
import PyFractal2DRenderer as fr
from typing import List

from torchvision import transforms
import matplotlib.pyplot as plt


from mpi4py import MPI

from io import BytesIO

comm = MPI.COMM_WORLD
mpirank = comm.Get_rank()
mpisize = comm.Get_size()


def print0(*args):
    if mpisize > 1:
        if mpirank == 0:
            print(*args, flush=True)
    else:
        print(*args, flush=True)

def cal_pix(gray):
    height, width = gray.shape
    num_pixels = np.count_nonzero(gray) / float(height * width)
    return num_pixels

def source (uri):
    ''' read gl code '''
    with open(uri, 'r') as fp:
        content = fp.read() 
    return content

def make_directory(save_root, name):
    if not os.path.exists(os.path.join(save_root, name)):
        os.mkdir(os.path.join(save_root, name))

parser = argparse.ArgumentParser(description='PyTorch fractal make FractalDB')
parser.add_argument('--load_root', default='./csv/data1k_fromPython/csv_rate0.2_category1000', type = str, help='load csv root')
parser.add_argument('--save_root', default='./bake_db/testCPU', type = str, help='save png root')
parser.add_argument('--image_size_x', default=362, type = int, help='image size x')
parser.add_argument('--image_size_y', default=362, type = int, help='image size y')
parser.add_argument('--image_res', default=362, type = int, help='image size y')
parser.add_argument('--pad_size_x', default=6, type = int, help='padding size x')
parser.add_argument('--pad_size_y', default=6, type = int, help='padding size y')
parser.add_argument('--iteration', default=200000, type = int, help='iteration')
parser.add_argument('--draw_type', default='patch_gray', type = str, help='{point, patch}_{gray, color}')
parser.add_argument('--weight_csv', default='./weights/weights_0.4.csv', type = str, help='weight parameter')
parser.add_argument('--instance', default=10, type = int, help='#instance, 10 => 1000 instance, 100 => 10,000 instance per category')
parser.add_argument('--rotation', default=4, type = int, help='Flip per category')
parser.add_argument('--nweights', default=25, type = int, help='Transformation of each weights. Original DB is 25 from csv files')
parser.add_argument('--checkpoint', default=0, type = int, help='From last class that was not created')
parser.add_argument('--pmode', default=0, type = int, help='Patch Mode...')
parser.add_argument('-t', '--tomemory', action='store_true',default=False,help='Do not save the image but only retain to memory')

	
def main():
    args = parser.parse_args()
    print0("\n\nAll arguments:\n",args)
    print0("\n\n")
    
    
    # Set the seeds
    np.random.seed(2041)
    random.seed(2041)
    
    # Main variables
    starttime = time.time()
    #args = conf()
    csv_names = os.listdir(args.load_root)
    csv_names.sort()
    weights = np.genfromtxt(args.weight_csv,dtype=float,delimiter=',')
    
    if args.nweights <= 25 and args.nweights > 0:
        weights = weights[:args.nweights]
    else:
        print('error on weights [1-25]')
        exit(0)

    # MPI related configurations
    nlist = len(csv_names)
    print0(f"\n\nNumber of Classes found in csv files {nlist}")
    nlist_per_rank = (nlist+mpisize-1)//mpisize
    start_list = mpirank*nlist_per_rank
    end_list = min((mpirank+1)*nlist_per_rank, nlist)

    csv_names = csv_names[start_list:end_list]
    print0(f"rank: {mpirank}, csv_names:{csv_names}]\n\n")

    comm.Barrier()
    if mpirank == 0:
        if not os.path.exists(os.path.join(args.save_root)):
            # print("Error: No directory to save DB")
            # exit(0)
            os.mkdir(os.path.join(args.save_root))

    comm.Barrier()

    print0("Rendering here: {}".format(os.path.join(args.save_root)))

    ####################### Initialize the library
    # Configure CPU render
    width:int = args.image_res
    height:int = args.image_res
    npts:int = args.iteration
    patch_mode:int = args.pmode
    pointgen_seed:int = 100
    
    ######### Not sure when this is ok...
    # patch_num:int = 10
    # patchgen_seed:int = 100
    # drift_weight:float = 0.4
    # render_mode:int = 0
    
    patchgen_rng:np.random.Generator = None
    patchgen_rng = np.random.default_rng()
    print0("\nStart the rendering loop...")
    print0(colored("Saving at {} x {} ressolution".format(args.image_res,args.image_res),'green'))
    
    if args.tomemory:
        print0(colored('Not saving the file to disk... only rendering to memory..','blue', 'on_black',['bold', 'blink']))
    else:
        print0("Saving the images to {}".format(args.save_root))    
    t  =  t1 = time.perf_counter()

    if args.checkpoint != 0:
        print("\nCheck point from {}\n".format(args.checkpoint))
        csv_names = csv_names[args.checkpoint:len(csv_names)]
        count = 0
        class_num = args.checkpoint
    
    else:
        count = 0
        class_num = 0
    
    
    # dtset_tensor = torch.FloatTensor()
    # dataset = []
    for csv_name in tqdm(csv_names):
        initial_time = time.perf_counter()
        name, ext = os.path.splitext(csv_name)
        # class_str =  '%05d' % class_num
        print0(' ->'+ csv_name)        
        
        if ext != '.csv': # Skip except for csv file
            continue
        #print(name)
        
        make_directory(args.save_root, name) # Make directory
        fractal_name=name
        fractal_weight = 0

        for weight in weights:
            padded_fractal_weight = '%02d' % fractal_weight
            fractal_weight_count = padded_fractal_weight
                       
            params = np.genfromtxt(os.path.join(args.load_root, csv_name), dtype=float, delimiter=',')
            
            flip_flg = 0
            
            param_size = params.shape[0]
            
            for i in range(param_size):
                params[i][0] = params[i][0] * weight[0]
                params[i][1] = params[i][1] * weight[1]
                params[i][2] = params[i][2] * weight[2]
                params[i][3] = params[i][3] * weight[3]
                params[i][4] = params[i][4] * weight[4]
                params[i][5] = params[i][5] * weight[5]
                # params[i][6] = params[i][6]

            _mapss:List[List[List[int]]] = [params]
            
            pts:np.array = fr.generate(npts, _mapss, pointgen_seed,class_num)
            
            for count in range(args.instance):
                
                for trans_type in range(args.rotation):
                
                    flip_flg = trans_type % 4
                    
                    _imgs:torch.LongTensor = None
                    _imgs= fr.render(pts,width,height,patch_mode, flip_flg, pointgen_seed)
                    # time.sleep(1)
                    out_data_tensor = _imgs[0].permute(2,0,1) #chw
                    out_data = transforms.ToPILImage()(out_data_tensor.squeeze_(0))

                    if args.tomemory:
                        
                        # membuf = BytesIO()
                        # out_data.save(membuf, format="png")
                        # dataset.append(membuf) 
                        # print(membuf.getvalue())
                        # print(colored('\nTotal amount of bytes: {:,}'.format(membuf.__sizeof__()),'blue'))
                        # exit(0)
                        
                        # dtset_tensor = torch.cat((dtset_tensor,out_data_tensor),1)
                        pass  
                    else:
                    # print(out_data)
                        out_data.save(os.path.join(args.save_root, fractal_name, fractal_name + "_" + fractal_weight_count + "_count_" + str(count) + "_flip" + str(trans_type) + ".png"),"PNG")
                    
            fractal_weight += 1

        class_str =  '%05d' % class_num
        # print ('save: '+class_str)        
        class_num += 1
        total_time = time.perf_counter() - initial_time
        print0(colored(" Total time render per class: {:.4f} sec, ({:0>4}) {:.4f} frm/sec ".format(total_time,str(timedelta(seconds=total_time)),1000/total_time),'magenta'))

    
    print0(f"rank: {mpirank}, Finished...\n")
    print0(f"Waiting for the rest of the ranks...")
    comm.Barrier()
    
    
    print0('\nInformation gater from memory..')
    added:int = 0
    # print(dataset)
    # for i, x in enumerate(dataset):
    #     added += x.__sizeof__()
    #     print0("total images:{} bytes {:,}".format(i+1,added))
    # print0('Total bytes readed from rank 0 {:,}'.format(added))
    
    # we are gonna communicate all the ranks and their added bytes...
    
    
    memory_t = torch.tensor(added)
    gather_mem = MPI.COMM_WORLD.reduce(memory_t, op=MPI.SUM,root=0)
    if mpirank == 0:
        print0('Total amount of data gather from MPI ranks on png dataset: {:,}'.format(gather_mem))
    
    
    
    fina_experiment_time = time.perf_counter() - t1
    print0(colored("\n\n\tTotal experiment time: {:.4f} seconds, {:0>4} ".format(fina_experiment_time,str(timedelta(seconds=fina_experiment_time))),'red'))
    
    print0("Rendering using CPU Finished...")

   

if __name__ == "__main__":
    main()
