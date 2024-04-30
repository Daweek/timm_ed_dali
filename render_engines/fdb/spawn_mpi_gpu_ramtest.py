import moderngl as ModernGL
import glfw
from OpenGL.GL import *
import struct
import time
import math
import glm as glm
import numpy as np
import pandas as pd
import numpy as np
from datetime import timedelta
from termcolor import colored
from tqdm import tqdm

from torch import BufferDict
import os
import cv2
import argparse

import torch
import torch.distributed as dist
import torch.nn as nn
import torch.nn.functional as F
from torchvision import datasets,transforms

import argparse
import glm as glm
import struct
from PIL import Image
import moderngl
import pycuda.driver as cuda_driver
import pycuda.gl as cuda_gl
# import pycuda.gl.autoinit
import pycuda.gpuarray as cuda_array
from pycuda.compiler import SourceModule
import numpy as np
import cupy as cp

import glfw
import random
import array

from mpi4py import MPI

comm = MPI.Comm.Get_parent()
g_universe = MPI.UNIVERSE_SIZE
g_mpirank = comm.Get_rank()
g_mpisize = comm.Get_size()

from io import BytesIO
import sys

# g_res = 362
# g_res = 362
g_components = 3
g_alignment = 1

def print0(*args):
    if g_mpisize > 1:
        if g_mpirank == 0:
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

def key_event(window,key,scancode,action,mods):
    global RenewParams
    global more
    global less
    if action == glfw.PRESS and key == glfw.KEY_ESCAPE:
        glfw.window_should_close = True
    if action == glfw.PRESS and key == glfw.KEY_Q:
        glfw.window_should_close = True
    if action == glfw.PRESS and key == glfw.KEY_D:
        RenewParams = True
        more = True
    if action == glfw.PRESS and key == glfw.KEY_S:
        RenewParams = True
        less = True

parser = argparse.ArgumentParser(description='PyTorch fractal make FractalDB')
parser.add_argument('--load_root', default='./csv/data1k_fromPython/csv_rate0.2_category1000', type = str, help='load csv root')
parser.add_argument('--save_root', default='./bake_db/test', type = str, help='save png root')
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
parser.add_argument('-g','--ngpus-pernode', default=1, type = int, help='Num of GPUs in the node')
parser.add_argument('--backend', default='egl', type = str, help='{GLFW, EGL}')
parser.add_argument('-d', '--debug', action='store_true',default=False,help='Check sanity for all the images... pixel count')
parser.add_argument('-t', '--tomemory', action='store_true',default=False,help='Do not save the image but only retain to memory')
parser.add_argument('-l', '--local_ranks', action='store_true',default=False,help='Select if we render within the resources of only one node - Using local Ranks')
parser.add_argument('-s', '--stats', action='store_true',default=False,help='Print some status on the RAM memory used...')
parser.add_argument('-p','--parent_rank', default=0, type = int, help='From parents of MPIs processes')
parser.add_argument('-w','--parent_worldsize', default=0, type = int, help='From parents of MPIs processes')

def main():
    args = parser.parse_args()
    
    # Re-define the global ranks on the MPI word
    global g_mpisize
    global g_mpirank
    g_mpirank = (int(g_mpisize)) * int(args.parent_rank) + int(comm.Get_rank())
    g_mpisize = int(args.parent_worldsize) * g_mpisize
    
    # print("Local_Rank: {} ,Parent_Rank: {}, Parent_Worldsize: {} ,  Global_Rank: {}, Global_Size: {}".format(comm.Get_rank(),args.parent_rank,args.parent_worldsize,g_mpirank,g_mpisize))
    
    # comm.barrier()
    
    print0("\n\nAll arguments:\n",args)
    print0("\n\n")

    # Parse the backend
    if not (args.backend == 'glfw' or args.backend == 'egl'):
        print("No available backend.->>>>>>>>>>>>>>> exit(0)")
        exit(0)
    
    # Added global relosolution
    g_res = args.image_res
    
    # Main variables
    csv_names = os.listdir(args.load_root)
    csv_names.sort()
    weights = np.genfromtxt(args.weight_csv,dtype=float,delimiter=',')
    
    if args.nweights <= 25 and args.nweights > 0:
        weights = weights[:args.nweights]
    else:
        print('error on weights [1-25]')
        exit(0)
    
    args.save_root = args.save_root + '_' + args.backend

    # MPI related configurations
    nlist = len(csv_names)
    print0(f"Number of Classes found in csv files {nlist}")
    
    # Select if we render within the resources of only one node - Using local Ranks
    if args.local_ranks:
        print0(colored('Using per local ranks to render to local SSD. Render within the node.','yellow', 'on_black',['bold', 'blink']))
        mpisize = int(os.getenv('OMPI_COMM_WORLD_LOCAL_SIZE', '0'))
        mpirank = int(os.getenv('OMPI_COMM_WORLD_LOCAL_RANK', '0'))
        # Set the seeds
        np.random.seed(1+mpirank)
        random.seed(1+mpirank)
    else:
        mpisize = g_mpisize 
        mpirank = g_mpirank 
        # Set the seeds
        np.random.seed(1+mpirank)
        random.seed(1+mpirank)
    
    nlist_per_rank = (nlist+mpisize-1)//mpisize
    start_list = mpirank*nlist_per_rank
    end_list = min((mpirank+1)*nlist_per_rank, nlist)

    csv_names = csv_names[start_list:end_list]
    print0(f"rank: {mpirank}, csv_names:{csv_names}]\n\n")
    # comm.Barrier()
    # print0("Universe MPI: {}".format(g_mpisize))
    
    # comm.Disconnect()
    
    # exit(0)
    
    #Check per node to create directory to render
    if args.tomemory:
        print0(colored('Not saving the file to disk... only rendering to memory..','blue', 'on_black',['bold', 'blink']))
        if int(os.getenv('OMPI_COMM_WORLD_LOCAL_RANK', '0')) == 0:
            if not os.path.exists(os.path.join(args.save_root)):
                # print("Error: No directory to save DB")
                # exit(0)
                os.mkdir(os.path.join(args.save_root))
    else:
        if int(os.getenv('OMPI_COMM_WORLD_LOCAL_RANK', '0')) == 0:
            if not os.path.exists(os.path.join(args.save_root)):
                # print("Error: No directory to save DB")
                # exit(0)
                os.mkdir(os.path.join(args.save_root))
        # comm.Barrier()
        print0("Saving the images to {}".format(args.save_root))
        # print0("Rendering here: {}".format(os.path.join(args.save_root)))   
    
    DEV = mpirank % args.ngpus_pernode #->Fix Thissssssssssssssssss
    # comm.Barrier()
    
        
    # Initialize backend and context
    if args.backend == 'glfw':
          # Initialize GLFW.........................................
        print0(colored('\nBACKEND--> {}'.format('GLFW v'+ str(glfw.get_version())),'blue' if args.backend == 'glfw' else 'green'))
        if not glfw.init():
            return
        glfw.window_hint(glfw.DOUBLEBUFFER, 1)
        
        window = glfw.create_window(g_res, g_res, "Profilling fdb rendering...", None, None)
        glfw.set_window_monitor(window,monitor=None,xpos=4500,ypos=1500,width=g_res,height=g_res,refresh_rate=0)
        if not window:
            glfw.terminate()
            return
        # Make the window's context current
        glfw.make_context_current(window)
        glfw.set_key_callback(window,key_event)
        ctx = moderngl.create_context(require=460,device_index=DEV)
        data = None
  
    elif args.backend == 'egl':
        print0(colored('BACKEND--> {}'.format("EGL"),'blue' if args.backend == 'glfw' else 'green'))
        ctx = moderngl.create_context(standalone=True, backend='egl',require=460,device_index=DEV)
        texture = ctx.texture(size=(g_res, g_res),components=g_components, alignment=g_alignment)
        depth_attach = ctx.depth_renderbuffer(size=(g_res, g_res))

        fbo = ctx.framebuffer(texture,depth_attach)
        fbo.clear(0.0,0.0,0.0)
        fbo.use()
        buf1 = ctx.buffer(reserve=g_res*g_res*g_components)           
     
    ####################### Initialize the library
          
    ctx.point_size = 3
    ctx.gc_mode = None
    
    print0(colored('Vendor :{}'.format( ctx.info["GL_VENDOR"]),'blue' if args.backend == 'glfw' else 'green'))
    print0(colored('GPU :{}'.format( ctx.info["GL_RENDERER"]),'blue' if args.backend == 'glfw' else 'green'))
    print0(colored('OpenGL version :{}'.format(ctx.info["GL_VERSION"]),'blue' if args.backend == 'glfw' else 'green'))    
    # comm.Barrier()
    
    if args.debug:
        print0(colored('\nDebug enabled------Checking boundaries and image sanity------','red', 'on_black',['bold', 'blink']))
    
    # Prepare shaders
    def source(uri):
        ''' read gl code '''
        with open(uri, 'r') as fp:
            content = fp.read()
        return content
    
    compute = ctx.compute_shader(source('shaders/compute_rev.glsl'))
    borders = ctx.compute_shader(source('shaders/borders_rev.glsl'))
    prog = ctx.program( vertex_shader=source('shaders/vert_rev.glsl'), fragment_shader=source('shaders/frag_rev.glsl'))

    # Set up uniforms   
    npts = args.iteration
    compute['numPoints']   = npts
    borders['n']           = npts
    prog['rtype'] = 1

    # Prepare buffers    
    poss = ctx.buffer(reserve=npts * 4 * 2)
    poss.clear()
    color = ctx.buffer(reserve=npts * 4 * 2)
    color.clear()
    prj = ctx.buffer(reserve= 4 * 4)
    unif = ctx.buffer(reserve= 7 * 4 * 8) 
    uniforms = ctx.buffer(reserve= (2 * 4)+( 1 * 4 )+( 2 * 4 )) 

    # Prepare Vertex Array
    vao =  ctx.vertex_array(prog,[
            (poss,'2f','vert'),
            (color,'2f','pixcolor'),
            ])
    
    
    print0("\nStart the rendering loop...")
    print0(colored("Saving at {} x {} ressolution".format(g_res,g_res),'green'))
  
    initial_experiment_time = time.perf_counter()
    # Compute time realted variables
    
    count = 0
    class_num = 0
    
    dataset = []
    for csv_name in tqdm(csv_names):
        initial_time = time.perf_counter()
        name, ext = os.path.splitext(csv_name)
        # class_str =  '%05d' % class_num
        print0(' ->'+ csv_name)        
        
        if ext != '.csv': # Skip except for csv file
            continue
        #print(name)
        if not args.tomemory:
            make_directory(args.save_root, name) # Make directory
        fractal_name=name
        fractal_weight = 0

        ################### This for is the main rendering whic changes parameters
        for weight in weights:
            padded_fractal_weight = '%02d' % fractal_weight
            fractal_weight_count = padded_fractal_weight
            
            params = np.genfromtxt(os.path.join(args.load_root, csv_name), dtype=float, delimiter=',')
                            
            rnd = np.random.uniform(size=2)
            param_size = params.shape[0]
            # param_size = len(params[class_num])
            rotation = [1.0,1.0]      
            
            uniforms_buffer = struct.pack('ffi2f',rnd[0],rnd[1],param_size,rotation[0],rotation[1])
            uniforms.write(uniforms_buffer)
            uniforms.bind_to_uniform_block(3)
   
            m_arr = array.array('f')
            for i in range(param_size):
                m_arr.append(params[i][0] * weight[0])
                m_arr.append(params[i][1] * weight[1])
                m_arr.append(params[i][2] * weight[2])
                m_arr.append(params[i][3] * weight[3])
                m_arr.append(params[i][4] * weight[4])
                m_arr.append(params[i][5] * weight[5])
                m_arr.append(params[i][6])
            unif.write(m_arr)
            unif.bind_to_uniform_block(0)
            del m_arr
            
            # prj.clear()
            # poss.clear()
            # color.clear()
            
            poss.bind_to_storage_buffer(0)
            color.bind_to_storage_buffer(1)
            prj.bind_to_storage_buffer(2)
            
            compute.run(group_x=1)
            ctx.memory_barrier(barriers=ModernGL.SHADER_STORAGE_BARRIER_BIT)
            borders.run(group_x=1)
            ctx.memory_barrier(barriers=ModernGL.SHADER_STORAGE_BARRIER_BIT)
            
            prj.bind_to_uniform_block(0)

            for count in range(args.instance):
                                
                for trans_type in range(args.rotation):
                                            
                    if trans_type == 0:
                        rotation = [1.0,1.0]
                    elif trans_type == 1:
                        rotation = [-1.0,1.0]
                    elif trans_type == 2:
                        rotation = [1.0,-1.0]
                    elif trans_type == 3:
                        rotation = [-1.0,-1.0]
        
                    ctx.clear(0.0, 0.0, 0.0, 1.0) 
                    
                    # prog['rotation'].write(struct.pack('2f',rotation[0],rotation[1]))
                    rnd = np.random.uniform(size=2)
                    uniforms_buffer = struct.pack('ffi2f',rnd[0],rnd[1],param_size,rotation[0],rotation[1])
                    uniforms.write(uniforms_buffer)
                    uniforms.bind_to_uniform_block(3)
                    
                    ######### Get the image
                    if args.backend == 'glfw':
                        vao.render(mode=ModernGL.POINTS)
                        glfw.swap_buffers(window)
                        data = glReadPixels(0.0, 0.0, g_res, g_res, GL_RGB, GL_UNSIGNED_BYTE, None)
                    elif args.backend == 'egl':
                        fbo.use()
                        vao.render(mode=ModernGL.POINTS) 
                        fbo.read_into(buf1,attachment=0, components=g_components, alignment=g_alignment)
                        data =  buf1.read()
                        
                    ####### Choose the image sanity
                    
                    if args.debug:
                        data_np = np.frombuffer(data, dtype=np.byte)
                        print("Item size in np array and itemsize: {:,} , {}".format(data_np.size,data_np.itemsize))
                        print("Raw data from framebuffer to numpy: {:,} bytes".format(int(data_np.size * data_np.itemsize)))
                        
                        pixel_count = np.sum(data_np/127/3)
                    
                        print(colored("\nPixel count: {:,}".format(int(pixel_count)),'light_magenta'))
                                                
                        data = glReadPixels(0.0, 0.0, g_res, g_res, GL_RGB, GL_UNSIGNED_BYTE, None)
                        data_np = np.frombuffer(data, dtype=np.byte)
                        df = pd.DataFrame(data_np)
                        print('FrameBuffer dataframe info: {}'.format(df.describe()))
                        # print(data_np)
                        
                        poss_bytes = poss.read()
                        poss_np = np.frombuffer(poss_bytes, dtype=np.float32)
                        
                        df = pd.DataFrame(poss_np)      
                                            
                        print('Points generated dataframe info: {}'.format(df.describe()))
                        print('Info related to borders...')
                        print(' From CPU:\n rand: {} {}, param_size: {}, rot: {} {}'.format(rnd[0],rnd[1],param_size,rotation[0],rotation[1]))
                                                
                        limit_bytes = prj.read()
                        limits = np.frombuffer(limit_bytes, dtype=np.float32)
                        print('limits: {}'.format(limits))
                        
                        im = Image.frombytes('RGB', (g_res, g_res), data)
                        print('PIL image actual size {}'.format(im ))
                        membuf = BytesIO()
                        im.save(membuf, format="png") 
                        
                        print(membuf.getvalue())
                        print('\nTotal amount of bytes: {:,}'.format(membuf.__sizeof__()))
                        
                        im.save(os.path.join(args.save_root, fractal_name, fractal_name + "_" + fractal_weight_count + "_count_" + str(count) + "_flip"+ str(trans_type) + ".png"))
                        
                        print(os.path.join(args.save_root, fractal_name, fractal_name + "_" + fractal_weight_count + "_count_" + str(count) + "_flip"+ str(trans_type) + ".png"))
                        
                        # exit(0)
                        
                        if pixel_count <= 9.0:
                            print("\n\nFrameBuffer bytes is not full !!")
                            exit(0)
                        else:
                            image = Image.frombytes('RGB', (g_res, g_res), data)
                            
                            if not image.getbbox():
                                print("\n\nSave image is empty!!")
                                
                                image.save(os.path.join(args.save_root, fractal_name, fractal_name + "_" + fractal_weight_count + "_count_" + str(count) + "_flip"+ str(trans_type) + ".png"))
                                print(os.path.join(args.save_root, fractal_name, fractal_name + "_" + fractal_weight_count + "_count_" + str(count) + "_flip"+ str(trans_type) + ".png"))
                                exit(0)
                                                
                                                                        
                            image.save(os.path.join(args.save_root, fractal_name, fractal_name + "_" + fractal_weight_count + "_count_" + str(count) + "_flip"+ str(trans_type) + ".png"))
                            
                    else:
                        
                        if args.tomemory:
                           
                            ## Using OpenCV
                            data_np = np.frombuffer(data, dtype=np.byte)
                            data_np = data_np.reshape((362,362,3))
                            # _ , data_np = cv2.imencode('.png', data_np,[cv2.IMWRITE_PNG_COMPRESSION, 1])
                            dataset.append(data_np)
                            
                            # print0(data_np)
                            # print0(type(data_np))
                            # exit(0)
                            ######### Using BytesIO
                            # im = Image.frombytes('RGB', (g_res, g_res), data)
                            # membuf = BytesIO()
                            # im.save(membuf, format="png",compress_level=1)
                            # dataset.append(membuf) 
                            
                            ## Using Numpy
                            # data_np = np.frombuffer(data, dtype=np.byte)
                            
                            # img_array = np.array([membuf],dtype=object)
                            # dataset_ar = np.append(dataset_ar,img_array)
                            
                            # print(membuf.getvalue())
                            # print(colored('\nTotal amount of bytes: {:,}'.format(membuf.__sizeof__()),'blue'))
                            # exit(0)
                            pass  
                        else:
                            
                            ## Using PIL to save PNG
                            # image = Image.frombytes('RGB', (g_res, g_res), data)
                            # image.save(os.path.join(args.save_root, fractal_name, fractal_name + "_" + fractal_weight_count + "_count_" + str(count) + "_flip"+ str(trans_type) + ".png"), compress_level=1)
                            
                            ## Using OPenCV
                            data_np = np.frombuffer(data, dtype=np.byte)
                            data_np = data_np.reshape((362,362,3))
                            cv2.imwrite(os.path.join(args.save_root, fractal_name, fractal_name + "_" + fractal_weight_count + "_count_" + str(count) + "_flip"+ str(trans_type) + ".png"),data_np,[cv2.IMWRITE_PNG_COMPRESSION, 1])
                    
                    if args.backend == 'glfw':
                        glfw.poll_events()        

            fractal_weight += 1
                
        class_num += 1
        total_time = time.perf_counter() - initial_time
        
        if args.backend == 'glfw':
            print0(colored(" Total time render per class: {:.4f} sec, ({:0>4}) {:.4f} frm/sec ".format(total_time,str(timedelta(seconds=total_time)),1000/total_time),'blue'))
        elif args.backend == 'egl':
            print0(colored(" Total time render per class: {:.4f} sec, ({:0>4}) {:.4f} frm/sec  ".format(total_time,str(timedelta(seconds=total_time)),1000/total_time),'green'))
        ######### Debug ###########
        # if class_num == 10:
        #     break
        if args.backend == 'glfw':
            glfw.poll_events()

    print0(f"rank: {mpirank}, Finished...\n")
    print0(f"Waiting for the rest of the ranks...")
    # comm.Barrier()
    
    ## Debug on the RAM statistics
    
    if args.stats :
        print0('\nInformation gater from memory..')
        print0('Number of items on Dataset in RAM lenght: {:,}'.format(len(dataset)))
        
        # comm.Barrier()
        
        added:int = 0
        for i, x in enumerate(dataset):
            added += x.__sizeof__()
            # image = Image.open(x)
            # image.save(os.path.join(args.save_root, str(g_mpirank) + '00000' + "_" + '0' + "_count_" + str(i) + "_flip"+ str('i') + ".png"))
            
            # if i <= 10 :
            #     print0("total bytes from system {:,}".format(sys.getsizeof(x)))
            #     print0("total bytes {:,}".format(x.__sizeof__()))
            #     print0("total bytes added {:,}".format(added))
        # print0("total images:{} bytes {:,}".format(i+1,added))
        # print0('Total bytes readed from rank 0 {:,}'.format(added))
        
        # comm.Barrier()
        
        print("Total Bytes from rank {} is {:,}".format(g_mpirank,added))
        
        
        # comm.Barrier()
        # # we are gonna communicate all the ranks and their added bytes...
        # memory_t = torch.tensor(added)
        # gather_mem = torch.tensor(data=1)
        gather_mem = MPI.COMM_WORLD.reduce(added, op=MPI.SUM,root=0)
        if g_mpirank == 0:
            print0('Total amount of data gather from MPI ranks on png dataset: {:,}'.format(int(gather_mem)))
    
    
    if args.backend == 'glfw':
        glfw.terminate()
    fina_experiment_time = time.perf_counter() - initial_experiment_time
    print0(colored("\n\n\tTotal experiment time: {:.4f} seconds, {:0>4} ".format(fina_experiment_time,str(timedelta(seconds=fina_experiment_time))),'cyan'))

    
    ## Clean the dataset to RAM
    del dataset
    
    print0("Rendering using GPU-EGL Finished...")
    ctx.finish()

    comm.Disconnect()
   

if __name__ == "__main__":
    main()
