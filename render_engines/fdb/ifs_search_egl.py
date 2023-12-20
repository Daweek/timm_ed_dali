import moderngl 
import glfw
import struct
import time
import math
import glm as glm
import numpy as np
from torch import BufferDict
import os
import cv2
import argparse
from datetime import timedelta
from termcolor import colored
from tqdm import tqdm
import array
import random

from mpi4py import MPI
comm = MPI.COMM_WORLD
mpirank = comm.Get_rank()
mpisize = comm.Get_size()

from PIL import Image


# g_res = 362
g_components = 3
g_alignment = 1
DEV = 0

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


parser = argparse.ArgumentParser(description='PyTorch fractal random search')
parser.add_argument('--rate', default=0.2, type = float, help='filling rate: (fractal pixels) / (all pixels in the image)')
parser.add_argument('--category', default=10, type = int, help='# of category')
parser.add_argument('--numof_point', default=200000, type = int, help='# of point')
parser.add_argument('--save_dir', default='./csv/searched_params', type = str, help='save directory')
parser.add_argument('--image_res', default=362, type = int, help='image size')


def main():
    args = parser.parse_args()
    print0("\n\nAll arguments:\n",args)
    print0("\n\n")
    
    # Main variables
    # Set the seeds
    np.random.seed(2041)
    random.seed(2041)
    
    # Added global relosolution
    g_res = args.image_res
    
    threshold = args.rate
    class_num = 0
    save_dir = args.save_dir

    # Check and create directories
    img_dir = os.path.join(args.save_dir, 'rate' + str(args.rate) + '_category' + str(args.category)+'_points'+str(args.numof_point))
    if os.path.exists(img_dir) == False:
        os.makedirs(img_dir)

    cat_dir = os.path.join(args.save_dir, 'csv_rate' + str(args.rate) + '_category' + str(args.category)+'_points'+str(args.numof_point))
    if os.path.exists(cat_dir) == False:
        os.makedirs(cat_dir)


    # Start ModernGL context and options
    ctx = moderngl.create_context(standalone=True, backend='egl',require=460,device_index=DEV)
    texture = ctx.texture(size=(g_res, g_res),components=g_components, alignment=g_alignment)
    depth_attach = ctx.depth_renderbuffer(size=(g_res, g_res))

    fbo = ctx.framebuffer(texture,depth_attach)
    fbo.clear(0.0,0.0,0.0)
    fbo.use()
    buf1 = ctx.buffer(reserve=g_res*g_res*g_components)           

    print0(colored('Vendor :{}'.format( ctx.info["GL_VENDOR"]), 'green'))
    print0(colored('GPU :{}'.format( ctx.info["GL_RENDERER"]),'green'))
    print0(colored('OpenGL version :{}'.format(ctx.info["GL_VERSION"]),'green'))    
    comm.Barrier()
   

    # Prepare sahders
    compute = ctx.compute_shader(source('shaders/compute_rev.glsl'))
    borders = ctx.compute_shader(source('shaders/borders_rev.glsl'))
    prog = ctx.program( vertex_shader=source('shaders/vert_rev.glsl'), fragment_shader=source('shaders/frag_rev.glsl'))


    # Set up uniforms   
    npts = args.numof_point
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


    # Loop until the user closes the window
    while (class_num < args.category):
                  
        # Prepare mappings
        #print("Computing Params...")
        #np.random.seed(66)
        a,b,c,d,e,f,prob = 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0
        param_size = np.random.randint(2,8)
        #param_size = 2
        params = np.zeros((param_size,7), dtype=float)
        sum_proba = 0.0
        # Initially, this is False, parameters are saved when inverse matrix exists
        for i in range(param_size):
            a,b,c,d,e,f = 0.0, 0.0, 0.0, 0.0, 0.0, 0.0
            param_rand = np.random.uniform(-1.0,1.0,6)
            a,b,c,d,e,f = param_rand[0:6]
            prob = abs(a*d-b*c)
            sum_proba += prob
            params[i,0:7] = a,b,c,d,e,f,prob
        for i in range(param_size):
            params[i,6] /= sum_proba
            
        rotation = [1.0,1.0]
        
        rnd = np.random.uniform(size=2)

        uniforms_buffer = struct.pack('ffi2f',rnd[0],rnd[1],param_size,rotation[0],rotation[1])
        uniforms.write(uniforms_buffer)
        uniforms.bind_to_uniform_block(3)

        m_arr = array.array('f')
        for i in range(param_size):
            m_arr.append(params[i][0] )
            m_arr.append(params[i][1] )
            m_arr.append(params[i][2] )
            m_arr.append(params[i][3] )
            m_arr.append(params[i][4] )
            m_arr.append(params[i][5] )
            m_arr.append(params[i][6] )
        unif.write(m_arr)
        unif.bind_to_uniform_block(0)
        del m_arr            
        
        # compute['mappings'] = param_size
        # prj.clear()
        # poss.clear()
        poss.bind_to_storage_buffer(0)
        color.bind_to_storage_buffer(1)
        prj.bind_to_storage_buffer(2)
        
        compute.run(group_x=1)
        ctx.memory_barrier(barriers=moderngl.SHADER_STORAGE_BARRIER_BIT)
        borders.run(group_x=1)
        ctx.memory_barrier(barriers=moderngl.SHADER_STORAGE_BARRIER_BIT)
        
        prj.bind_to_uniform_block(0)
        
        
        ctx.clear(0.0, 0.0, 0.0, 1.0) 
        fbo.use()
        vao.render(mode=moderngl.POINTS) 
        fbo.read_into(buf1,attachment=0, components=g_components, alignment=g_alignment)
        data =  buf1.read()
        
        img = Image.frombytes('RGB', (g_res,g_res), data)
        img = img.transpose(Image.FLIP_TOP_BOTTOM)

        image = np.asarray(img)
       
        pixels = cal_pix(image[:,:,0].copy())

        if pixels >= threshold:
            class_str =  '%07d' % class_num
            print ('save: '+class_str)
            #cv2.imwrite(os.path.join(img_dir, class_str + '.png'),img)
            img.save(os.path.join(img_dir, class_str + '.png'))
            np.savetxt(os.path.join(cat_dir, class_str + '.csv'), params, delimiter=',')
            class_num += 1
        else:
            pass
       

        

    print("Finished...")

if __name__ == "__main__":
    main()

