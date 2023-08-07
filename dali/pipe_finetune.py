from nvidia.dali.pipeline import pipeline_def
import nvidia.dali.types as types
import nvidia.dali.fn as fn
from nvidia.dali.auto_aug import rand_augment
from nvidia.dali.auto_aug import trivial_augment
import nvidia

import random as rnd

from termcolor import colored
import torch.distributed as dist

interpo_dali_2 = [types.INTERP_GAUSSIAN,types.INTERP_LANCZOS3]
interpo_dali_3 = [types.INTERP_GAUSSIAN,types.INTERP_LANCZOS3,types.INTERP_TRIANGULAR]
interpo_dali_all = [types.INTERP_GAUSSIAN,types.INTERP_LANCZOS3,types.INTERP_TRIANGULAR,types.INTERP_LINEAR,types.INTERP_NN,types.INTERP_CUBIC]
interpo_dali_all_noNN = [types.INTERP_GAUSSIAN,types.INTERP_LANCZOS3,types.INTERP_TRIANGULAR,types.INTERP_LINEAR,types.INTERP_CUBIC]

@pipeline_def(enable_conditionals=True)
def create_dali_pipeline_Aug(external_render,data_dir, crop, size, shard_id, num_shards, dali_cpu=False, is_training=True, gpu_render=False,args=None,data_config=None):
    
    dali_device = 'cpu' if dali_cpu else 'gpu'
    decoder_device = 'cpu' if dali_cpu else 'mixed'
    
    if gpu_render:
        images, labels = fn.external_source(source=external_render, num_outputs=2, dtype=types.UINT8, device=dali_device)
        
    else:
        images, labels = fn.readers.file(file_root=data_dir,
                                        shard_id=shard_id,
                                        num_shards=num_shards,
                                        prefetch_queue_depth = 4,
                                        random_shuffle=True,
                                        pad_last_batch=True,
                                        name="Reader_Train")
               
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
                                               random_aspect_ratio=args.ratio,
                                               random_area=args.scale,
                                               memory_stats=True,
                                               num_attempts=100)
        images = fn.resize(images,
                           device=dali_device,
                           size = 224,
                           interp_type=rnd.choice(interpo_dali_2),
                           #interp_type=types.INTERP_LANCZOS3,
                           antialias=True)
    
    mirror = fn.random.coin_flip(probability=0.5)

    images = rand_augment.rand_augment(images, n=2, m=9)
    # For Random Erase
    # axis_names = 'WH'
    # nregions   = 3
    # ndims = len(axis_names)
    # args_shape=(ndims*nregions,)
    # random_anchor = fn.random.uniform(range=(0., 1.), shape=args_shape)
    # random_shape = fn.random.uniform(range=(20., 50), shape=args_shape)
    
    # images = fn.erase(  images,
    #                     device=dali_device,
    #                     anchor=random_anchor,
    #                     shape=random_shape,
    #                     axis_names=axis_names,
    #                     fill_value=(118, 185, 0),
    #                     normalized_anchor=True,
    #                     normalized_shape=False,
    #                  )
    images = fn.crop_mirror_normalize(images,
                                      device=dali_device,
                                      dtype=types.FLOAT,
                                      output_layout="CHW",
                                      crop=(crop, crop),
                                      mean=[0.485 * 255,0.456 * 255,0.406 * 255],
                                      std=[0.229 * 255,0.224 * 255,0.225 * 255],
                                      mirror=mirror,
                                      )
        
    labels = labels.gpu()
    return images, labels

@pipeline_def(enable_conditionals=True)
def create_dali_pipeline_No_Aug(external_render,data_dir, crop, size, shard_id, num_shards, dali_cpu=False, is_training=False, gpu_render=False,args=None,data_config=None):
    
    dali_device = 'cpu' if dali_cpu else 'gpu'
    decoder_device = 'cpu' if dali_cpu else 'mixed'
    
    if gpu_render:
        images, labels = fn.external_source(source=external_render, num_outputs=2, dtype=types.UINT8, device=dali_device)
        
    else:
        images, labels = fn.readers.file(file_root=data_dir,
                                        shard_id=shard_id,
                                        num_shards=num_shards,
                                        prefetch_queue_depth = 4,
                                        random_shuffle=False,
                                        pad_last_batch=True,
                                        name="Reader_Eval")
        
        # ask nvJPEG to preallocate memory for the biggest sample in ImageNet for CPU and GPU to avoid reallocations in runtime
        device_memory_padding = 211025920 if decoder_device == 'mixed' else 0
        host_memory_padding = 140544512 if decoder_device == 'mixed' else 0
        # ask HW NVJPEG to allocate memory ahead for the biggest image in the data set to avoid reallocations in runtime
        preallocate_width_hint = 5980 if decoder_device == 'mixed' else 0
        preallocate_height_hint = 6430 if decoder_device == 'mixed' else 0
        
        images = fn.decoders.image( images,
                                    device=decoder_device, 
                                    output_type=types.RGB,
                                  )
        
    images = fn.resize(images,
                       size = 224,
                       device=dali_device,
                       interp_type=types.INTERP_LANCZOS3,
                       #interp_type=types.INTERP_GAUSSIAN,
                       antialias=True,
                       )
        
    images = fn.crop_mirror_normalize(images,
                                      dtype=types.FLOAT,
                                      output_layout="CHW",
                                      crop=(crop, crop),
                                      mean=[0.485 * 255,0.456 * 255,0.406 * 255],
                                      std=[0.229 * 255,0.224 * 255,0.225 * 255],
                                      mirror=False,
                                      )
    
    labels = labels.gpu()
    return images, labels
