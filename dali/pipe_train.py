
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

interpo_dali_all = [types.INTERP_GAUSSIAN,types.INTERP_LANCZOS3,types.INTERP_TRIANGULAR,types.INTERP_LINEAR,types.INTERP_NN,types.INTERP_CUBIC]
interpo_dali_all_noNN = [types.INTERP_GAUSSIAN,types.INTERP_LANCZOS3,types.INTERP_TRIANGULAR,types.INTERP_LINEAR,types.INTERP_CUBIC]
# @nvidia.dali.pipeline.experimental.pipeline_def(enable_conditionals=True,debug=True)
@pipeline_def(enable_conditionals=True)
def create_dali_pipeline_TrainAug(external_render,data_dir, crop, size, shard_id, num_shards, dali_cpu=False, is_training=True, gpu_render=False,args=None,data_config=None):
    
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
                                        # shuffle_after_epoch=True,
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
                           size = size,
                           interp_type=rnd.choice(interpo_dali_2),
                        #    interp_type=types.INTERP_LANCZOS3,
                           antialias=True)
 
    mirror = fn.random.coin_flip(probability=0.5)
    # images = fn.jitter(images, fill_value=0.4 * 255, interp_type=types.INTERP_NN)
    images = rand_augment.rand_augment(images, n=2, m=9)
    
    images = fn.crop_mirror_normalize(images,
                                      device=dali_device,
                                      dtype=types.FLOAT,
                                      output_layout="CHW",
                                      crop=(crop, crop),
                                      mean=[0.5* 255,0.5 * 255,0.5 * 255],
                                      std=[0.5 * 255,0.5 * 255,0.5 * 255],
                                      mirror=mirror,
                                      )   
        
    labels = labels.gpu()
    return images, labels
