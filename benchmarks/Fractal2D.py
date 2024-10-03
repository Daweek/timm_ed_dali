import sys
import os

sys.path.insert(1,"/home/acc12930pb/working/transformer/timm_ed_dali/render_engines/fdb/ed_fractal2d_cpu/build/lib.linux-x86_64-cpython-312")
import PyFractal2DRenderer as fr

from tqdm import tqdm
import torch
import torchvision
import numpy as np
from termcolor import colored

from typing import Any, Callable, cast, Dict, List, Optional, Tuple
import math
import random
import re
from typing import List

from PIL import Image
from torchvision import datasets, transforms
from torchvision.datasets.folder import make_dataset, find_classes, is_image_file, has_file_allowed_extension
import torch.distributed as dist

import io

def print0(*args):
    if dist.is_initialized():
        if dist.get_rank() == 0:
            print(*args, flush=True)
    else:
        print(*args, flush=True)


def worker_init_fdb_fn(worker_id):
    info = torch.utils.data.get_worker_info()
    seed = info.dataset.patchgen_seed + worker_id
    info.dataset.patchgen_rng = np.random.default_rng()

IMG_EXTENSIONS = ('.jpg', '.jpeg', '.png', '.ppm', '.bmp', '.pgm', '.tif', '.tiff', '.webp')


def pil_loader(path: str) -> Image.Image:
    # open path as file to avoid ResourceWarning (https://github.com/python-pillow/Pillow/issues/835)
    with open(path, 'rb') as f:
        img = Image.open(f)
        return img.convert('RGB')


# TODO: specify the return type
def accimage_loader(path: str) -> Any:
    import accimage
    try:
        return accimage.Image(path)
    except IOError:
        # Potentially a decoding problem, fall back to PIL.Image
        return pil_loader(path)


def default_loader(path: str) -> Any:
    from torchvision import get_image_backend
    if get_image_backend() == 'accimage':
        return accimage_loader(path)
    else:
        return pil_loader(path)
    
class Fractal2D_cpu(datasets.DatasetFolder):
    """Dataset of generating 2D fractal image on time
    """
    RENDER_MODE_POINT = 0
    RENDER_MODE_LINE = 1
    RENDER_MODE_LINENN = 2
    RENDER_MODE_LINEXS = 3
    RENDER_MODE_FIELD = 4
    RENDER_MODE_FIELDLOG = 5
    RENDER_MODE_FIELDSINC = 6
    def __init__(
            self,
            root: str,
            loader: Callable[[str], Any] = default_loader,
            extensions: Optional[Tuple[str, ...]] = IMG_EXTENSIONS,
            transform: Optional[Callable] = None,
            target_transform: Optional[Callable] = None,
            is_valid_file: Optional[Callable[[str], bool]] = None,
            width:int = 362, height:int = 362,
            npts:int = 100000, drift_weight:float = 0.4, render_mode:int = 0,
            patch_mode:int = 0, patch_num:int = 10,
            patchgen_seed:int = 100, pointgen_seed:int = 100,
            oneinstance:bool = False, countpatch:int = 1,ram:bool = False,batch:int = 32
    ) -> None:
        super(datasets.DatasetFolder, self).__init__(root, transform=transform,
                                            target_transform=target_transform)

        self.ram = ram
        self.one_instance = oneinstance
        self.loader = loader
        self.extensions = extensions
        self.current_epoch:int = 0
        self.batch_size:int = batch
        self.width:int = width
        self.height:int = height
        self.npts:int = npts
        self.clist:List[int,str,List[List[float]]] = None
        self.ilist:List[int,str,List[List[float]]] = None
        self.ncategory:int = 0
        self.ninstance:int = 0
        self.transform:torchvision.transforms = transform
        self.drift_weight:float = drift_weight
        self.batch_num:int = 0
        self.render_mode:int = render_mode
        self.patch_mode:int = patch_mode
        self.patchgen_seed:int = patchgen_seed
        self.patchgen_rng:np.random.Generator = None
        self.pointgen_seed:int = pointgen_seed
        self.patch_num:int = patch_num
        
        self.load_categories(self.root)

        self.generate_instances()
        
        self.patch_list:List = None
        self.countpatch = countpatch
        self.patchgen_rng = np.random.default_rng(100)
        
        print0("CountPatch: {}".format(self.countpatch))
        
        # print0(self.ilist)
        
        if self.one_instance:
            self.ninstance = len(self.ilist)
            self.patch_list = self.patchgen_rng.integers(np.iinfo(np.uint32).max, size = self.countpatch)
            print0("Patch List: {}".format(self.patch_list))
        else:
            self.ninstance = len(self.ilist)*4*self.patch_num # drifting * flip_patterns * patch_patterns
        #self.batch_num = int(math.ceil(self.ninstance/self.batch_size))
        print0("Datset Size (total instances):{}".format(self.ninstance))
        # print(self.ilist)
        if ram:
            print0("---------------Preparing Batch to RAM--------------")
            self.batch_to_ram()
   
        
    
    def __len__(self) -> int:
        return self.ninstance

    def __getitem__(self, index: int) -> Tuple[Any, Any]:
        
        if self.ram:
            sample,target = self.ram_image[index%self.batch_size],self.ram_class[index%self.batch_size]
        else:
            sample, target =  self.get_instance(index)
        
        
        
        if self.transform is not None:
            sample = self.transform(sample)
        if self.target_transform is not None:
            target = self.target_transform(target)
        
        return sample, target

    def batch_to_ram(self):
        self.ram_class=[]
        self.ram_image=[]
        
        total_bytes = 0
        for i in tqdm(range(self.batch_size)):
            
            # sidx:int = idx%self.ninstance
            sidx:int = random.randint(0,self.batch_size)
            nilist:int = len(self.ilist)
            # iid:int = sidx%nilist
            # flip_flg:int = (sidx//nilist)%4
            # patch_id:int = (sidx//nilist//4)%self.patch_num
            iid:int = (sidx//4//self.patch_num)%nilist
            flip_flg:int = sidx%4
            patch_id:int = (sidx//4)%self.patch_num
            #Choose patch count random
            # patch_seed = self.patchgen_rng.integers(np.iinfo(np.uint32).max)
            # Fixed patch seed
            patch_seed:int = 100
            

            
            _mapss:List[List[List[int]]] = [self.ilist[iid][2]] 
            # _mapss:List[List[List[int]]] = [self.ilist[idx][2]]

            pts:np.array = fr.generate(self.npts, _mapss, self.pointgen_seed,self.current_epoch)
            _imgs:torch.LongTensor = None
            _imgs= fr.render(pts, self.width, self.height, self.patch_mode, flip_flg, patch_seed)
            # Patch mode:
            # 0 -> Plain gray, random dots
            # 1 -> All dots, Plain gray
            # 2 -> Random dots, Plain Color
            # 3 -> All dots, Random Gray intensity
            # 4 -> Random dots, Random Gray intensity
            # 5 -> Simple dots, Gray 127
            # _imgs= fr.render(pts, self.width, self.height, self.patch_mode, flip_flg, self.patchgen_rng.integers(np.iinfo(np.uint32).max))
        
        
            out_data = _imgs[0].permute(2,0,1) #chw
            out_data = transforms.ToPILImage()(out_data.squeeze_(0))
            self.ram_image.append(out_data)
            self.ram_class.append(self.ilist[iid][0])
            
            w,h = out_data.size
            total_bytes += w * h * 3
            # print0("Total bytes so far... {:,}".format(total_bytes))

        # print0(self.ram_class)
        # print0(len(self.ram_class))
        # print0(colored('\nTotal amount of bytes class: {:,}'.format(self.ram_class.__sizeof__()),'blue'))
        
        # print0(self.ram_image)
        # print0(len(self.ram_image))
        print0(colored('\nTotal amount of bytes class: {:,}'.format(total_bytes),'blue'))
     
       

    def set_current_epoch(self, value: int) -> None:
        # print("Enter value:{}".format(value))
        self.current_epoch = value

    def load_categories(self, param_dir:str) -> None:
        """load category params from csv
        """
        self.clist = []
        ocdns = os.listdir(param_dir)
        print0(ocdns)
        i=0
        cns = os.listdir(param_dir)
        cns.sort()
        for cn in cns:
            cn_abs = os.path.join(param_dir,cn)
            nmaps = np.loadtxt(cn_abs, delimiter=',', dtype=float).reshape((-1,7))
            maps = nmaps.tolist()
            self.clist.append([i, cn, maps])
            i += 1
        self.nclasses = len(self.clist)
        
        print(self.nclasses)
        # print(self.clist)
        # exit(0)

    def generate_instances(self) -> None:
        """generate instance params by drifting value of category params
        """
        self.ilist=[]
        _ks=[-2,-1,1,2]
        
        # Choose one instance or not
        if self.one_instance:
            print0("One Instance per class .........")
            for ci in range(len(self.clist)):
                _ms=self.clist[ci][2]
                # self.ilist.append([self.clist[ci][0], self.clist[ci][1], _ms]) #no drift
                ###### Up to here to render original Fractal
                for pi in range(6):
                    for k in _ks:
                        __ms=np.array(_ms)
                        __ms[:,pi] *= 1.0+k*self.drift_weight
                        # self.ilist.append([self.clist[ci][0], self.clist[ci][1], __ms.tolist()]) #drifted -Oirginal
                        for c in range(10):
                            self.ilist.append([self.clist[ci][0], self.clist[ci][1], __ms.tolist()]) #drifted
                        
                        break
                    break
            # For only 1 instance
            ####################################### Original- Ed
            # print("One Instance per class - Using 1k-> same instance per class .........")
            # for ci in range(len(self.clist)):
            #     _ms=self.clist[ci][2]
            #     self.ilist.append([self.clist[ci][0], self.clist[ci][1], _ms]) #no drift
            #     # For only 1 instance
            #     # for pi in range(6):
            #     #     for k in _ks:
            #     #         __ms=np.array(_ms)
            #     #         __ms[:,pi] *= 1.0+k*self.drift_weight
            #     #         self.ilist.append([self.clist[ci][0], self.clist[ci][1], __ms.tolist()]) #drifted -> Original
            #     for pi in range(6):
            #         for k in _ks:
            #             # __ms=np.array(_ms)
            #             # __ms[:,pi] *= 1.0+k*self.drift_weight
            #             # self.ilist.append([self.clist[ci][0], self.clist[ci][1], __ms.tolist()]) #drifted -> Original
            #             self.ilist.append([self.clist[ci][0], self.clist[ci][1], _ms]) #no drift
        else:
            print("Full instances per class ......")
            for ci in range(len(self.clist)):
                _ms=self.clist[ci][2]
                self.ilist.append([self.clist[ci][0], self.clist[ci][1], _ms]) #no drift
                for pi in range(6):
                    for k in _ks:
                        __ms=np.array(_ms)
                        __ms[:,pi] *= 1.0+k*self.drift_weight
                        self.ilist.append([self.clist[ci][0], self.clist[ci][1], __ms.tolist()]) #drifted

    def get_instance(self,idx:int) -> Tuple[Any, Any]:
        """get instance image and label
        """        
        if self.one_instance:
            sidx:int = idx
            nilist:int = len(self.ilist)
            # iid:int = sidx%nilist
            # flip_flg:int = (sidx//nilist)%4
            # patch_id:int = (sidx//nilist//4)%self.patch_num
            
            iid:int = idx 
            # iid:int = sidx % self.countpatch
            
            flip_flg:int = 2
            patch_id:int = self.patch_num
            
            #Choose patch count
            # index_patch = self.current_epoch % self.countpatch
            index_patch = sidx % self.countpatch
            patch_seed = self.patch_list[index_patch]
            
            # patch_seed = self.patchgen_rng.integers(np.iinfo(np.uint32).max)
            
            # if idx == 0:
            #     print("Current Patch Seed: {}".format(patch_seed))
            #     print("Current Epoch:{}".format(self.current_epoch))
            #     print("Current index_patch:{}, Current Patch: {}".format(index_patch,patch_seed))

        else:
            sidx:int = idx%self.ninstance
            nilist:int = len(self.ilist)
            # iid:int = sidx%nilist
            # flip_flg:int = (sidx//nilist)%4
            # patch_id:int = (sidx//nilist//4)%self.patch_num
            iid:int = (sidx//4//self.patch_num)%nilist
            flip_flg:int = sidx%4
            patch_id:int = (sidx//4)%self.patch_num
            #Choose patch count random
            # patch_seed = self.patchgen_rng.integers(np.iinfo(np.uint32).max)
            # Fixed patch seed
            patch_seed:int = 100
            

        out_label:int = self.ilist[iid][0]
        _mapss:List[List[List[int]]] = [self.ilist[iid][2]] 
        # _mapss:List[List[List[int]]] = [self.ilist[idx][2]]

        pts:np.array = fr.generate(self.npts, _mapss, self.pointgen_seed,self.current_epoch)
        _imgs:torch.LongTensor = None
        _imgs= fr.render(pts, self.width, self.height, self.patch_mode, flip_flg, patch_seed)
            # Patch mode:
            # 0 -> Plain gray, random dots
            # 1 -> All dots, Plain gray
            # 2 -> Random dots, Plain Color
            # 3 -> All dots, Random Gray intensity
            # 4 -> Random dots, Random Gray intensity
            # 5 -> Simple dots, Gray 127
            # _imgs= fr.render(pts, self.width, self.height, self.patch_mode, flip_flg, self.patchgen_rng.integers(np.iinfo(np.uint32).max))
        
        
        # img:torch.LongTensor = _imgs[0].permute(2,0,1) #chw
        out_data = _imgs[0].permute(2,0,1) #chw
        out_data = transforms.ToPILImage()(out_data.squeeze_(0))
        # if self.transform:
        #     out_data = self.transform(img) #chw
        # else:
        #     out_data = _imgs.float()/255.0
        
        return out_data, out_label
    
