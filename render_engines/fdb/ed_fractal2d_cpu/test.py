import numpy as np
import torch,torchvision
from Fractal2DGenData import Fractal2DGenData
import time
from timm.data.transforms import _pil_interp

def worker_init_fn(worker_id):
    info = torch.utils.data.get_worker_info()
    seed = info.dataset.patchgen_seed + worker_id
    info.dataset.patchgen_rng = np.random.default_rng(seed)

transform=torchvision.transforms.Compose([
   torchvision.transforms.ToPILImage(),
  #  torchvision.transforms.RandomCrop([224,224]),
   torchvision.transforms.Resize(224, _pil_interp('bilinear')),
   torchvision.transforms.ToTensor(),
   #torchvision.transforms.Normalize([0.5,0.5,0.5],[0.5,0.5,0.5])
])
dataset = Fractal2DGenData(param_dir="./data", width = 256, height = 256, npts = 100000, patch_mode = 0, patch_num = 10, patchgen_seed = 100, pointgen_seed = 100, transform=transform)
train_loader = torch.utils.data.DataLoader(dataset, batch_size=256, num_workers=12,
                    shuffle=True, drop_last=True, worker_init_fn=worker_init_fn)


t  =  t1 = time.perf_counter()
for batch_idx, (data, target) in enumerate(train_loader):

  t2 = time.perf_counter()

  update = t2 - t
  d = t2 - t1

  t1 = t2


  if (update >= 1.0):  
    print("\tFrames per second: {} ".format(1.0/d))
    print("\tSeconds per frame: {:.9f} ".format(d))
    t = time.perf_counter()
    #print(data)

