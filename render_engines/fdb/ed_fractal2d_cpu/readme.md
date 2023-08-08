* [PyFractal2DRenderer](#PyFractal2DRenderer)
    FractalDB renderer using CPU.

---

# PyFractal2DRenderer

Generate and render FractalDB CPU.
## Install

### Requirements

* Python (3.8)
* PyTorch (12.0)


### Compile

```sh
python setup.py build
```

## How to use

```python
import sys
sys.path.insert(1, "/path/to/dynamic_lib")
import numpy as np
import torch
import PyFractal2DRenderer as fr

maps_shida=[
    [0,0,0,0.16,0,0,0.01],
    [0.85,0.04,-0.04,0.85,0,1.60,0.85],
    [0.20,-0.26,0.23,0.22,0,1.60,0.07],
    [-0.15,0.28,0.26,0.24,0,0.44,0.07]
]

mapss=[
    maps_shida
]

npts=100000
pointgen_seed=100 
pts = fr.generate(npts, mapss, pointgen_seed) #np.array [N,npts,2]

width=256
height=256
patch_mode=0 
flip_flg=0 
patchgen_seed=100 
imgs = fr.render(pts, width, height, patch_mode, flip_flg, patchgen_seed) #torch.Tensor(ByteTensor) [N,H,W,C]
```

---
