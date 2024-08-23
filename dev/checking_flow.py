
import os
import torch as th
import numpy as np
import pandas as pd
from pathlib import Path
import st_spix
from st_spix import flow_utils
import st_spix_cuda
import st_spix_prop_cuda
import st_spix_original_cuda
from st_spix import flow_utils as futils
import torchvision.io as iio
from einops import rearrange,repeat
from skimage.segmentation import mark_boundaries
import torchvision.utils as tv_utils
import torch.nn.functional as th_f
from st_spix.spix_utils import mark_spix_vid


import seaborn as sns
import matplotlib.pyplot as plt
from dev_basics.utils.metrics import compute_psnrs

import stnls

from dev_basics import flow as flow_pkg
from dev_basics.utils.timer import ExpTimer,TimeIt

from easydict import EasyDict as edict


# -- colorwheel --
import matplotlib.pyplot as plt
import matplotlib as mpl
from matplotlib.colors import LinearSegmentedColormap, ListedColormap


def main():

    # -- read --
    root = Path("output/checking_flow/")
    read_root = root / "read_info/"
    img0 = th.load(read_root / "img0.pth")
    img1 = th.load(read_root / "img1.pth")
    spix0 = th.load(read_root / "spix0.pth")
    spix1 = th.load(read_root / "spix1.pth")
    flow = th.load(read_root / "flow.pth")
    F,H,W = img0.shape
    # print(img0.shape,img1.shape,spix0.shape,spix1.shape,flow.shape)

    vid = th.stack([img0,img1])[None,:]
    spix = [spix0[None,],spix1[None,]]
    # print(vid.shape)
    # print(spix0.shape)
    marked = mark_spix_vid(vid,spix)
    # print(marked.shape)
    # print(spix0[180:220,80:100])

    # -- view region of interest --
    ix = 139
    inds_0 = th.where(spix0==ix)
    marked[0,0][inds_0] = 0.
    marked[0,1][inds_0] = 0.
    marked[0,2][inds_0] = 1.
    inds_1 = th.where(spix1==ix)
    marked[1,0][inds_1] = 0.
    marked[1,1][inds_1] = 0.
    marked[1,2][inds_1] = 1.
    tv_utils.save_image(marked,root/"marked.png")

    # -- review superpixel pooling of that image --
    grid = futils.index_grid(H,W,dtype=spix0.dtype,
                             device=spix0.device,normalize=False)
    grid0,grid0_p = st_spix.sp_pool_from_spix(grid,spix0[None,:],return_ds=True)
    grid1,grid1_p = st_spix.sp_pool_from_spix(grid,spix1[None,:],return_ds=True)
    # print(grid0.shape,grid0_p.shape)
    print(grid0_p[0,139])
    # print(grid0.shape,grid0_p.shape)
    print(grid1_p[0,139])
    print(grid1_p[0,139] - grid0_p[0,139])

    flow0,flow0_p = st_spix.sp_pool_from_spix(flow[None,:],spix0[None,:],return_ds=True)
    # print(flow0.shape,flow0_p.shape)
    print(flow0_p[0,139])



if __name__ == "__main__":
    main()
