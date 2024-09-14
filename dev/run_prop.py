"""

      Execute the Algorithm's Pipeline

"""

import torch as th
import numpy as np
from einops import rearrange,repeat
from pathlib import Path
from functools import reduce


# -- masked tensors --
from torch.masked import masked_tensor, as_masked_tensor
import warnings
warnings.filterwarnings(action='ignore', category=UserWarning)


from st_spix.spix_utils import mark_spix_vid,img4bass
import torchvision.utils as tv_utils

import st_spix
from st_spix import flow_utils as futils

from torchvision.transforms.functional import resize

import st_spix_cuda
from st_spix import scatter
from st_spix import deform
from st_spix.sp_pooling import pooling,SuperpixelPooling

import stnls
from dev_basics import flow as flow_pkg

import matplotlib.cm as cm
from matplotlib import colormaps
from matplotlib import patches, pyplot as plt
# import matplotlib.pyplot as plt

from st_spix.prop import stream_bass

def main():

    # -- get root --
    root = Path("./output/run_prop/")
    if not root.exists(): root.mkdir()

    # -- config --
    niters = 30
    niters_seg = 4
    sp_size = 10
    alpha,potts = 10.,10.

    # -- read img/flow --
    # vid = st_spix.data.davis_example(isize=None,nframes=10,vid_names=['tennis'])
    vid = st_spix.data.davis_example(isize=None,nframes=10,vid_names=['baseball'])
    size = 256
    vid = vid[0,5:7,:,50:50+size,300:300+size]
    vid = resize(vid,(128,128))

    # -- run flow [raft] --
    from st_spix.flow_utils import run_raft
    fflow,bflow = run_raft(th.clip(255.*vid,0.,255.).type(th.uint8))
    # fflow,bflow = run_raft(vid)
    if fflow.shape[-1] != vid.shape[-1]:
        print("RAFT wants image size to be a multiple of 8.")
        exit()

    # -- resize again --
    vid = resize(vid,(64,64))
    fflow = resize(fflow,(64,64))/2. # reduced scale by 2

    # -- save --
    B,F,H,W = vid.shape
    tv_utils.save_image(vid,root / "vid.png")

    # -- propogate --
    spix,children = stream_bass(vid,flow=fflow,niters=niters,niters_seg=niters_seg,
                                sp_size=sp_size,alpha=alpha,potts=potts)
    # -- view --
    marked = mark_spix_vid(vid,spix)
    marked[1,0][th.where(spix[1]<0)] = 0.
    marked[1,1][th.where(spix[1]<0)] = 1.
    marked[1,2][th.where(spix[1]<0)] = 0.
    marked[1,0][th.where(spix[1]==60)] = 1.
    marked[1,1][th.where(spix[1]==60)] = 0.
    marked[1,2][th.where(spix[1]==60)] = 0.
    futils.viz_flow_quiver(root/"flow.png",fflow[[0]],step=2)
    tv_utils.save_image(marked,root / "marked_fill.png")


if __name__ == "__main__":
    main()
