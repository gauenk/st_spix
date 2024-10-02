"""

      Check sp_video_pooling.py

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
import prop_cuda

from torchvision.transforms.functional import resize

from st_spix import scatter
from st_spix import deform
from st_spix.sp_pooling import pooling,SuperpixelPooling

import stnls
from dev_basics import flow as flow_pkg

import matplotlib as mpl
import matplotlib.cm as cm
from matplotlib import colormaps
from matplotlib import patches, pyplot as plt
# import matplotlib.pyplot as plt

from st_spix.prop import stream_bass,run_fwd_bwd
from st_spix.sp_video_pooling import video_pooling



def main():

    # -- get root --
    root = Path("./output/check_sp_video_pooling/")
    if not root.exists(): root.mkdir()

    # -- config --
    niters = 80
    niters_seg = 4
    sm_start = 10
    sp_size = 15
    # sp_size = 4
    alpha_hastings,potts = 1.,8.
    pix_var = 0.09

    # -- read img/flow --
    vid = st_spix.data.davis_example(isize=None,nframes=10,vid_names=['tennis'])
    # vid = st_spix.data.davis_example(isize=None,nframes=10,vid_names=['baseball'])
    size = 256
    # vid = vid[0,5:7,:,50:50+size,300:300+size]
    vid = vid[0,2:4,:,50:50+size,200:200+size]
    vid = resize(vid,(64,64))
    vid_og = vid.clone()

    # -- run flow [raft] --
    from st_spix.flow_utils import run_raft,run_spynet
    # fflow,bflow = run_raft(th.clip(255.*vid,0.,255.).type(th.uint8))
    fflow,bflow = run_spynet(vid)
    # print(vid.shape,fflow.shape)
    # fflow,bflow = run_raft(vid)
    if fflow.shape[-1] != vid.shape[-1]:
        print("RAFT wants image size to be a multiple of 8.")
        exit()

    # -- resize again --
    # vid = resize(vid,(64,64))
    # fflow = resize(fflow,(64,64))/2. # reduced scale by 2
    size = 128
    vid = resize(vid,(size,size))
    fflow = resize(fflow,(size,size))/(128./size) # reduced scale by X

    # -- save --
    B,F,H,W = vid.shape
    tv_utils.save_image(vid,root / "vid.png")

    # -- propogate --
    outs = stream_bass(vid,flow=fflow,
                       niters=niters,niters_seg=niters_seg,
                       sp_size=sp_size,pix_var=pix_var,
                       alpha_hastings=alpha_hastings,
                       potts=potts,sm_start=sm_start)
    spix,params,children,missing,pmaps = outs

    # -- pooling --
    pooled,down = video_pooling(vid[None,:],spix[None,:])
    tv_utils.save_image(pooled[0],root / "pooled.png")

    # -- grads --
    vid = th.cat([vid,vid,vid])
    spix = th.cat([spix,spix,spix])
    vid = th.randn_like(vid).clip(-1,1)+vid
    vid = vid[None,...,:32,:32].contiguous().float().requires_grad_(True)
    spix = spix[None,...,:32,:32].contiguous()
    pooling_fxn = lambda in_vid: video_pooling(in_vid,spix)[0]
    th.autograd.gradcheck(pooling_fxn, vid, eps=1e-3,
                          atol=1e-2, nondet_tol=1e-7, raise_exception=True)


if __name__ == "__main__":
    main()
