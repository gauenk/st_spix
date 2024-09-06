

import os
import torch as th
import numpy as np
import pandas as pd
from pathlib import Path
import st_spix
from st_spix import flow_utils
from st_spix.spix_utils import img4bass,mark_spix
import st_spix_cuda
import st_spix_prop_cuda
import st_spix_original_cuda
from st_spix import flow_utils as futils
import torchvision.io as iio
from einops import rearrange,repeat
from skimage.segmentation import mark_boundaries
import torchvision.utils as tv_utils
import torch.nn.functional as th_f
from torchvision.utils import draw_segmentation_masks

import seaborn as sns
import matplotlib.pyplot as plt
from dev_basics.utils.metrics import compute_psnrs

try:
    import stnls
except:
    pass

from dev_basics import flow as flow_pkg
from dev_basics.utils.timer import ExpTimer,TimeIt
from torchvision.transforms.functional import resize

from st_spix.prop_seg import *

from easydict import EasyDict as edict

def main():

    # -- get root --
    print("PID: ",os.getpid())
    root = Path("./output/test_split_disconnected/")
    if not root.exists(): root.mkdir()

    # -- config --
    sp_size = 15
    nrefine = 0
    niters,inner_niters = 1,1
    i_std,alpha,beta = 0.1,1.,10.
    # sp_size=80
    # alpha=0.001
    # beta=10.
    # nrefine=30,

    # -- read img/flow --
    vid = st_spix.data.davis_example(isize=None,nframes=10,vid_names=['tennis'])
    # vid = vid[0,3:6,:,:128,290-128:290]
    print(vid.shape,vid.min(),vid.max())
    vid = resize(vid[0,3:6,:,:480,:480],(128,128))
    print(vid.shape)

    # -- run flow [raft] --
    from st_spix.flow_utils import run_raft
    fflow,bflow = run_raft(vid)

    # -- save --
    B,F,H,W = vid.shape
    tv_utils.save_image(vid,root / "vid.png")

    # # -- save --
    # marked = mark_spix_vid(vid,spix)
    # tv_utils.save_image(marked,root / "marked.png")

    # # -- get largest superpixel size --
    # mode = th.mode(spix.reshape(B,-1)).values[:,None]
    # largest_spix = th.max(th.sum(spix.reshape(B,-1) == mode,-1)).item()
    # R = largest_spix # "R" for "Radius"
    # nspix = int(spix.max()+1)

    # -=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-
    #
    #          Compute First Superpixel
    #
    # -=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-

    # -- bass --
    img0 = img4bass(vid[None,0])
    # bass_fwd = st_spix_cuda.bass_forward
    bass_fwd = st_spix_original_cuda.bass_forward
    spix,means,cov,counts,ids = bass_fwd(img0,sp_size,i_std,alpha,beta)
    ids = ids.unsqueeze(1).expand(-1, means.size(-1)).long()[None,:]
    print(spix)
    print(spix.shape)
    print(ids.shape)


    # -=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-
    #
    #         Propogate one Frame
    #
    # -=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-

    # -- unpack --
    ix = 0
    img = img4bass(vid[None,ix+1])
    flow = fflow[[ix]].contiguous()
    spix0 = spix.clone()

    # -- unpack --
    nspix = spix.max().item()+1
    max_SP = nspix-1
    eps = 1e-13

    # ----------------------------------
    #  Shift Superpixel Labels & Means
    # ----------------------------------

    # -- get superpixel flows and shift means --
    flow_sp,means_shift = st_spix.pool_flow_and_shift_mean(flow,means.clone(),spix,ids)
    # -- shift & mark overlaps/holes --
    spix_s,missing,invalid = shift_labels(spix,means[0],flow_sp)

    # -- edge case --
    if missing.numel() == 0:
        print("edge case. just run me again using a video with motion. no actual problem")
        exit()

    # ------------------------------
    #     Filling Missing Values
    # ------------------------------

    fill_debug,use_xfer = False,False
    fxn = st_spix_prop_cuda.spix_prop_dev
    outs = fxn(img,spix_s,missing,means,cov,counts,sp_size,
               i_std,alpha,beta,niters,inner_niters,nrefine,
               nspix,max_SP,fill_debug,0,use_xfer)
    boarder,spix_s,children,db_spix,db_border,db_seg,_means,cov,counts,unique_ids = outs

    # --------------------------------------
    #     Split Disconnected Superpixels
    # --------------------------------------

    print(spix_s)
    print(spix_s.shape)
    spix_split = spix_s.clone()
    fxn = st_spix_prop_cuda.split_disconnected
    spix_split,children,split_starts = fxn(spix_split,nspix)

    # ----------------------------------------------------------
    #        Spotcheck
    # ----------------------------------------------------------

    print(spix_split.shape)
    print(children.shape)
    print(children)
    # print(spix_split)
    # print(spix_s)
    print(split_starts)
    print(spix_split.max().item(),spix.max().item(),spix_s.max().item())

    # ----------------------------------------------------------
    #     Vizualize Split Superpixels with Difference Colors
    # ----------------------------------------------------------

    if children.shape[1] == 0: spix_id = 0
    else: spix_id = th.where(children[:,0]>0)[0][0].item()
    spix_id = 14
    print(spix_id)
    # print(spix_split[0,58:70,80:92])
    print(spix0[0,58:70,80:92])
    mask0 = spix0 == spix_id
    print(mask0[0,58:70,80:92])
    mask1_v0 = spix_s == spix_id
    mask1_v1 = spix_split == spix_id
    if children.shape[1] > 0:
        mask1_v2 = th.logical_or(spix_split == spix_id,spix_split == children[spix_id,0])
    else:
        mask1_v2 = th.zeros_like(mask1_v0)
    print(mask0.shape,vid[0].shape)
    print(mask1_v2)
    print(mask1_v2.sum())

    colors = ["red"]
    alpha = 0.8
    seg0 = draw_segmentation_masks(vid[0],mask0[0],alpha=alpha,colors=colors)
    seg0[0,58:70,80:92] = 0.
    print(seg0.shape)
    seg1_v0 = draw_segmentation_masks(vid[1],mask1_v0[0],alpha=alpha,colors=colors)
    seg1_v1 = draw_segmentation_masks(vid[1],mask1_v1[0],alpha=alpha,colors=colors)
    seg1_v2 = draw_segmentation_masks(vid[1],mask1_v2[0],alpha=alpha,colors=colors)
    segs = th.stack([seg0,seg1_v0,seg1_v1,seg1_v2])
    print("segs.shape: ",segs.shape)

    tv_utils.save_image(segs,root / "segs.png")

if __name__ == "__main__":
    main()
