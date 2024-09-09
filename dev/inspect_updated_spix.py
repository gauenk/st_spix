


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
import matplotlib as mpl
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

def load_spix_from_cpp(fname):
    return list(th.jit.load(fname).parameters())[0]

def main():

    # -- get root --
    print("PID: ",os.getpid())
    root = Path("./output/inspect_updated_spix/")
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
    print("Filename: ",root / "img0.png")
    tv_utils.save_image(vid[[0]],root / "img0.png")

    # -=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-
    #
    #          Compute First Superpixel
    #
    # -=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-

    # -- bass --
    img0 = img4bass(vid[None,0])


    # spix0 = load_spix_from_cpp("output/debug_disc/split_iter_4.pth")
    # spix1 = load_spix_from_cpp("output/debug_disc/split_iter_8.pth")
    # spix2 = load_spix_from_cpp("output/debug_disc/split_iter_12.pth")

    # spix0 = load_spix_from_cpp("output/debug_disc/merge_iter_2.pth")
    # spix1 = load_spix_from_cpp("output/debug_disc/merge_iter_6.pth")
    # spix2 = load_spix_from_cpp("output/debug_disc/merge_iter_10.pth")
    # spix = th.stack([spix0,spix1,spix2])

    # spix0 = load_spix_from_cpp("split1")
    # spix1 = load_spix_from_cpp("seg_prev")
    # spix2 = load_spix_from_cpp("seg_post")
    # spix = th.stack([spix0,spix1,spix2])

    # spix0 = load_spix_from_cpp("split1_pre")
    # spix1 = load_spix_from_cpp("split2_pre")
    # spix2 = load_spix_from_cpp("split1_post")
    # seg = load_spix_from_cpp("seg")
    # border = load_spix_from_cpp("border")
    # spix = th.stack([spix0,spix1,spix2])
    # print(seg[15:25,15:25])
    # print(border[15:25,15:25])

    spix0 = load_spix_from_cpp("seg_pre")
    spix1 = load_spix_from_cpp("seg_post")
    spix2 = th.zeros_like(spix1)
    seg = load_spix_from_cpp("seg")
    border = load_spix_from_cpp("border")
    spix = th.stack([spix0,spix1,spix2])


    # print(seg[58:68,70:80])
    # print(border[58:68,70:80])

    # print(seg[40:55,70:80])
    # print(border[40:55,70:80])

    # print(seg[52:64,70:80])
    # print(border[52:64,70:80])

    viz_spix = []
    segs = []
    for spix_ix in spix:

        # --------------------------------------
        #     Split Disconnected Superpixels
        # --------------------------------------

        nspix = int(spix_ix.max()+1)
        spix_split = spix_ix.clone()[None,:]
        fxn = st_spix_prop_cuda.split_disconnected
        spix_split,children,split_starts = fxn(spix_split,nspix)
        print(children.shape)

        # ----------------------------------------------------------
        #     Vizualize Split Superpixels with Difference Colors
        # ----------------------------------------------------------
        # print(spix_ix[40:50,-45:-35])
        # print(spix_ix[10:20,20:30])
        # print(spix_ix[15:25,15:25])
        # print(spix_ix[58:68,70:80])
        # print(spix_ix[40:55,70:80])
        # print(spix_ix[52:64,70:80])

        if children.shape[1] == 0: spix_ids = [38]
        else: spix_ids = th.where(children[:,0]>0)[0]
        # print(spix_ids)
        nsplit = len(spix_ids)
        print("Num Splits: ",nsplit)
        viridis = mpl.colormaps['tab10']#.resampled(nsplit)
        seg0 = vid[0].clone()
        for ix,spix_id in enumerate(spix_ids):
            colors = [(255*a for a in viridis(ix/(1.*nsplit))[:3])]
            mask0 = spix_ix == spix_id
            seg0 = draw_segmentation_masks(seg0,mask0,alpha=0.8,colors=colors)
        seg0[0,40:50,-45:] = 0.
        segs.append(seg0)

        viridis = mpl.colormaps['hot'].resampled(nspix)
        colors = [list(255*a for a in viridis(i/(1.*nspix))[:3]) for i in range(nspix)]
        # print(spix_ix[10:20,10:20])
        # print(colors)
        spix_mask = th.nn.functional.one_hot(spix_ix.long()).bool().transpose(2,0)
        # print(spix_max.shape)
        # exit()
        # print(spix_mask.shape)
        spix_mask = spix_ix == 0
        colors = [list(255*a for a in viridis(0.5)[:3])]
        viz_spix_ix = draw_segmentation_masks(vid[0].clone(),spix_mask,
                                              alpha=0.8,colors=colors)
        # viz_spix_ix[0,10:20,20:30] = 0.
        viz_spix_ix[0,58:68,70:80] = 0.
        viz_spix.append(viz_spix_ix)
        # break
    segs = th.stack(segs)
    viz_spix = th.stack(viz_spix)
    print(segs.shape)

    # -- save seg --
    print(root)
    tv_utils.save_image(segs,root / "seg.png")
    tv_utils.save_image(viz_spix,root / "viz_spix.png")


if __name__ == "__main__":
    main()
