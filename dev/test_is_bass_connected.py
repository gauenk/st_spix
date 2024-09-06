

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

def main():

    # -- get root --
    print("PID: ",os.getpid())
    root = Path("./output/test_is_bass_connected/")
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
    # spix,means,cov,counts,ids = bass_fwd(img0,sp_size,i_std,alpha,beta)
    # ids = ids.unsqueeze(1).expand(-1, means.size(-1)).long()[None,:]
    spix = pd.read_csv("../BASS/result/img0.csv",header=None)
    spix = th.tensor(spix.values)[None,:].int().to(vid.device)
    nspix = int(spix.max())+1

    # --------------------------------------
    #     Split Disconnected Superpixels
    # --------------------------------------

    spix_split = spix.clone()
    fxn = st_spix_prop_cuda.split_disconnected
    spix_split,children,split_starts = fxn(spix_split,nspix)

    # ----------------------------------------------------------
    #        Spotcheck
    # ----------------------------------------------------------

    print(spix_split.shape)
    print(children.shape)
    print(children)
    print(split_starts)
    print(spix_split.max().item(),spix.max().item())

    # ----------------------------------------------------------
    #     Vizualize Split Superpixels with Difference Colors
    # ----------------------------------------------------------

    if children.shape[1] == 0: spix_id = [0]
    else: spix_ids = th.where(children[:,0]>0)[0]
    nsplit = len(spix_ids)
    viridis = mpl.colormaps['tab10']#.resampled(nsplit)
    seg0 = vid[0].clone()
    print(seg0.min(),seg0.max())
    for ix,spix_id in enumerate(spix_ids):
        colors = [(255*a for a in viridis(ix/(1.*nsplit))[:3])]
        # print(colors)
        mask0 = spix == spix_id
        alpha = 0.8
        seg0 = draw_segmentation_masks(seg0,mask0[0],alpha=alpha,colors=colors)
        # seg0[0,58:70,80:92] = 0.
        # seg0[0,10:20,10:20] = 0.
    tv_utils.save_image(seg0[None,:],root / "seg.png")

if __name__ == "__main__":
    main()
