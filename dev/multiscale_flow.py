

import os
import torch as th
import numpy as np
import pandas as pd
from pathlib import Path
import st_spix
from st_spix import flow_utils
from st_spix.spix_utils import img4bass,mark_spix_vid
import st_spix_cuda
import st_spix_prop_cuda
import st_spix_original_cuda
from st_spix import flow_utils as futils
import torchvision.io as iio
from einops import rearrange,repeat
from skimage.segmentation import mark_boundaries
import torchvision.utils as tv_utils
import torch.nn.functional as th_f
from torchvision.transforms import InterpolationMode
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

from st_spix.prop_seg import shift_labels

from easydict import EasyDict as edict

def run_spix(img,spix_opts):

    # -- unpack --
    olist = ["sp_size","i_std","alpha","beta"]
    sp_size,i_std,alpha,beta = [spix_opts[k] for k in olist]

    # -- bass --
    img0 = img4bass(img)
    bass_fwd = st_spix_cuda.bass_forward
    spix,means,cov,counts,ids = bass_fwd(img0,sp_size,i_std,alpha,beta)
    ids = ids.unsqueeze(1).expand(-1, means.size(-1)).long()[None,:]

    # -- check connected --
    fxn = st_spix_prop_cuda.split_disconnected
    nspix = spix.max().item()+1
    spix_split,children,split_starts = fxn(spix.clone(),nspix)
    if children.shape[1] != 0:
        print("WARNING: BASS return disconnected superpixels.")
    return spix,means,cov,counts,ids

def run_spix_prop(img,flow,spix,means,cov,counts,ids,spix_opts):

    # ----------------------------------
    #  Shift Superpixel Labels & Means
    # ----------------------------------

    # -- get superpixel flows and shift means --
    flow_sp,means_shift = st_spix.pool_flow_and_shift_mean(flow,means.clone(),spix,ids)
    spix_s,missing,invalid = shift_labels(spix,means[0],flow_sp)
    spix_s_m = spix_s.clone()

    # ------------------------------
    #     Filling Missing Values
    # ------------------------------

    # -- unpack --
    olist = ["sp_size","i_std","alpha","beta"]
    sp_size,i_std,alpha,beta = [spix_opts[k] for k in olist]
    olist = ["niters","inner_niters","nrefine"]
    niters,inner_niters,nrefine = [spix_opts[k] for k in olist]

    # -- defaults --
    nspix = spix.max().item()+1
    max_SP = nspix-1
    fill_debug,use_xfer = False,False

    # -- prop spix --
    fxn = st_spix_prop_cuda.spix_prop_dev
    outs = fxn(img,spix_s,missing,means,cov,counts,sp_size,
               i_std,alpha,beta,niters,inner_niters,nrefine,
               nspix,max_SP,fill_debug,0,use_xfer)
    boarder,spix_s,children,db_spix,db_border,db_seg,_means,cov,counts,unique_ids = outs

    # --------------------------------------
    #     Split Disconnected Superpixels
    # --------------------------------------

    # print(spix_s)
    # print(spix_s.shape)
    # spix_split = spix_s.clone()
    # fxn = st_spix_prop_cuda.split_disconnected
    # spix_split,children,split_starts = fxn(spix_split,nspix)
    return spix_s,_means

def run_multiscale(vid,spix_opts):
    B,F,H,W = vid.shape
    flows_prev = None
    spix_prev = None
    scales = [8,4,2,1]
    # spix,means,cov,counts,ids = run_spix(vid[0],spix_opts)
    for ix,scale in enumerate(scales):

        # -- rescale --
        vid_s = resize(vid[:2],(H//scale,W//scale))
        seg_s = resize(seg_s[:1],(H//scale,W//scale),
                       interpolation=InterpolationMode.NEAREST)
        # means,cov,counts = spix_to_params(vid_s[:1],seg_s)
        # nspix = spix.max().item()+1

        # -- run bass --
        spix_curr,means_curr,cov_curr,counts,ids = run_spix(vid_s[0],spix_opts)

        # -- recompute means --
        # print(means.shape,means_prop.shape)
        # from st_spix.flow_utils import index_grid
        # grid = index_grid(H,W,dtype=th.float,device="cuda",normalize=True)
        # ones = th.ones_like(grid[:,:1])
        # _,means_s = pooling(grid,seg_s,nspix)
        # _,counts_s = pooling(ones,seg_s,nspix)

        # -- update flow --
        if flows is None:
            flows = th.zeros((1,2,H//scale,W//scale),device=vid.device,dtype=vid.dtype)
        else:
            flows = resize(flows,(H//scale,W//scale),
                           interpolation=InterpolationMode.NEAREST)
        # else:
        #     flows = resize(flows_prev,(H//scale,W//scale),
        #                    interpolation=InterpolationMode.NEAREST)

        # -- run superpixels --
        # if spix_prev is None:
        #     spix,means,cov,counts,ids = run_spix(vid[0],spix_opts)
        # else:
        spix_next,means_next,cov_next = run_spix_prop(vid_s[1],flows,
                                                      spix_curr,means_curr,cov_curr,
                                                      counts,ids,spix_opts)

        # -- compute flow --
        flows = compute_flow(spix_curr,means_curr,cov_curr,
                             spix_next,means_next,cov_next)

        # # -- update flow --
        # flow_sp = th.mean((means - means_prop)**2,-1)
        # # valid_sp = th.argmin(spix[:,None] - spix_prop[:,:,None])==0

        # # -- update --
        # ds_to_pooled = st_spix_prop_cuda.downsampled_to_pooled
        # flow = ds_to_pooled(flow_sp,seg,nspix)
        # # flow_mask = ds_to_pooled(valid_sp,seg,nspix)

    return vid,flows

# def compute_flow(spix_curr,means_curr,cov_curr,
#                  spix_next,means_next,cov_next):
#     pass

def main():

    # -- get root --
    print("PID: ",os.getpid())
    root = Path("./output/mutliscale_flow/")
    if not root.exists(): root.mkdir()

    # -- config --
    cfg = edict()
    cfg.sp_size = 15
    cfg.nrefine = 30
    cfg.niters,cfg.inner_niters = 1,1
    cfg.i_std,cfg.alpha,cfg.beta = 0.1,1.,1.
    # sp_size=80
    # alpha=0.001
    # beta=10.
    # nrefine=30,

    # -- read img/flow --
    # vid = st_spix.data.davis_example(isize=None,nframes=10,vid_names=['tennis'])
    vid = st_spix.data.davis_example(nframes=15,vid_names=['baseball'],data_set="all")
    # vid = vid[0,3:6,:,:128,290-128:290]
    print(vid.shape,vid.min(),vid.max())
    vid = resize(vid[0,[5,7],:,:480,:480],(32,32))
    print(vid.shape)

    # -- run flow [raft] --
    # from st_spix.flow_utils import run_raft
    # fflow,bflow = run_raft(vid)

    # tmp = fflow[:,0].clone()
    # fflow[:,0] = fflow[:,1]
    # fflow[:,1] = tmp
    # if fflow.shape[-1] != vid.shape[-1]:
    #     print("Padding added from raft.")
    #     exit()

    # -- run flow [cv2] --
    flows = flow_pkg.run(vid[None,:]/255.,sigma=0.0,ftype="cv2")
    fflow,bflow = flows.fflow[0],flows.bflow[0]

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
    bass_fwd = st_spix_cuda.bass_forward
    spix,means,cov,counts,ids = bass_fwd(img0,sp_size,i_std,alpha,beta)
    ids = ids.unsqueeze(1).expand(-1, means.size(-1)).long()[None,:]

    # -- check connected --
    fxn = st_spix_prop_cuda.split_disconnected
    nspix = spix.max().item()+1
    spix_split,children,split_starts = fxn(spix.clone(),nspix)
    if children.shape[1] != 0:
        print("WARNING: BASS return disconnected superpixels.")

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

    print("flow: ",flow.shape,
          flow[:,0].abs().min().item(),flow[:,0].abs().max().item(),
          flow[:,1].abs().min().item(),flow[:,1].abs().max().item(),
    )

    # -- unpack --
    nspix = spix.max().item()+1
    max_SP = nspix-1
    eps = 1e-10
    print(spix0[0,28:43,48:64])

    # ----------------------------------
    #  Shift Superpixel Labels & Means
    # ----------------------------------

    # -- get superpixel flows and shift means --
    flow = th.zeros_like(flow)
    flow_sp,means_shift = st_spix.pool_flow_and_shift_mean(flow,means.clone(),spix,ids)
    # -- shift & mark overlaps/holes --
    print("flow_sp.shape: ",flow_sp.shape)
    spix_s,missing,invalid = shift_labels(spix,means[0],flow_sp)
    spix_s_m = spix_s.clone()

    # -- edge case --
    # if missing.numel() == 0:
    #  print("edge case. just run me again using a video with motion. no actual problem")
    #     exit()

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

    # print(spix_s)
    # print(spix_s.shape)
    spix_split = spix_s.clone()
    fxn = st_spix_prop_cuda.split_disconnected
    spix_split,children,split_starts = fxn(spix_split,nspix)

    # -------------------------------------
    #              Spotcheck
    # -------------------------------------

    print("spix.shape: ",spix.shape)
    spix = th.cat([spix,spix_split])
    print("spix.shape: ",spix.shape)
    vid_rs = resize(vid,(256,256),interpolation=InterpolationMode.BILINEAR)
    spix_rs = resize(spix,(256,256),interpolation=InterpolationMode.NEAREST)

    print(vid.shape,spix.shape)
    print(vid_rs.shape,spix_rs.shape)
    marked = mark_spix_vid(vid[:2],spix[:2])
    tv_utils.save_image(marked,root/"marked_og.png")

    marked = mark_spix_vid(vid_rs[:2],spix_rs[:2])
    tv_utils.save_image(marked,root/"marked_rs.png")


    # -- run alg --
    run_multiscale(vid,spix_opts)

if __name__ == "__main__":
    main()
