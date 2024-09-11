

import os
import torch as th
import numpy as np
import pandas as pd
from pathlib import Path
import st_spix
from st_spix import deform
from st_spix import flow_utils
from st_spix.spix_utils import img4bass,mark_spix_vid
import st_spix_cuda
import st_spix_prop_cuda
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

def run_spix_bass(img,spix_opts):

    # -- unpack --
    olist = ["sp_size","i_std","alpha","beta"]
    sp_size,i_std,alpha,beta = [spix_opts[k] for k in olist]
    th.cuda.synchronize()

    # -- bass --
    img0 = img4bass(img)
    bass_fwd = st_spix_cuda.bass_forward
    spix,means,cov,counts,ids = bass_fwd(img0,sp_size,i_std,alpha,beta)
    ids = ids.unsqueeze(1).expand(-1, means.size(-1)).long()[None,:]
    # print("hi.")
    # print(th.all(spix>=0))
    # print("hey.")

    # -- check connected --
    # fxn = st_spix_prop_cuda.split_disconnected
    # nspix = spix.max().item()+1
    # spix_split,children,split_starts = fxn(spix.clone(),nspix)
    # if children.shape[1] != 0:
    #     print("WARNING: BASS return disconnected superpixels.")
    return spix,means,cov,counts,ids

def run_spix_prop(img,flow,spix,means,cov,counts,ids,spix_opts):

    # ----------------------------------
    #  Shift Superpixel Labels & Means
    # ----------------------------------

    # -- get superpixel flows and shift means --
    # print(img.shape,flow.shape,spix.shape)
    # print(th.all(spix>=0).item())
    # exit()
    flow_sp,means_shift = st_spix.pool_flow_and_shift_mean(flow,means.clone(),spix,ids)
    spix_s,missing,invalid = shift_labels(spix,means[0],flow_sp)
    spix_s_m = spix_s.clone()
    # print(img.shape,flow.shape,spix.shape)
    # print(spix_s.shape)
    # print(spix_s_m.shape)
    # exit()

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
    img = img4bass(img)
    fxn = st_spix_prop_cuda.spix_prop_dev
    outs = fxn(img,spix_s,missing,means,cov,counts,sp_size,
               i_std,alpha,beta,niters,inner_niters,nrefine,
               nspix,max_SP,fill_debug,0,use_xfer)
    boarder,spix_s,children,db_spix,db_border,db_seg,means,cov,counts,unique_ids = outs

    # --------------------------------------
    #     Split Disconnected Superpixels
    # --------------------------------------

    # print(spix_s)
    # print(spix_s.shape)
    # spix_split = spix_s.clone()
    # fxn = st_spix_prop_cuda.split_disconnected
    # spix_split,children,split_starts = fxn(spix_split,nspix)
    return spix_s,means,cov,counts>0

def run_spix(vid,flows,spix_opts):

    # -- unpack --
    device = vid.device
    B,F,H,W = vid.shape

    # -- run bass --
    spix_curr,means_curr,cov_curr,counts,ids = run_spix_bass(vid[0],spix_opts)

    # -- run prop --
    assert B == 2,"Only two for now."
    outs = run_spix_prop(vid[1].contiguous(),flows[[0]].contiguous(),
                         spix_curr,means_curr,cov_curr,counts,ids,spix_opts)
    spix_next,means_next,cov_next,valid_next = outs

    # -- combine --
    spix = th.cat([spix_curr,spix_next])
    means = th.stack([means_curr,means_next])
    covs = th.stack([cov_curr,cov_next])
    valid = th.stack([th.ones_like(valid_next),valid_next]) # this is not right.

    return spix,means,covs,valid

def run_multiscale(vid,flows,spix_opts):

    root = Path("./output/mutliscale_flow/")
    if not root.exists(): root.mkdir()
    resize_nn = lambda a,b: resize(a,b,interpolation=InterpolationMode.NEAREST)
    resize_bl = lambda a,b: resize(a,b,interpolation=InterpolationMode.BILINEAR)


    device = vid.device
    B,F,H,W = vid.shape
    flows = None
    # scales = [8,4,2,1]
    # scales = [4,2,1]
    scales = [1]
    warps,deltas = [],[]
    for ix,scale in enumerate(scales):

        # -- rescale --
        vid_s = resize_bl(vid[:2],(H//scale,W//scale)).contiguous()

        # -- update flow --
        if flows is None:
            flows = th.zeros((B-1,2,H//scale,W//scale),device=device,dtype=vid.dtype)
        else:
            flows = resize_nn(flows,(H//scale,W//scale))//scale

        # -- run bass --
        spix,means,covs,valid = run_spix(vid_s,flows,spix_opts)
        centers = means[...,-2:].contiguous()

        # -- viz marked --
        # print(vid_s.shape,spix.shape,centers.shape,covs.shape)
        marked = mark_spix_vid(vid_s,spix)
        tv_utils.save_image(marked,root/("inloop_marked_%d.png"%scale))

        # -- compute flow --
        dvals,dinds = deform.get_deformation(vid_s[:2],spix,centers,covs)
        flows = deform.inds2flow(dinds,H//scale,W//scale)[:,0]*scale

        # -- warp --
        warped_s = deform.warp_video(vid_s,dvals,dinds)
        warps.append(resize_nn(warped_s,(H,W)))
        delta_s = th.mean((warped_s-vid_s[1:])**2,-3,keepdim=True)
        deltas.append(resize_nn(delta_s,(H,W)))

        # -- info --
        quants = th.tensor([0.8,0.9,0.95,0.99,0.999]).to(device)
        print(th.quantile(delta_s.ravel(),quants))

    warps = th.stack(warps)
    deltas = th.stack(deltas)
    return flows,warps,deltas

# def compute_flow(spix_curr,means_curr,cov_curr,
#                  spix_next,means_next,cov_next):
#     pass

def main():

    # -- get root --
    print("PID: ",os.getpid())
    root = Path("./output/mutliscale_flow/")
    if not root.exists(): root.mkdir()

    # -- config --
    spix_opts = edict()
    spix_opts.sp_size = 15
    spix_opts.nrefine = 30
    spix_opts.niters,spix_opts.inner_niters = 1,1
    spix_opts.i_std,spix_opts.alpha,spix_opts.beta = 0.1,1.,1.
    olist = ["sp_size","i_std","alpha","beta"]
    sp_size,i_std,alpha,beta = [spix_opts[k] for k in olist]
    olist = ["niters","inner_niters","nrefine"]
    niters,inner_niters,nrefine = [spix_opts[k] for k in olist]

    # -- read img/flow --
    # vid = st_spix.data.davis_example(isize=None,nframes=10,vid_names=['tennis'])
    vid = st_spix.data.davis_example(nframes=15,vid_names=['baseball'],data_set="all")
    # vid = vid[0,3:6,:,:128,290-128:290]
    print(vid.shape,vid.min(),vid.max())
    vid = resize(vid[0,[6,5],:,:480,:480],(256,256))
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
    # print("flow: ",flow.shape,
    #       flow[:,0].abs().min().item(),flow[:,0].abs().max().item(),
    #       flow[:,1].abs().min().item(),flow[:,1].abs().max().item(),
    # )

    # -- unpack --
    nspix = spix.max().item()+1
    max_SP = nspix-1
    eps = 1e-10
    # print(spix0[0,28:43,48:64])

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
    flows,warps,deltas = run_multiscale(vid,fflow,spix_opts)
    print(warps.shape)
    tv_utils.save_image(warps[:,0],root/"warps.png")
    tv_utils.save_image(deltas[:,0]/(deltas[:,0].max()+1e-10),root/"deltas.png")

if __name__ == "__main__":
    main()
