




# -- basic --
import os
import torch as th
import numpy as np
import pandas as pd
from pathlib import Path
import torchvision.io as iio
from einops import rearrange,repeat
from easydict import EasyDict as edict

# -- image io --
import torch.nn.functional as th_f
import torchvision.utils as tv_utils

# -- helpers --
import st_spix
from st_spix import flow_utils
from st_spix.spix_utils import img4bass,mark_spix
from st_spix import flow_utils as futils

# -- dev basics --
from dev_basics import flow as flow_pkg
from dev_basics.utils.timer import ExpTimer,TimeIt
from dev_basics.utils.metrics import compute_psnrs

# -- c++ code --
import st_spix_cuda
import prop_cuda


def stream_bass(vid,flow=None,niters=30,niters_seg=5,
                sp_size=80,pix_cov=0.1,alpha=0.01,potts=10.):

    # -- load images --
    T,F,H,W = vid.shape
    B,F,H,W = vid.shape

    # -- get flow --
    if flow is None: # run flow [raft]
        from st_spix.flow_utils import run_raft
        flow,bflow = run_raft(th.clip(255.*vid,0.,255.).type(th.uint8))

    # -- bass --
    img_t = img4bass(vid[None,0])
    bass_fwd = st_spix_cuda.bass_forward
    # spix_t,means_t,cov_t,counts_t,ids_t = bass_fwd(img_t,sp_size,pix_cov,alpha,potts)
    spix_t,params_t,ids_t = bass_fwd(img_t,sp_size,pix_cov,alpha,potts)

    print(flow[:,0].abs().min().item(),flow[:,0].abs().max().item(),
          flow[:,1].abs().min().item(),flow[:,1].abs().max().item())

    # -- iterations --
    spix = [spix_t]
    params = [params_t]
    children = []
    for ix in range(vid.shape[0]-1):

        # -- unpack --
        img_t = rearrange(vid[None,ix+1],'b f h w -> b h w f').contiguous()
        flow_t = flow[[ix]].contiguous()
        spix_tm1 = spix[-1]
        params_tm1 = params[-1]

        # -- run --
        outs = run_prop(img_t,flow_t,spix_tm1,params_tm1,
                        niters,niters_seg,sp_size,pix_cov,alpha,potts)
        spix_t,means_t,cov_t,counts_t,children_t = outs

        # -- append --
        spix.append(spix_t)
        children.append(children_t)

    spix = th.stack(spix)[:,0]
    return spix,children

def run_prop(img,flow,spix_tm1,params_tm1,
             niters,niters_seg,sp_size,pix_cov,alpha,potts):

    # -- unpack --
    nspix = spix_tm1.max().item()+1
    eps = 1e-13

    # -- get superpixel flows and shift means --
    fxn = st_spix.pool_flow_and_shift_mean
    flow_sp,means_shift = fxn(flow,means_tm1.clone(),spix_tm1)
    outs = shift_labels(spix_tm1,means_tm1[0],flow_sp)
    spix_prop,missing,missing_mask = outs


    # # -- edge case --
    # if missing.numel() == 0:
    #     return spix_prop,means,cov,counts

    # print(means_tm1[0,60,-2:])
    # print(means_shift[0,60,-2:])
    # print(flow_sp.shape)
    # print(flow_sp[0,60])

    # -- fill missing pixels --
    centers = means_shift[...,-2:].contiguous()
    # print(centers)
    # print(spix_prop.shape,centers.shape,missing.shape,nspix)
    spix_prop = prop_cuda.fill_missing(spix_prop,centers,missing,nspix,0)
    # print(spix_prop.min().item(),spix_prop.max().item())
    # exit()

    spix = spix_prop
    means = means_tm1
    cov = cov_tm1
    counts = counts_tm1

    # -- split disconnected --
    spix_prop,children,splits = prop_cuda.split_disconnected(spix_prop,nspix)

    # -- refine missing --
    # print(img.dtype,spix_prop.dtype,missing_mask.dtype,means_tm1.dtype,spix_tm1.dtype)
    # print(spix_prop.min(),spix_prop.max())
    # exit()

    # missing_mask[...] = 1
    # missing_mask = ~missing_mask
    # print("missing_mask.shape: ",missing_mask.shape)
    # exit()
    prior_map = get_prior_map(spix_prop,children)
    spix = prop_cuda.refine_missing(img,spix_prop,missing_mask,
                                    params_tm1,prior_map,
                                    nspix,niters,niters_seg,
                                    sp_size,pix_cov,potts)
    # spix,means,cov,counts,ids = outs
    # spix[th.where(missing_mask>0)]=-1

    # -- then bass iters --
    # ...

    return spix,means,cov,counts,children


"""

        Helpers for Superpixel Propogation

"""

def get_prior_map(spix_split,children):
    # -- get prior map --
    # print("nspix: ",nspix,spix.min().item(),spix.max().item())
    prior_map = th.arange(spix_split.max().item()+1).to(spix_split.device)
    # print(prior_map.max(),len(prior_map),nspix_split)
    prior_map[nspix+1:] = -1
    for i in range(children.shape[1]):
        if (children[:,i]<0).all(): break
        old_locations = th.where(children[:,i]>=0)[0]
        new_locations = children[:,i][old_locations].long()
        print(new_locations,old_locations)
        prior_map[new_locations] = old_locations
    assert th.all(prior_map>=0).item(), "Must all be valid after the splitting disconnected regions"
    return prior_map

def shift_labels(spix,means,flow,eps=1e-13):

    # -- unpack --
    B,H,W = spix.shape
    flow = flow.clone()

    # -- scatter --
    grid = futils.index_grid(H,W,dtype=spix.dtype,
                             device=spix.device,normalize=True)
    grid = st_spix.sp_pool_from_spix(grid,spix)
    gscatter,gcnts = st_spix.scatter.run(grid,flow,swap_c=True)

    # -- invalidate --
    invalid = th.where(th.logical_or(gcnts<1-eps,1+eps<gcnts))
    for i in range(2):
        gscatter[:,i][invalid] = -100.

    # -- all pairwise differences --
    locs = th.stack([means[:,-2]/(W-1),means[:,-1]/(H-1)],-1)
    gscatter = rearrange(gscatter,'b f h w -> (b h w) f')
    dists = th.cdist(gscatter,locs)

    # -- gather --
    spix_grid = th.arange(means.shape[0]).to(spix.device) #  I think "0" in invalid?
    shifted_spix = spix_grid[dists.argmin(1)].int()
    shifted_spix = rearrange(shifted_spix,'(b h w) -> b h w',h=H,w=W)
    shifted_spix[invalid] = -1

    # -- get missing --
    invalid = th.logical_or(gcnts<1-eps,1+eps<gcnts)
    missing = th.where(invalid.ravel())[0][None,:].type(th.int)

    return shifted_spix,missing,invalid

