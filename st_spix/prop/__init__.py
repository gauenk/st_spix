

# -- basic --
import os
import copy
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
# import st_spix_cuda
import bass_cuda
import prop_cuda
from .param_utils import copy_spix_params,unpack_spix_params_to_list


def stream_bass(vid,flow=None,niters=30,niters_seg=5,
                sp_size=25,pix_var=0.1,alpha_hastings=0.01,potts=10.,sm_start=0):

    # -- load images --
    T,F,H,W = vid.shape
    B,F,H,W = vid.shape

    # -- get flow --
    if flow is None: # run flow [raft]
        from st_spix.flow_utils import run_raft
        flow,bflow = run_raft(th.clip(255.*vid,0.,255.).type(th.uint8))

    # -- bass --
    # img_t = img4bass(vid[None,0])
    # bass_fwd = st_spix_cuda.bass_forward
    # # spix_t,means_t,cov_t,counts_t,ids_t = bass_fwd(img_t,sp_size,pix_var,alpha,potts)
    # spix_t,params_t,ids_t = bass_fwd(img_t,sp_size,pix_var,alpha,potts)

    # bass_forward_cuda(const torch::Tensor imgs,
    #                   int nPixels_in_square_side, float i_std,
    #                   float alpha, float beta){

    # print(sp_size,pix_var,alpha,potts)
    # print(bass_cuda.__file__)
    # print(dir(bass_cuda))
    # print(bass_cuda.SuperpixelParams)

    # -- old version --
    # img_t = img4bass(vid[None,0])
    # spix_t,params_t = bass_cuda.bass_forward(img_t,sp_size,pix_var,alpha_hastings,potts)
    # nspix_t = spix_t.max().item()+1

    # -- updated version --
    img_t = rearrange(vid[None,0],'b f h w -> b h w f').contiguous()
    sm_start = 0
    niters,niters_seg = sp_size,4 # a fun choice from BASS authors
    spix_t,params_t = prop_cuda.bass(img_t,niters,niters_seg,sm_start,
                                     sp_size,pix_var,potts,alpha_hastings)
    nspix_t = spix_t.max().item()+1

    # -- info --
    # print(spix_t.shape)
    # print(spix_t)
    # exit()

    # print(params_t)
    # print(params_t.mu_i)
    # params_copy = copy_spix_params(params_t)
    # print(params_copy.mu_i)
    # exit()

    # -- info --
    # print(flow[:,0].abs().min().item(),flow[:,0].abs().max().item(),
    #       flow[:,1].abs().min().item(),flow[:,1].abs().max().item())

    # -- iterations --
    spix = [spix_t]
    params = [params_t]
    children = []
    missing = []
    pmaps = [th.arange(nspix_t).to(spix_t.device).int()]
    for ix in range(vid.shape[0]-1):

        # -- unpack --
        img_t = rearrange(vid[None,ix+1],'b f h w -> b h w f').contiguous()
        flow_t = flow[[ix]].contiguous()
        spix_tm1 = spix[-1]
        params_tm1 = params[-1]

        # -- run --
        outs = run_prop(img_t,flow_t,spix_tm1,params_tm1,
                        niters,niters_seg,sp_size,pix_var,potts,alpha_hastings,sm_start)
        spix_t,params_t,children_t,missing_t,pmaps_t = outs

        # -- append --
        spix.append(spix_t)
        params.append(params_t)
        children.append(children_t)
        missing.append(missing_t)
        pmaps.append(pmaps_t)

    spix = th.stack(spix)[:,0]
    missing = th.stack(missing)[:,0]
    return spix,params,children,missing,pmaps

def run_prop(img,flow,spix_tm1,params_tm1,
             niters,niters_seg,sp_size,pix_var,potts,alpha_hastings,sm_start):

    # -- unpack --
    nspix_tm1 = spix_tm1.max().item()+1
    eps = 1e-13

    # -- get superpixel flows and shift means --
    B,H,W,F = img.shape
    # print(params_tm1.mu_s[:4]/th.tensor([[W-1,H-1]]).to(img.device))
    params_t0 = params_tm1
    params_tm1 = copy_spix_params(params_tm1)
    # print([p.device for p in unpack_spix_params_to_list(params_tm1)])
    means_tm1 = params_tm1.mu_shape[None,:]
    fxn = st_spix.pool_flow_and_shift_mean
    # print("AHY.")
    # print(spix_tm1.shape)
    # print(params_tm1.mu_shape.shape)
    # print(spix_tm1.max())
    # exit()
    flow_sp,means_shift = fxn(flow,params_tm1.mu_shape[None,:],spix_tm1)
    # print("OH.")
    # print(params_tm1.mu_s[:4]/th.tensor([[W-1,H-1]]).to(img.device))
    # outs = shift_labels(spix_tm1,params_tm1.mu_s,flow_sp) #? mu_s [shifted]/[unshifted]?
    outs = shift_labels(spix_tm1,params_t0.mu_shape,flow_sp)
    # print("YO.")
    spix_prop,missing,missing_mask = outs
    # print("missing_mask.shape: ",missing_mask.shape)
    # print(params_tm1.mu_s[:4]/th.tensor([[W-1,H-1]]).to(img.device))

    # # -- edge case --
    # if missing.numel() == 0:
    #     return spix_prop,means,cov,counts

    # print(means_tm1[0,60,-2:])
    # print(means_shift[0,60,-2:])
    # print(flow_sp.shape)
    # print(flow_sp[0,60])

    # -- fill missing pixels --
    # centers = means_shift[...,-2:].contiguous()
    # print(centers)
    # print(spix_prop.shape,centers.shape,missing.shape,nspix_tm1)
    spix_prop = prop_cuda.fill_missing(spix_prop,params_tm1.mu_shape,missing,nspix_tm1,0)
    assert(spix_prop.min().item() >= 0)
    # print(spix_prop.min().item(),spix_prop.max().item())
    # exit()

    # spix = spix_prop
    # means = means_tm1
    # cov = cov_tm1
    # counts = counts_tm1

    # -- split disconnected --
    spix_prop,children_t,splits = prop_cuda.split_disconnected(spix_prop,nspix_tm1)

    # -- refine missing --
    # print(img.dtype,spix_prop.dtype,missing_mask.dtype,means_tm1.dtype,spix_tm1.dtype)
    # print(spix_prop.min(),spix_prop.max())
    # exit()

    missing_copy = missing_mask.clone()
    missing_mask[...] = 1
    init_border = prop_cuda.find_border(spix_prop)
    # missing_mask = th.logical_or(missing_mask,init_border)
    # missing_mask = ~missing_mask
    # print("missing_mask.shape: ",missing_mask.shape)
    # exit()
    # niters = 10
    prior_map = get_prior_map(spix_prop,children_t,nspix_tm1).int()

    nspix_t = spix_prop.max().item()+1
    # prior_map = th.arange(nspix_t).to(spix_prop.device).int()
    # print(niters,niters_seg)
    # print("py a.")
    # print(nspix_t)
    # print(params_tm1.mu_app.is_contiguous())
    # print(params_tm1.mu_shape.is_contiguous(),params_tm1.mu_shape.device)
    # print(params_tm1.sigma_shape.is_contiguous(),params_tm1.sigma_shape.device)
    # th.cuda.synchronize()
    # print(img)
    # print(spix_prop)
    # exit()
    rescales = th.tensor([1.,1.,1.,1.])
    spix_t,params_t = prop_cuda.refine_missing(img,spix_prop,missing_mask,
                                               params_tm1,prior_map,rescales,
                                               nspix_t,niters,niters_seg,
                                               sp_size,pix_var,potts)
    # print("py b.")

    # nspix_t = spix_t.max().item()+1
    # print("nspix_t: ",spix_t.max().item()+1,spix_t.min().item())

    # spix_t = spix_tm1
    # spix_t = spix_prop
    # params_t = params_tm1

    # spix,means,cov,counts,ids = outs
    # spix_t[th.where(missing_mask>0)]=-1

    # -- bass iters --
    # niters = 10
    # niters_prop = niters
    # sm_start = 0
    # # alpha_hastings = 0.1
    # # print("alpha_hastings: ",alpha_hastings)
    # spix_t,params_t = prop_cuda.prop_bass(img,spix_t,params_tm1,prior_map,
    #                                       nspix_t,niters_prop,niters_seg,sm_start,
    #                                       sp_size,pix_var,potts,alpha_hastings)

    print("nspix_t: ",spix_t.max().item()+1)

    # spix_t,params_t = prop_cuda.refine_missing(img,spix_t,missing_mask,
    #                                            params_t,prior_map,
    #                                            nspix_t,niters,niters_seg,
    #                                            sp_size,pix_var,potts)

    return spix_t,params_t,children_t,missing_copy,prior_map




"""

        Helpers for Superpixel Propogation

"""

def run_fwd_bwd(vid,spix,params,pmaps,sp_size,pix_var,potts,niters_fwd_bwd,niters_ref):
    spix = spix.clone()
    params = [copy_spix_params(p) for p in params]
    niters_seg = 1
    T,F,H,W = vid.shape
    nomissing = th.ones((T,H,W)).bool().to(vid.device)
    # flow_sp,means_shift = fxn(flow,params_tm1.mu_s[None,:],spix_tm1)
    for ix in range(niters_fwd_bwd):
        for t in range(T):
            img_t = rearrange(vid[[t]],'1 f h w -> 1 h w f').contiguous()
            nspix_t = spix[t].max().item()+1
            tp = (t+1) % T
            rescales = th.tensor([1.,1.,1.,1.])
            spix[t],params[t] = prop_cuda.refine_missing(img_t,spix[[t]],nomissing,
                                                         params[tp],pmaps[tp],rescales,
                                                         nspix_t,niters_ref,niters_seg,
                                                         sp_size,pix_var,potts)
    return spix,params


def get_prior_map(spix_split,children,nspix_before_split):
    # -- get prior map --
    # print("nspix: ",nspix,spix.min().item(),spix.max().item())
    prior_map = th.arange(spix_split.max().item()+1).to(spix_split.device)
    # print(prior_map.max(),len(prior_map),nspix_split)
    prior_map[nspix_before_split+1:] = -1
    for i in range(children.shape[1]):
        if (children[:,i]<0).all(): break
        old_locations = th.where(children[:,i]>=0)[0]
        new_locations = children[:,i][old_locations].long()
        # print(new_locations,old_locations)
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
    # print(grid.shape)
    # print(grid[0,:,:3,:3])
    grid = st_spix.sp_pool_from_spix(grid,spix)
    # print(grid[0,:,:3,:3])
    # print(grid.shape)
    gscatter,gcnts = st_spix.scatter.run(grid,flow,swap_c=True)
    # gscatter,gcnts = st_spix.scatter.run_v1(grid,flow)
    # print("gcnts.shape: ",gcnts.shape)
    # exit()

    # -- invalidate --
    eps_ub = eps
    eps_lb = eps
    invalid = th.where(th.logical_or(gcnts<1-eps_lb,1+eps_ub<gcnts))
    for i in range(2):
        gscatter[:,i][invalid] = -100.
    # tmp = gscatter.clone()
    # tmp2 = th.stack([means[:,-2]/(W-1),means[:,-1]/(H-1)],-1)

    # -- all pairwise differences --
    locs = th.stack([means[:,-2]/(W-1),means[:,-1]/(H-1)],-1)
    gscatter = rearrange(gscatter,'b f h w -> (b h w) f')
    dists = th.cdist(gscatter,locs)
    # print("dists.shape: ",dists.shape)

    # print(tmp[0,0,:5,20:20+5])
    # print(tmp[0,1,:5,20:20+5])
    # print("tmp2.shape: ",tmp2.shape)
    # print(tmp2[2])
    # print(tmp2[3])

    # print(dists.argmin(1).reshape(H,W)[:10,:10])
    # print(dists[:,2].reshape(H,W)[:5,10:15])
    # print(dists[:,3].reshape(H,W)[:5,10:15])

    # -- gather --
    spix_grid = th.arange(means.shape[0]).to(spix.device)
    shifted_spix = spix_grid[dists.argmin(1)].int()
    shifted_spix = rearrange(shifted_spix,'(b h w) -> b h w',h=H,w=W)
    shifted_spix[invalid] = -1

    # -- get missing --
    invalid = th.logical_or(gcnts<1-eps,1+eps<gcnts)
    missing = th.where(invalid.ravel())[0][None,:].type(th.int)

    return shifted_spix,missing,invalid

