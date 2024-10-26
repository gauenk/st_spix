

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
from .param_utils import copy_spix_params,unpack_spix_params_to_list,spix_dict_to_list
from .param_utils import KEYS as param_names


def indepent_bass(vid,niters=30,niters_seg=5,
                  sp_size=25,pix_var=0.1,alpha_hastings=0.01,potts=10.,
                  sm_start=0,rgb2lab=True):

    # # -- rgb 2 lab --
    # if rgb2lab:
    #     vid_lab = st_spix.utils.vid_rgb2lab_th(vid.clone(),normz=False)
    # else:
    #     vid_lab = vid

    # sp_size = 20
    # pix_var_g = [0.08,0.07,0.06]
    # potts = 4.

    # -- updated version --
    spix = []
    for ix in range(vid.shape[0]):
        # print("ix: ",ix)
        # pix_var = pix_var_g[ix]

        # img_t = rearrange(vid_lab[None,ix],'b f h w -> b h w f').contiguous()
        img_t = vid[[ix]]
        sm_start = 0
        niters,niters_seg = sp_size,4 # a fun choice from BASS authors
        spix_t,params_t = prop_cuda.bass(img_t,niters,niters_seg,sm_start,
                                         sp_size,pix_var,potts,alpha_hastings)


        # print("vid.min(),vid.max(): ",vid.min(),vid.max(),vid.shape)
        # print("img_t.shape: ",img_t.shape)
        # print("dir(bass_cuda): ",dir(bass_cuda))
        img_t = (vid[[ix]]*255).type(th.uint8).contiguous()
        # img_t = rearrange(img_t,'b f h w -> b h w f').contiguous()
        # print("[b] img_t.shape: ",img_t.shape)

        # -- vanilla bass --
        # fwd = bass_cuda.bass_forward
        # print(pix_var,alpha_hastings,potts)
        # spix_t,params_t = fwd(img_t,sp_size,pix_var,alpha_hastings,potts)

        nspix_t = spix_t.max().item()+1
        spix.append(spix_t)

    # print(spix[0].shape)
    # print(spix[0][:3,:3])

    return th.cat(spix)

def stream_bass(vid,flow=None,niters=30,niters_seg=5,
                sp_size=25,pix_var=0.01,alpha_hastings=0.1,potts=10.,
                sm_start=0,rgb2lab=True):

    # -- load images --
    T,H,W,F = vid.shape
    B,H,W,F = vid.shape

    # -- get flow --
    if flow is None: # run flow [raft]
        from st_spix.flow_utils import run_raft
        flow,bflow = run_raft(th.clip(255.*vid,0.,255.).type(th.uint8))
        exit()

    # -- rgb 2 lab --
    # vid_lab = st_spix.utils.vid_rgb2lab(vid.clone(),normz=False)
    # if rgb2lab:
    #     vid_lab = st_spix.utils.vid_rgb2lab_th(vid.clone(),normz=False)
    # else:
    #     vid_lab = vid
    # print("delta: ",th.mean((vid_lab - vid_lab1)**2))
    # exit()

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

    th.cuda.synchronize()
    T,H,W,F = vid.shape
    npix = 1.*H*W
    # print("npix / sp_size: ",npix / (sp_size*sp_size))
    # th.cuda.synchronize()

    # -- updated version --
    # img_t = rearrange(vid_lab[None,0],'b f h w -> b h w f').contiguous()
    img_t = vid[[0]]
    sm_start = 0
    # niters,niters_seg = sp_size,4 # a fun choice from BASS authors
    spix_t,params_t = prop_cuda.bass(img_t,niters,niters_seg,sm_start,
                                     sp_size,pix_var,potts,alpha_hastings)

    # print(spix_t.sum())
    # print("post.")
    # th.cuda.synchronize()

    # img_t = img4bass(vid[None,0])
    # spix_t,params_t = prop_cuda.bass(img_t,niters,niters_seg,sm_start,
    #                                  sp_size,pix_var,potts,alpha_hastings)
# bass_forward_cuda(const torch::Tensor imgs,
#                   int nPixels_in_square_side,
#                   float pix_cov, float alpha, float beta){

    params_t = edict({key:getattr(params_t,key).clone() for key in param_names})
    nspix_t = spix_t.max().item()+1
    # params_t.ids[...] = th.arange(len(params_t.ids)).to(vid.device)


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
    tmp = []
    for ix in range(vid.shape[0]-1):

        # -- unpack --
        img_t = vid[[ix+1]]
        flow_t = flow[[ix]]
        spix_tm1 = spix[-1]
        params_tm1 = params[-1]
        # print([p.ids for p in params])
        # print("a: ",params_tm1.ids)
        # print(hex(id(params_tm1)),hex(id(params_tm1.ids)))
        # print(hex(id(params_tm1.ids)),hex(id(params_tm1.ids[0])))

        # -- run --
        outs = run_prop(img_t,flow_t,spix_tm1,params_tm1,
                        niters,niters_seg,sp_size,pix_var,potts,alpha_hastings,sm_start)
        spix_t,params_t,children_t,missing_t = outs
        # print(spix_t.shape)
        # print(spix_t.sum())

        # th.cuda.synchronize()
        # print("hey: ",params_t.ids)
        # print(hex(id(params_t.ids)),hex(id(params_t.ids[0])))
        # print(dir(params_t))


        # -- append --
        spix.append(spix_t)
        params.append(params_t)
        # params[0] = params_t
        children.append(children_t)
        missing.append(missing_t)

    spix = th.stack(spix)[:,0]
    missing = th.stack(missing)[:,0]
    return spix,params,children,missing

def run_prop(img,flow,spix_tm1,params_tm1,niters,niters_seg,
             sp_size,pix_var,potts,alpha_hastings,sm_start):

    # return spix_tm1,params_tm1,th.tensor([[]]),th.tensor([[]]),th.tensor([[]])

    # print(params_tm1.ids)
    # -- unpack --
    nspix_tm1 = spix_tm1.max().item()+1
    eps = 1e-13
    # uniq_spix = spix_tm1

    # -- get superpixel flows and shift means --
    B,H,W,F = img.shape
    # print(params_tm1.mu_s[:4]/th.tensor([[W-1,H-1]]).to(img.device))
    params_t0 = params_tm1
    params_tm1 = copy_spix_params(params_tm1)
    # print([p.device for p in unpack_spix_params_to_list(params_tm1)])
    mu_shape_tm1 = params_tm1.mu_shape[None,:]
    spix_ids = params_tm1.ids

    # print("AY")
    # print(spix_ids)
    # print("AHY.")
    # print(spix_tm1.shape)
    # print(params_tm1.mu_shape.shape)
    # print(spix_tm1.max())
    # exit()
    # print("a: ",params_t0.mu_shape.shape,spix_tm1.max().item()+1)

    # -- flow and shift --
    # print(len(th.unique(spix_tm1)))
    # print(spix_tm1.max()+1,params_tm1.mu_shape[None,:].shape)
    # print(mu_shape_tm1.shape)
    # print(spix_ids,spix_tm1.max()+1)
    # exit()
    # print("len(spix_ids): ",len(th.unique(spix_tm1)))
    fxn = st_spix.pool_flow_and_shift_mean
    flow_sp,flow_down,means_shift = fxn(flow,mu_shape_tm1,spix_tm1,spix_ids)
    params_tm1.prior_mu_shape[...] = params_tm1.mu_shape[...]
    # print("OH.")
    # print(params_tm1.mu_s[:4]/th.tensor([[W-1,H-1]]).to(img.device))

    # -- debug --
    # data1 = th.randn(1,320,480)>2
    # mask1 = th.randn(1,320,480)>0
    # data1.masked_fill_(mask1,1)

    # -- [shift_labels] updated one --
    flow_down = flow_down.round().int()
    # spix_prop,missing_mask = prop_cuda.shift_labels(spix_tm1,flow_down)
    # missing_mask = th.logical_or(missing_mask > 0,spix_prop<0)
    spix_prop,counts = prop_cuda.shift_labels(spix_tm1,flow_down)
    spix_prop[counts!=1] = -1
    missing_mask = counts != 1


    # _spix,_flow = spix0[None,:],flow_down
    # _spix_toshift = spix0[None,:,:,None].clone().float()
    # _sizes = th.bincount(_spix.ravel())[None,:].int()
    # ishift,counts,contrib = prop_cuda.shift_tensor(_spix_toshift,_spix,_flow)
    # select = prop_cuda.shift_order(contrib,_sizes)
    # spix_prop,_bdcounts = prop_cuda.shift_tensor_ordered(_spix_toshift,_spix,_flow,select)
    # spix_prop = spix_prop[:,:,:,0].int()
    # print(spix_prop.shape)


    # # spix_prop = spix_prop * (1-missing_mask) + -1 * missing_mask
    # missing_mask = missing_mask>0
    # print(missing_mask)
    # print(missing_mask.sum())
    # exit()
    # print(th.logical_and(spix_prop<0,missing_mask==0).sum())
    # print(th.logical_and(spix_prop<0,counts==1).sum())
    # exit()


    # print(spix_prop[0,30:40,30:40])
    # missing_mask = missing_mask>0
    # th.cuda.synchronize()
    # print(spix_prop.shape,missing_mask.shape)

    # missing_mask = missing_mask>0
    # # print(missing_mask.sum())
    # spix_prop[missing_mask] = -1
    # # print((1.*spix_prop).sum()+1)

    # # return spix_prop,params_tm1,th.tensor([[]]),th.tensor([[]]),th.tensor([[]])
    # missing_mask = th.logical_or(missing_mask > 0,spix_prop<0)
    # missing_mask = missing_mask>0
    # spix_prop = spix_prop.masked_fill(missing_mask>0, -1)
    # spix_prop = spix_prop.masked_fill(missing_mask>0, -1)
    # spix_prop = spix_prop.masked_fill(missing_mask>0, -1)

    # print(spix_prop)
    # print(missing_mask)
    # print(spix_prop.shape,missing_mask.shape)#,missing_mask.sum())

    # return spix_tm1,params_tm1,th.tensor([[]]),th.tensor([[]]),th.tensor([[]])
    # return data1,params_tm1,th.tensor([[]]),th.tensor([[]]),th.tensor([[]])


    # -- [shift_labels] old one --
    # outs = shift_labels(spix_tm1,params_t0.mu_shape,flow_sp)
    # spix_prop,missing_mask = outs
    # print(spix_prop)
    # print(missing_mask)
    # print(spix_prop.shape,missing_mask.shape)#,missing_mask.sum())
    # exit()

    # return spix_tm1,params_tm1,th.tensor([[]]),th.tensor([[]]),th.tensor([[]])


    # print("missing_mask.shape: ",missing_mask.shape)
    # print(params_tm1.mu_s[:4]/th.tensor([[W-1,H-1]]).to(img.device))

    # # -- edge case --
    # if missing.numel() == 0:
    #     return spix_prop,means,cov,counts

    # params_tm1.sigma_app[...] = pix_var**2/4.
    # curr_params[k].prior_mu_app = prev_params[k].mu_app;
    # curr_params[k].prior_sigma_app = prev_params[k].sigma_app;
    # curr_params[k].prior_mu_app_count = max(rescale_mu_app * count,1.0);
    # curr_params[k].prior_sigma_app_count = max(rescale_sigma_app * count,1.0);

    # print(means_tm1[0,60,-2:])
    # print(means_shift[0,60,-2:])
    # print(flow_sp.shape)
    # print(flow_sp[0,60])

    # return spix_tm1,params_tm1,th.tensor([[]]),th.tensor([[]]),th.tensor([[]])

    # -- fill missing pixels --
    # centers = means_shift[...,-2:].contiguous()
    # print(centers)
    # print(spix_prop.shape,centers.shape,missing.shape,nspix_tm1)
    missing = th.where(missing_mask.ravel())[0][None,:].type(th.int)
    # print("spix_prop.shape: ",spix_prop.shape)
    # print("missing.shape: ",missing.shape)
    # exit()

    # th.save(spix_tm1,"spix_tm1.pth")
    # th.save(spix_prop,"spix_prop_prefill.pth")

    spix_prop = prop_cuda.fill_missing(spix_prop,params_tm1.mu_shape,missing,0)
    # assert(spix_prop.min().item() >= 0)
    # print(spix_prop.min().item(),spix_prop.max().item())
    # exit()

    # print("hi.")

    # return spix_tm1,params_tm1,th.tensor([[]]),th.tensor([[]]),th.tensor([[]])

    # spix = spix_prop
    # means = means_tm1
    # cov = cov_tm1
    # counts = counts_tm1

    # -- split disconnected --
    # spix_prop,children_t,splits = prop_cuda.split_disconnected(spix_prop,nspix_tm1)

    # -- refine missing --
    # print(img.dtype,spix_prop.dtype,missing_mask.dtype,means_tm1.dtype,spix_tm1.dtype)
    # print(spix_prop.min(),spix_prop.max())
    # exit()

    # init_border = prop_cuda.find_border(spix_prop)
    # missing_mask = th.logical_or(missing_mask,init_border)
    # missing_mask = ~missing_mask
    # print("missing_mask.shape: ",missing_mask.shape)
    # exit()
    # niters = 10

    # prior_map = th.tensor([[]])
    # missing_copy = th.tensor([[]])

    # -- update missing --
    missing_copy = missing_mask.clone()
    missing_mask[...] = 1

    # -- get prior map --
    # prior_map = get_prior_map(spix_prop,children_t,nspix_tm1).int()

    # -- spoof --
    children_t = th.tensor([[]])
    nspix_t = spix_prop.max().item()+1

    # spix_t,params_t = spix_prop,params_tm1
    # prior_map = th.arange(nspix_t).to(spix_prop.device).int()
    # print(niters,niters_seg)
    # print("py a.")
    # print(nspix_t)
    # print(params_tm1.mu_app.is_contiguous())
    # print(params_tm1.mu_shape.is_contiguous(),params_tm1.mu_shape.device)
    # print(params_tm1.sigma_shape.is_contiguous(),params_tm1.sigma_shape.device)
    th.cuda.synchronize()
    # print(img)
    # print(spix_prop)
    # exit()

    # -- try different variations --
    # print(params_tm1.ids)
    device = params_tm1.ids.device
    # params_tm1.ids[...] = th.arange(len(params_tm1.ids)).to(device)
    # # print(params_tm1.ids)
    # # params_tm1.sigma_app[...] /=
    # # params_tm1.prior_sigma_app[...] = pix_var**2/4.
    # # print(params_tm1.sigma_app)
    rescales = th.tensor([1.,1.,1.,1.])
    # # print("niters: ",niters)
    # spix_t,params_t = spix_tm1,params_tm1

    # -- save spix for debug --
    # print("hi.")
    # th.save(spix_tm1,"spix_tm1.pth")
    # th.save(spix_prop,"spix_prop.pth")

    # -- convert [see description below] --
    # niters = 10#sp_size
    niters = sp_size
    if isinstance(params_tm1,edict):
        params_tm1 = spix_dict_to_list(params_tm1)
    spix_t,params_t,log_t = prop_cuda.refine_missing(img,spix_prop,missing_mask,
                                                     params_tm1,
                                                     nspix_t,niters,niters_seg,
                                                     sp_size,pix_var,potts)
    # spix_t = spix_prop
    # params_t = params_tm1
    # log_t = None

    # spix_t = spix_t.clone()
    # -- Pybind11 data-structure has problems [it was memory access error elsewhere] --
    #  -- tensor memory is randomly set of invalid values, so we copy to a dict --
    params_t = edict({key:getattr(params_t,key).clone() for key in param_names})

    # spix_t,params_t = spix_tm1,params_tm1
    # print("post.")
    # print(params_t.ids)

    # print("b: ",params_t.mu_shape.shape,spix_t.max().item()+1)
    # print(log_t)

    # print("py b.")
    # nspix_t = spix_t.max().item()+1
    # print("nspix_t: ",spix_t.max().item()+1,spix_t.min().item())

    # spix_t = spix_tm1
    # spix_t = spix_prop
    # params_t = params_tm1

    # spix,means,cov,counts,ids = outs
    # spix_t[th.where(missing_mask>0)]=-1

    # print(spix_t)
    # print(spix_t.min(),spix_t.max())
    # print(params_tm1.mu_shape.shape)
    # # print(params_tm1.mu_shape
    # exit()


    # -- bass iters --
    # niters = 10

    # sm_start = 0
    # # alpha_hastings = 0.1
    # # print("alpha_hastings: ",alpha_hastings)
    # niters_prop = niters
    niters_prop = sp_size
    # niters_prop = 0
    # _alpha_hastings = 0.
    _alpha_hastings = alpha_hastings
    spix_t,params_t = prop_cuda.prop_bass(img,spix_t,params_tm1,
                                          niters_prop,niters_seg,sm_start,
                                          sp_size,pix_var,potts,_alpha_hastings)
    # print("nspix_t: ",spix_t.max().item()+1)

    # spix_t,params_t = prop_cuda.refine_missing(img,spix_t,missing_mask,
    #                                            params_t,prior_map,
    #                                            nspix_t,niters,niters_seg,
    #                                            sp_size,pix_var,potts)
    # exit()

    return spix_t,params_t,children_t,missing_copy




"""

        Helpers for Superpixel Propogation

"""

def run_fwd_bwd(vid,spix,params,sp_size,pix_var,potts,niters_fwd_bwd,niters_ref):
    spix = spix.clone()
    params = [copy_spix_params(p) for p in params]
    niters_seg = 1
    device = vid.device
    T,F,H,W = vid.shape
    vid_lab = st_spix.utils.vid_rgb2lab(vid,normz=False)
    nomissing = th.ones((T,H,W)).bool().to(vid.device)
    # flow_sp,means_shift = fxn(flow,params_tm1.mu_s[None,:],spix_tm1)
    for ix in range(niters_fwd_bwd):
        for t in range(T):
            img_t = rearrange(vid_lab[[t]],'1 f h w -> 1 h w f').contiguous()
            nspix_t = spix[t].max().item()+1
            tp = (t+1) % T
            params_adj = copy_spix_params(params[tp])
            params_adj.ids[...] = th.arange(len(params_adj.ids)).to(device)
            params_adj.sigma_app[...] = pix_var**2/4.
            params_adj.prior_sigma_app[...] = pix_var**2/4.
            fwd = prop_cuda.refine_missing
            spix[t],params[t],log = fwd(img_t,spix[[t]],nomissing,
                                        params_adj,nspix_t,niters_ref,
                                        niters_seg,sp_size,pix_var,potts)
            # print(log)
    return spix,params


# def get_prior_map(spix_split,children,nspix_before_split):
#     # -- get prior map --
#     # print("nspix: ",nspix,spix.min().item(),spix.max().item())
#     prior_map = th.arange(spix_split.max().item()+1).to(spix_split.device)
#     # print(prior_map.max(),len(prior_map),nspix_split)
#     prior_map[nspix_before_split+1:] = -1
#     for i in range(children.shape[1]):
#         if (children[:,i]<0).all(): break
#         old_locations = th.where(children[:,i]>=0)[0]
#         new_locations = children[:,i][old_locations].long()
#         # print(new_locations,old_locations)
#         prior_map[new_locations] = old_locations
#     assert th.all(prior_map>=0).item(), "Must all be valid after the splitting disconnected regions"
#     return prior_map

def shift_labels(spix,means,flow,eps=1e-13):


    # -- unpack --
    B,H,W = spix.shape
    # flow = flow.clone()

    # -- scatter --
    grid = futils.index_grid(H,W,dtype=spix.dtype,device=spix.device,
                             normalize=True,stack_dim=-1)
    grid = st_spix.sp_pool_from_spix(grid,spix)
    gscatter,gcnts = st_spix.scatter.run(grid,flow,swap_c=False)
    # gscatter,gcnts = th.tensor([]),th.tensor([])

    # -- invalidate --
    eps_ub,eps_lb = eps,eps
    invalid = th.where(th.logical_or(gcnts<1-eps_lb,1+eps_ub<gcnts))
    gscatter[...,0][invalid] = -100.
    gscatter[...,1][invalid] = -100.

    # -- all pairwise differences --
    locs = th.stack([means[:,-2]/(W-1),means[:,-1]/(H-1)],-1)
    gscatter = rearrange(gscatter,'b h w f -> (b h w) f')
    dists = th.cdist(gscatter,locs)
    # dists = 1.*th.zeros((65536, 83),device=spix.device)

    # -- gather --
    spix_grid = th.arange(means.shape[0]).to(spix.device)
    shifted_spix = spix_grid[dists.argmin(1)].int()
    shifted_spix = rearrange(shifted_spix,'(b h w) -> b h w',h=H,w=W)
    shifted_spix[invalid] = -1
    # shifted_spix = spix

    # -- get missing --
    invalid = th.logical_or(gcnts<1-eps,1+eps<gcnts)
    missing_mask = invalid
    missing = th.where(invalid.ravel())[0][None,:].type(th.int)

    # invalid = th.tensor([])
    # missing = th.tensor([])


    return shifted_spix,missing_mask

