
import math
import torch as th
import numpy as np
import pandas as pd
from pathlib import Path
import st_spix
from st_spix import flow_utils
import st_spix_cuda
import st_spix_original_cuda
from st_spix import flow_utils as futils
import torchvision.io as iio
from einops import rearrange,repeat
from skimage.segmentation import mark_boundaries
import torchvision.utils as tv_utils
import torch.nn.functional as th_f

import seaborn as sns
import matplotlib.pyplot as plt
from dev_basics.utils.metrics import compute_psnrs

import stnls

from dev_basics import flow as flow_pkg
from dev_basics.utils.timer import ExpTimer

from easydict import EasyDict as edict

def img_for_bass(img,device="cuda"):
    img= (th.clip(img,0.,1.)*255.).type(th.uint8)
    img = rearrange(img,'f h w -> 1 h w f').to(device)
    return img
def swap_c(img):
    return rearrange(img,'... h w f -> ... f h w')

def get_warp(img,flow,warp_type="grid"):
    if warp_type == "stnls":
        return get_stnls_warp(img,flow)
    elif warp_type == "grid":
        return get_grid_warp(img,flow)
    else:
        raise NotImplemented("")

def get_grid_warp(img,flow):
    warp = futils.flow_warp(img, flow,
                            interp_mode='bilinear',
                            padding_mode='reflection',
                            align_corners=True)
    return warp

def get_stnls_warp(img,flow):
    vid = rearrange(img,'1 f h w -> 1 1 f h w')
    flows_k = rearrange(flow,'b f h w -> b 1 h w 1 f')
    zeros = th.zeros_like(flows_k[...,:1])
    flows_k = th.cat([zeros,flows_k],-1)
    stacking = stnls.agg.NonLocalGather(1,1,itype="float")
    ones = th.ones_like(flows_k[...,0])
    flows_k = flows_k.contiguous()
    stack = stacking(vid,ones,flows_k)[0,0,0]
    # print(stack.shape)
    return stack

def rot_flow(flow,deg):
    rads = np.deg2rad(deg)
    cos_theta = np.cos(rads)
    sin_theta = np.sin(rads)
    rmat = th.tensor([[cos_theta, -sin_theta],
                      [sin_theta, cos_theta]],
                     dtype=th.double,device=flow.device)
    B,_,H,W = flow.shape
    dtype = flow.dtype
    flow = rearrange(flow,'b f h w -> (b h w) f').double()
    flow = flow @ rmat
    flow = rearrange(flow,'(b h w) f -> b f h w',b=B,h=H)
    return flow.type(dtype)

def run_stnls(img0,img1,in_flow,ws,ps,full_ws=False):
    s0 = 1
    s1 = 1
    wt = 1
    k = 1
    # in_flow = in_flow.flip(0)
    # in_flow[...] = 0.
    # print(img0.shape,img1.shape,in_flow.shape)
    search_p = stnls.search.PairedSearch(ws,ps,k,
                                         nheads=1,dist_type="l2",
                                         stride0=s0,stride1=s1,
                                         self_action=None,use_adj=False,
                                         full_ws=full_ws,itype="float")
    _,flows_k = search_p(img0,img1,in_flow) # b hd h w k f
    out_flow = rearrange(flows_k,'b 1 h w 1 f -> b f h w')
    return out_flow

def run_exp(cfg):

    # -- config --
    root = Path("./output/flow_remap")
    if not root.exists(): root.mkdir(parents=True)

    # -- load images --
    vid = st_spix.data.davis_example()[0,:2]
    img0,img1 = img_for_bass(vid[0]),img_for_bass(vid[1])
    B,H,W,F = img0.shape

    # -- run img0 bass --
    npix_in_side = 40
    i_std,alpha,beta = 0.018,2.,2.
    spix,means,cov,counts = st_spix_original_cuda.bass_forward(img0,npix_in_side,
                                                               i_std,alpha,beta)

    # -- format for remaining code --
    img0,img1 = swap_c(img0/255.),swap_c(img1/255.)

    # -- run flow --
    flows = flow_pkg.run(vid[None,:],sigma=0.0,ftype="cv2")
    flow = flows.bflow[0,1][None,:]
    futils.viz_flow_quiver(root/"flow",flow)

    # -- spix pool --
    flow_sp = st_spix.sp_pool_from_spix(flow,spix)
    warp_sp = get_warp(img0,flow_sp.flip(1))
    print("[sp] PSNR: ",compute_psnrs(warp_sp,img1))
    tv_utils.save_image(warp_sp,root/"warped_sp.png")
    futils.viz_flow_quiver(root/"flow_sp",flow_sp)

    # -- stnls --
    ws,ps = 11,1
    flow_stnls = run_stnls(img1,img0,flow,ws,ps,full_ws=False)
    # flow_stnls = st_spix.sp_pool_from_spix(flow_stnls,spix)
    warp_stnls = get_warp(img0,flow_stnls)
    print("[stnls] PSNR: ",compute_psnrs(warp_stnls,img1))
    tv_utils.save_image(warp_stnls,root/"warped_stnls.png")
    futils.viz_flow_quiver(root/"flow_stnls",flow_stnls.flip(1))

    # -- warp image --
    alpha = 1.
    # flow_sgd = flow.clone().requires_grad_(True)
    flow_sgd = flow_sp.clone().requires_grad_(True)
    # flow_sgd = flow_stnls.clone().requires_grad_(True)
    nsteps = 2000
    for curr_iter in range(nsteps):
        warp_sgd = get_warp(img0,flow_sgd.flip(1))
        error = th.mean((warp_sgd - img1)**2)
        error.backward()
        if curr_iter == (nsteps-1):
            print("error: ",error,th.norm(flow_sgd.grad).item())
        with th.no_grad():
            flow_sgd -= alpha * flow_sgd.grad/th.norm(flow_sgd.grad)
        flow_sgd.grad[...] = 0.
    print("[sgd] PSNR: ",compute_psnrs(warp_sgd,img1))
    tv_utils.save_image(warp_sgd,root/"warped_sgd.png")
    futils.viz_flow_quiver(root/"flow_sgd",flow_sgd)

def main():


    cfg = edict()
    cfg.name = "a"
    run_exp(cfg)

if __name__ == "__main__":
    main()
