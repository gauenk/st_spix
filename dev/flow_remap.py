
import torch as th
import numpy as np
import pandas as pd
from pathlib import Path
import st_spix
from st_spix import flow_utils
import st_spix_cuda

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


# -- colorwheel --
import matplotlib.pyplot as plt
import matplotlib as mpl
from matplotlib.colors import LinearSegmentedColormap, ListedColormap

def img_for_bass(img,device="cuda"):
    img= (th.clip(img,0.,1.)*255.).type(th.uint8)
    img = rearrange(img,'f h w -> 1 h w f').to(device)
    return img
def to_th(tensor):
    return th.from_numpy(tensor)
def swap_c(img):
    return rearrange(img,'... h w f -> ... f h w')

def remap_spix(spix,ids,device="cuda"):
    B,H,W = spix.shape
    spix = spix*1.
    spix_remap = th.cdist(spix.ravel()[:,None],1.*ids[:,None]).argmin(1)
    spix_remap = spix_remap.reshape_as(spix)
    # num, letter = pd.factorize(spix.cpu().numpy().ravel())
    # spix_remap = th.from_numpy(num).to(device).reshape((B,H,W)).type(th.int)
    # # print(spix_remap.max(),spix_remap.min(),len(th.unique(spix_remap)))
    # return spix_remap
    return spix_remap


def viz_spix(spix,N):
    B,H,W = spix.shape
    viridis = mpl.colormaps['viridis'].resampled(N)
    spix = spix / N
    cols = viridis(spix.cpu().numpy())
    cols = rearrange(to_th(cols),'b h w f -> b f h w')
    return cols

def get_warp(img,flow,warp_type="grid"):
    if warp_type == "stnls":
        return get_stnls_warp(img,flow)
    elif warp_type == "stnls_fwd":
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

def shift_labels(spix,means,flow,root):
    B,H,W = spix.shape
    grid = futils.index_grid(H,W,dtype=spix.dtype,
                             device=spix.device,normalize=True)
    grid = st_spix.sp_pool_from_spix(grid,spix)

    gscatter,gcnts = st_spix.scatter.run(grid,flow,swap_c=True)
    valid = th.where(th.logical_and(0.95<gcnts,gcnts<1.05))
    invalid = th.where(th.logical_or(gcnts<0.95,1.05<gcnts))
    for i in range(2):
        gscatter[:,i][invalid] = -100.

    # -- normalize --
    # for i in range(2):
    #     gscatter[:,i][valid] = gscatter[:,i][valid]/gcnts[valid]

    # -- save --
    gscatter_fmt = gscatter.abs().mean(1,keepdim=True)
    gscatter_fmt = gscatter_fmt / gscatter_fmt.max()
    tv_utils.save_image(gscatter_fmt,root / "gscatter.png")

    # -- all pairwise differences [stupid] --
    spix_grid = th.arange(means.shape[0]).to(spix.device)+1
    locs = th.stack([means[:,-2]/(W-1),means[:,-1]/(H-1)],-1)
    print(locs[:,0].max(),locs[:,0].min())
    print(locs[:,1].max(),locs[:,1].min())
    print(gscatter[0,:,:3,:3])
    print(locs[470])
    print(locs[478])
    # print("locs.shape: ",locs.shape)
    # print(locs)
    gscatter = rearrange(gscatter,'b f h w -> (b h w) f')
    dists = th.cdist(gscatter,locs)
    # print("Max Num of Transfer Spix.:",len(th.unique(dists.argmin(1))))
    shifted_spix = spix_grid[dists.argmin(1)]
    print("Spix Min/Max: ",spix.min().item(),spix.max().item())
    print("Shifted Spix Min/Max: ",shifted_spix.min().item(),shifted_spix.max().item())
    shifted_spix = rearrange(shifted_spix,'(b h w) -> b h w',h=H,w=W)
    shifted_spix[invalid] = -1

    # viridis = mpl.colormaps['viridis'].resampled(8)
    # -- viz both spix --
    K = spix.max()+1
    shifted_spix_fmt  = shifted_spix/K
    tv_utils.save_image(shifted_spix_fmt,root / "shifted_spix.png")
    spix_fmt  = spix/K
    tv_utils.save_image(spix_fmt,root / "spix.png")

    print(spix[0,:3,:3]+1)
    print(shifted_spix[0,:3,:3])

    # -- viz both spix --
    K = spix.max()+1
    spix_fmt = viz_spix(spix+1,K)
    tv_utils.save_image(spix_fmt,root / "spix_c.png")
    shifted_spix_fmt  = viz_spix(shifted_spix,K)
    tv_utils.save_image(shifted_spix_fmt,root / "shifted_spix_c.png")


    return shifted_spix

def run_exp(cfg):

    # -- config --
    root = Path("./output/flow_remap")
    if not root.exists(): root.mkdir(parents=True)

    # -- load images --
    vid = st_spix.data.davis_example(isize=None)[0,:2]
    # vid = vid + (25./255.)*th.randn_like(vid)
    tv_utils.save_image(vid,"./output/flow_remap/vid.png")
    img0,img1 = img_for_bass(vid[0]),img_for_bass(vid[1])
    B,H,W,F = img0.shape

    # -- run img0 bass --
    npix_in_side = 40
    # i_std,alpha,beta = 0.018,20.,100.
    i_std,alpha,beta = 0.1,0.001,100.
    spix,means,cov,counts,ids = st_spix_cuda.bass_forward(img0,npix_in_side,
                                                          i_std,alpha,beta)
    print(spix[0,:3,:3])
    spix = remap_spix(spix,ids,device="cuda")
    print(spix[0,:3,:3])
    print("Num Superpixels: ",len(th.unique(spix)))
    marked = mark_boundaries(img0.cpu().numpy(),spix.cpu().numpy())
    marked = to_th(swap_c(marked))
    print("marked.shape: ",marked.shape)
    tv_utils.save_image(marked,root / "marked.png")

    # -- format for remaining code --
    img0,img1 = swap_c(img0/255.),swap_c(img1/255.)

    # -- run flow --
    flows = flow_pkg.run(vid[None,:],sigma=0.0,ftype="cv2")
    flow = flows.bflow[0,1][None,:]
    # flow[...] = 0.
    futils.viz_flow_quiver(root/"flow",flow)

    # -- run scatter/warp --
    # print("img0.shape,flow.shape: ",img0.shape,flow.shape)
    scatter,cnts = st_spix.scatter.run(img0,flow,swap_c=True)
    print("[scatter] PSNR: ",compute_psnrs(scatter+1e-8,img0))
    tv_utils.save_image(scatter,root/"scatter.png")
    # cnts = cnts / cnts.max()
    # tv_utils.save_image(cnts,root/"counts.png")

    # -- spix pool --
    flow_sp = st_spix.sp_pool_from_spix(flow,spix)
    spix1 = shift_labels(spix,means,flow_sp,root) # propogate labels
    warp_sp = get_warp(img0,flow_sp.flip(1))
    print("[sp] PSNR: ",compute_psnrs(warp_sp,img1))
    scatter,cnts = st_spix.scatter.run(img0.contiguous(),
                                       flow_sp.contiguous())
    _scatter = scatter.clone()
    scatter = th.clamp(scatter,0,1.)
    for i in range(3):
        scatter[:,i][th.where(cnts>=1.01)] = i==0
        scatter[:,i][th.where(cnts<=0.99)] = i==2
    tv_utils.save_image(scatter,root/"scatter_sp_anno.png")
    tv_utils.save_image(_scatter,root/"scatter_sp.png")
    tv_utils.save_image(th.clamp(cnts,0,1.),root/"counts_sp.png")
    print("[sp-scatter] PSNR: ",compute_psnrs(_scatter+1e-8,img0))
    tv_utils.save_image(warp_sp,root/"warped_sp.png")
    futils.viz_flow_quiver(root/"flow_sp",flow_sp)

    # -- stnls --
    ws,ps = 9,1
    flow_stnls = run_stnls(img1,img0,flow,ws,ps,full_ws=False)
    # flow_stnls = st_spix.sp_pool_from_spix(flow_stnls,spix)
    flow_stnls_sp = st_spix.sp_pool_from_spix(flow_stnls,spix)
    scatter,cnts = st_spix.scatter.run(img0.contiguous(),
                                       flow_stnls_sp.contiguous().flip(1))
    tv_utils.save_image(scatter,root/"scatter_stnls.png")
    print("[stnls-scatter] PSNR: ",compute_psnrs(scatter+1e-8,img0))
    warp_stnls = get_warp(img0,flow_stnls)
    print("[stnls] PSNR: ",compute_psnrs(warp_stnls,img1))
    tv_utils.save_image(warp_stnls,root/"warped_stnls.png")
    futils.viz_flow_quiver(root/"flow_stnls",flow_stnls.flip(1))

    # -- warp image --
    alpha = 5.
    flow_sgd = flow.clone().requires_grad_(True)
    # flow_sgd = flow_sp.clone().requires_grad_(True)
    # flow_sgd = flow_stnls.clone().requires_grad_(True)
    nsteps = 200
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
