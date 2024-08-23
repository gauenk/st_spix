
import os
import torch as th
import numpy as np
import pandas as pd
from pathlib import Path
import st_spix
from st_spix import flow_utils
import st_spix_cuda
import st_spix_prop_cuda
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
from dev_basics.utils.timer import ExpTimer,TimeIt

from easydict import EasyDict as edict


# -- colorwheel --
import matplotlib.pyplot as plt
import matplotlib as mpl
from matplotlib.colors import LinearSegmentedColormap, ListedColormap

def img_for_bass(img,device="cuda"):
    img= (th.clip(img,0.,1.)*255.).type(th.uint8)
    img = rearrange(img,'... f h w -> ... h w f').to(device)
    if img.ndim == 3: img = img[None,:]
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
    debug = False
    B,H,W = spix.shape
    flow = flow.clone()
    # print("flow.shape: ",flow.shape)
    # flow = -flow.flip(1).contiguous()
    # flow = flow.flip(1).contiguous()
    # flow[...] = 0.
    # flow = 2*flow
    # flow[:,0] = -flow[:,0]
    # flow[:,1] = -flow[:,1]
    # print("flow.shape: ",flow.shape)
    grid = futils.index_grid(H,W,dtype=spix.dtype,
                             device=spix.device,normalize=True)
    grid = st_spix.sp_pool_from_spix(grid,spix)

    gscatter,gcnts = st_spix.scatter.run(grid,flow,swap_c=True)
    # valid = th.where(th.logical_and(0.95<gcnts,gcnts<1.05))
    eps = 1e-13
    invalid = th.where(th.logical_or(gcnts<1-eps,1+eps<gcnts))
    for i in range(2):
        gscatter[:,i][invalid] = -100.

    # -- normalize --
    # for i in range(2):
    #     gscatter[:,i][valid] = gscatter[:,i][valid]/gcnts[valid]

    # -- save --
    # gscatter_fmt = gscatter.abs().mean(1,keepdim=True)
    # gscatter_fmt = gscatter_fmt / gscatter_fmt.max()
    # tv_utils.save_image(gscatter_fmt,root / "gscatter.png")

    # -- all pairwise differences [stupid] --
    spix_grid = th.arange(means.shape[0]).to(spix.device)+1
    locs = th.stack([means[:,-2]/(W-1),means[:,-1]/(H-1)],-1)
    # print(locs[:,0].max(),locs[:,0].min())
    # print(locs[:,1].max(),locs[:,1].min())
    # print(gscatter[0,:,:3,:3])
    # print(locs[470])
    # print(locs[478])
    # print("locs.shape: ",locs.shape)
    # print(locs)
    gscatter = rearrange(gscatter,'b f h w -> (b h w) f')
    dists = th.cdist(gscatter,locs)
    # print("Max Num of Transfer Spix.:",len(th.unique(dists.argmin(1))))
    shifted_spix = spix_grid[dists.argmin(1)]
    # print("Spix Min/Max: ",spix.min().item(),spix.max().item())
    # print("Shifted Spix Min/Max: ",shifted_spix.min().item(),shifted_spix.max().item())
    shifted_spix = rearrange(shifted_spix,'(b h w) -> b h w',h=H,w=W)
    shifted_spix[invalid] = -1

    # viridis = mpl.colormaps['viridis'].resampled(8)

    # -- viz both spix --
    if debug:
        K = spix.max()+1
        shifted_spix_fmt  = shifted_spix/K
        tv_utils.save_image(shifted_spix_fmt,root / "shifted_spix.png")
        spix_fmt  = spix/K
        tv_utils.save_image(spix_fmt,root / "spix.png")

    # print(spix[0,:3,:3]+1)
    # print(shifted_spix[0,:3,:3])

    # -- viz both spix --
    if debug:
        K = spix.max()+1
        spix_fmt = viz_spix(spix+1,K)
        tv_utils.save_image(spix_fmt,root / "spix_c.png")
        shifted_spix_fmt  = viz_spix(shifted_spix,K)
        tv_utils.save_image(shifted_spix_fmt,root / "shifted_spix_c.png")


    return shifted_spix,gcnts

def run_exp(cfg):

    print("PID: ",os.getpid())

    # -- config --
    root = Path("./output/dev_prop_seg")
    if not root.exists(): root.mkdir(parents=True)
    timer = ExpTimer()

    # -- load images --
    vid = st_spix.data.davis_example(isize=None)[0,:10,:,:480,:480]
    # print("vid [min,max]: ",vid.min(),vid.max())
    # vid = th.clip(vid,0.,1.)
    # vid = vid + (50./255.)*th.randn_like(vid)
    tv_utils.save_image(vid,root/"vid.png")
    img0,img1 = img_for_bass(vid[0]),img_for_bass(vid[1])
    B,H,W,F = img0.shape

    # -- run img0 bass --
    npix_in_side = 30
    # i_std,alpha,beta = 0.018,20.,100.
    i_std,alpha,beta = 0.1,0.001,100.
    spix,means,cov,counts,ids = st_spix_original_cuda.bass_forward(img0,npix_in_side,
                                                                   i_std,alpha,beta)
    timer.sync_start("bass")
    spix,means,cov,counts,ids = st_spix_original_cuda.bass_forward(img0,npix_in_side,
                                                                   i_std,alpha,beta)
    timer.sync_stop("bass")
    # spix = remap_spix(spix,ids,device="cuda")


    K = len(th.unique(spix))
    max_SP = th.max(spix).item()
    print(spix[0,:3,:3])
    spix = remap_spix(spix,ids,device="cuda")
    print(spix[0,:3,:3])
    print("Num Superpixels: ",len(th.unique(spix)))
    marked = mark_boundaries(img0.cpu().numpy(),spix.cpu().numpy())
    marked = to_th(swap_c(marked))
    print("marked.shape: ",marked.shape)
    tv_utils.save_image(marked,root / "marked.png")
    _spix0 = spix.clone().float()
    args = th.logical_and(_spix0<70.,_spix0>50)
    _spix0[th.where(th.logical_not(args))] = 0.
    _spix0[th.where(args)] = 1.
    tv_utils.save_image(_spix0[None,:],root / "spix0.png")

    # -- format for remaining code --
    img0,img1 = swap_c(img0/255.),swap_c(img1/255.)

    # -- run flow --
    flows = flow_pkg.run(vid[None,:],sigma=0.0,ftype="cv2")
    flow = flows.fflow[0,0][None,:]
    # flow = -flows.bflow[0,1][None,:]

    # flow[:,0] = -flow[:,0]
    # flow[:,1] = -flow[:,1]
    # flow = 2*flow
    # flow[...] = 0.
    futils.viz_flow_quiver(root/"flow",flow)

    # -- run scatter/warp --
    # print("img0.shape,flow.shape: ",img0.shape,flow.shape)
    # scatter,cnts = st_spix.scatter.run(img0,-flow.flip(1),swap_c=True)
    # flow[:,1] = -flow[:,1]
    # flow[...] = 10.
    # flow = flow.flip(1)
    # _flow = flow.clone()
    # # _flow = -_flow.flip(1)
    # _flow[:,0] = -_flow[:,0]
    # # _flow[:,1] = -_flow[:,1]
    scatter,cnts = st_spix.scatter.run(img0,flow,swap_c=True)
    # scatter,cnts = st_spix.scatter.run(img0,flow,swap_c=True)
    print("[scatter] PSNR: ",compute_psnrs(scatter+1e-8,img0))
    tv_utils.save_image(scatter,root/"scatter.png")
    # cnts = cnts / cnts.max()
    # tv_utils.save_image(cnts,root/"counts.png")

    # -- spix pool --

    timer.sync_start("flow pool.")
    flow_sp = st_spix.sp_pool_from_spix(flow,spix)
    timer.sync_stop("flow pool.")

    timer.sync_start("flow pool 2.")
    flow_sp = st_spix.sp_pool_from_spix(flow,spix)
    timer.sync_stop("flow pool 2.")

    timer.sync_start("shift.")
    spix1,cnts = shift_labels(spix.clone(),means,flow_sp,root) # propogate labels
    timer.sync_stop("shift.")
    timer.sync_start("shift 2")
    spix1,cnts = shift_labels(spix.clone(),means,flow_sp,root) # propogate labels
    timer.sync_stop("shift 2")
    warp_sp = get_warp(img0,flow_sp.flip(1))
    print("[sp] PSNR: ",compute_psnrs(warp_sp,img1))
    # scatter,cnts = st_spix.scatter.run(img0.contiguous(),
    #                                    flow_sp.contiguous())
    #                                    # -flow_sp.flip(1).contiguous())
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

    # -- run prop code --
    # print("img1.shape: ",img1.shape)
    # exit()
    # _img1 = rearrange(img1,'... f h w -> ... h w f')
    _img1 = img_for_bass(img1)
    eps = 1e-13
    logic = th.logical_or(cnts.ravel()>1+eps,cnts.ravel()<1-eps)
    missing = th.where(logic)[0][None,:]
    missing = missing.type(th.int)
    print("missing: ",missing)
    print("missing.shape: ",missing.shape)
    fxn = st_spix_prop_cuda.spix_prop_dev
    _img1 = _img1.contiguous()
    # spix1 = remap_spix(spix1,ids,device="cuda")
    eps = 1e-13
    invalid = th.where(th.logical_or(cnts<1-eps,1+eps<cnts))
    spix1[invalid] = -1
    # print(spix.min(),spix.max())
    # exit()
    K = spix.max().item()+1
    max_SP = K-1
    spix1 = spix1.contiguous().type(th.int)
    print("shapes: ",_img1.shape,_img1.dtype)
    niters = 1
    inner_niters = 5
    niters_refine = 40
    fill_debug = True
    _ = fxn(_img1,spix1,missing,means,cov,counts,
            npix_in_side,i_std,alpha,beta,niters,
            inner_niters,niters_refine,K,max_SP,fill_debug)

    timer.sync_start("fwd")
    print("_img1.shape: ",_img1.shape)
    border,spix1,debug = fxn(_img1,spix1,missing,means,cov,counts,
                             npix_in_side,i_std,alpha,beta,niters,
                             inner_niters,niters_refine,K,max_SP,fill_debug)
    timer.sync_stop("fwd")
    print(timer)
    # spix1 = spix.clone()

    print(spix[invalid])
    print(debug.shape)
    print(th.mean(1.*(spix1>=0)))
    print(th.mean(1.*(spix1<0)))
    print("no edge")
    print(th.mean(1.*(spix1[...,1:-1,1:-1]>=0)))
    print(th.mean(1.*(spix1[...,1:-1,1:-1]<0)))
    print("debug")
    print(th.mean(1.*(debug>=0),(-1,-2)))
    print(th.mean(1.*(debug<0),(-1,-2)))
    print("debug [no edge]")
    print(th.mean(1.*(debug[...,1:-1,1:-1]>=0),(-1,-2)))
    print(th.mean(1.*(debug[...,1:-1,1:-1]<0),(-1,-2)))
    debug_noedge = debug[...,1:-1,1:-1]
    print("*"*20)
    print(border.shape)
    print("border: ",th.sum(border))
    print(cnts.shape)
    th.cuda.synchronize()
    # return

    # -- viz on scatter --
    eps = 1e-13
    scatter = th.clamp(scatter,0,1.)
    for i in range(3):
        scatter[:,i][th.where(cnts>1+eps)] = i==0
        scatter[:,i][th.where(cnts<1-eps)] = i==2
        scatter[:,i][th.where(border)] = i==1
    tv_utils.save_image(scatter,root/"scatter_sp_anno_b.png")

    # -- view invalid boarder on img1  --
    _img1 = img1.clone()
    for i in range(3):
        _img1[:,i][th.where(cnts>1+eps)] = i==0
        _img1[:,i][th.where(cnts<1-eps)] = i==2
        _img1[:,i][th.where(border)] = i==1
    tv_utils.save_image(_img1,root/"img1_sp_anno_b.png")

    # -- alpha-mix [scatter+img1] --
    print("scatter.shape: ",scatter.shape)
    alpha = 0.5*th.ones_like(scatter[:,[0]])
    alpha_mix = th.cat([scatter,alpha],-3)
    tv_utils.save_image(alpha_mix,root/"alpha_mix.png")

    # -- delta spix from debug iterations --
    if debug.numel() > 0:
        _debug = th.abs(debug[1:] - debug[[0]])
        _debug = _debug / _debug.max()
        tv_utils.save_image(_debug[:,None],root / "delta_debug.png")

        # -- spix from debug --
        print("max,min: ",debug_noedge.max(),debug_noedge.min())
        _debug = debug_noedge / debug_noedge.max()
        _debug = repeat(_debug,'b h w -> b c h w',c=3).clone()
        _neg = th.where(_debug[:,0]<0)
        # print(_neg)
        _debug[:,2:] = 0.
        _debug[:,1][_neg] = 1.
        # print(_debug.shape)
        tv_utils.save_image(_debug,root / "debug.png")


        # -- marked regions from debug iterations --
        _img1 = rearrange(img1,'b f h w -> b h w f')
        marks = []
        for _debug in debug:
            marked_ = mark_boundaries(_img1.cpu().numpy(),_debug[None,:].cpu().numpy())
            marked_ = to_th(swap_c(marked_))
            # if len(marks) > 0:
            #     marked_[0] = th.abs(marked_[0] - marks[0])
            #     marked_[0] = marked_[0]/marked_[0].abs().max()
            marks.append(marked_[0])
        marks = th.stack(marks)
        tv_utils.save_image(marks,root / "marked_debug.png")

    # -- spix values --
    # _spix0 = spix/spix.max()
    # tv_utils.save_image(_spix0[None,:],root / "spix0.png")
    # _spix1 = spix1/spix1.max()
    _spix1 = spix1.clone().float()
    args = th.logical_and(_spix1<70.,_spix1>50)
    _spix1[th.where(th.logical_not(args))] = 0.
    _spix1[th.where(args)] = 1.
    # _spix1 = spix1.clone()
    # args = th.where(th.logical_and(_spix1<70.,_spix1>50))
    # _spix1[args] = 0.
    # _spix1[args] = 1.
    tv_utils.save_image(_spix1[None,:],root / "spix1.png")
    tv_utils.save_image(spix1[None,:]/spix1.max(),root / "spix1_m.png")

    # -- spixmarkboundaries --
    print("img1.shape: ",img1.shape)
    img1 = rearrange(img1,'b f h w -> b h w f')
    marked = mark_boundaries(img1.cpu().numpy(),spix1.cpu().numpy())
    marked = to_th(swap_c(marked))
    print("marked.shape: ",marked.shape)
    tv_utils.save_image(marked,root / "marked1.png")

    # -- spixmarkboundaries --
    print("img1.shape: ",img1.shape)
    # img1 = rearrange(img1,'b f h w -> b h w f')
    marked = mark_boundaries(img1.cpu().numpy(),spix.cpu().numpy())
    marked = to_th(swap_c(marked))
    print("marked.shape: ",marked.shape)
    tv_utils.save_image(marked,root / "marked1_0.png")

    # -- spix --
    cc = 210
    print(spix1.shape)
    _spix1 = spix1[None,:,cc:-cc,cc:-cc]
    print(_spix1[0,0,:20,:20])
    _spix1 = _spix1/spix1.max()
    tv_utils.save_image(_spix1,root / "spix1_cc.png")
    print(img1.shape)
    _img1 = rearrange(img1/img1.max(),'b h w f -> b f h w')
    tv_utils.save_image(_img1[:,:,cc:-cc,cc:-cc],root / "img1_cc.png")


# spix_prop_dev_cuda(const torch::Tensor imgs,
#                    const torch::Tensor in_spix,
#                    const torch::Tensor in_missing,
#                    const torch::Tensor in_means,
#                    const torch::Tensor in_cov,
#                    const torch::Tensor in_counts,
#                    int nPixels_in_square_side, float i_std,
#                    float alpha, float beta, int niters,
#                    int in_K, int max_SP){



def main():

    cfg = edict()
    cfg.name = "a"
    run_exp(cfg)

if __name__ == "__main__":
    main()
