
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

def shift_labels(spix,means,flow):

    # -- unpack --
    B,H,W = spix.shape
    flow = flow.clone()

    # -- scatter --
    grid = futils.index_grid(H,W,dtype=spix.dtype,
                             device=spix.device,normalize=True)
    grid = st_spix.sp_pool_from_spix(grid,spix)
    gscatter,gcnts = st_spix.scatter.run(grid,flow,swap_c=True)

    # -- invalidate --
    eps = 1e-13
    invalid = th.where(th.logical_or(gcnts<1-eps,1+eps<gcnts))
    for i in range(2):
        gscatter[:,i][invalid] = -100.

    # -- all pairwise differences [stupid] --
    locs = th.stack([means[:,-2]/(W-1),means[:,-1]/(H-1)],-1)
    gscatter = rearrange(gscatter,'b f h w -> (b h w) f')
    dists = th.cdist(gscatter,locs)

    # -- gather --
    spix_grid = th.arange(means.shape[0]).to(spix.device)+1
    shifted_spix = spix_grid[dists.argmin(1)].int()
    print("shaped: ",means.shape,flow.shape,shifted_spix.shape)
    shifted_spix = rearrange(shifted_spix,'(b h w) -> b h w',h=H,w=W)
    shifted_spix[invalid] = -1


    return shifted_spix,gcnts

def prop_seg(img,spix,flow,means,cov,counts,ids,
             niters,inner_niters,npix_in_side,i_std,alpha,beta):

    # -- unpack --
    K = spix.max().item()+1
    max_SP = K-1
    eps = 1e-13

    # -- rigid shift --
    # flow_sp = st_spix.sp_pool_from_spix(flow,spix)
    flow_sp,means = st_spix.pool_flow_and_shift_mean(flow,means,spix,ids,K)
    spix_s,cnts = shift_labels(spix.clone(),means[0],flow_sp) # propogate labels

    # -- mark overlapping and holes --
    invalid = th.logical_or(cnts>1+eps,cnts<1-eps)
    missing = th.where(invalid.ravel())[0][None,:].type(th.int)
    spix_s[th.where(invalid)] = -1
    # print(spix_s.shape,missing.shape,K,max_SP,img.shape)
    # th.cuda.synchronize()
    # exit()

    # -- exec refine --
    fill_debug = True
    fxn = st_spix_prop_cuda.spix_prop_dev
    border,spix_s,debug = fxn(img,spix_s,missing,means,cov,counts,
                              npix_in_side,i_std,alpha,beta,niters,
                              inner_niters,K,max_SP,fill_debug)

    return spix_s,debug,means

def viz_marked_debug(img,debug,root):

    # -- debug info --
    print("debug.")
    print(th.mean(1.*(debug>=0),(-1,-2)))
    print(th.mean(1.*(debug<0),(-1,-2)))

    # -- marked regions from debug iterations --
    # _img1 = rearrange(img1,'b f h w -> b h w f')
    _img1 = img
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

def mark_spix_vid(vid,spix):
    marked = []
    for ix,spix_t in enumerate(spix):
        img = rearrange(vid[:,ix],'b f h w -> b h w f')
        marked_t = mark_boundaries(img.cpu().numpy(),spix_t.cpu().numpy())
        marked_t = to_th(swap_c(marked_t))
        marked.append(marked_t)
    marked = th.cat(marked)
    return marked

def img4bass(img):
    img = rearrange(img,'... f h w -> ... h w f')
    img = img.contiguous()
    if img.ndim == 3: img = img[None,:]
    return img

def run_exp(cfg):

    # -- config --
    root = Path("./output/prop_seg")
    if not root.exists(): root.mkdir(parents=True)
    timer = ExpTimer()

    # -- config --
    npix_in_side = 80
    niters,inner_niters = 1,25
    # i_std,alpha,beta = 0.018,20.,100.
    i_std,alpha,beta = 0.1,0.001,100.


    # -- load images --
    vid = st_spix.data.davis_example(isize=None,nframes=10)[:1,:10,:,:480,:480]
    # vid = vid + (25./255.)*th.randn_like(vid)
    vid = th.clip(255.*vid,0.,255.).type(th.uint8)
    B,T,F,H,W = vid.shape
    tv_utils.save_image(vid[0]/255.,root/"vid.png")

    # -- bass --
    img0 = img4bass(vid[:,0])
    bass_fwd = st_spix_cuda.bass_forward
    spix0,means,cov,counts,ids = bass_fwd(img0,npix_in_side,i_std,alpha,beta)
    # print(len(th.unique(spix0)))
    # print(spix0.min(),spix0.max())
    # exit()
    timer.sync_start("bass")
    spix0,means,cov,counts,ids = bass_fwd(img0,npix_in_side,i_std,alpha,beta)
    timer.sync_stop("bass")

    timer.sync_start("remap")
    # spix0 = remap_spix(spix0,ids,device="cuda")
    timer.sync_stop("remap")

    # -- run flow --
    timer.sync_start("flow")
    flows = flow_pkg.run(vid/255.,sigma=0.0,ftype="cv2")
    flow = flows.fflow[0,0][None,:]
    # flow = flows.bflow[0,1][None,:]
    timer.sync_stop("flow")
    ids = ids.unsqueeze(1).expand(-1, means.size(-1)).long()[None,:]

    # -- iterations --
    spix_st = [spix0]
    spix_s = [spix0]
    for ix in range(5):

        # -- unpack --
        img_curr = img4bass(vid[:,ix+1])
        flow_curr = flows.fflow[0,ix][None,:]

        # -- run --
        timer.sync_start("st_iter_%d"%ix)
        spix_curr_st,debug,means = prop_seg(img_curr,spix_st[-1],flow_curr,
                                            means,cov,counts,ids,niters,
                                            inner_niters,npix_in_side,i_std,alpha,beta)
        timer.sync_stop("st_iter_%d"%ix)

        # -- run --
        timer.sync_start("s_iter_%d"%ix)
        spix_curr_s,_,_,_,_ = bass_fwd(img_curr,npix_in_side,i_std,alpha,beta)
        # spix0,means,cov,counts,ids = st_spix_original_cuda.bass_forward(
        # spix_curr_st,debug = prop_seg(img_curr,spix_st[-1],flow_curr,means,cov,counts,
        #                               niters,inner_niters,npix_in_side,i_std,alpha,beta)
        timer.sync_stop("s_iter_%d"%ix)

        # -- debug --
        # viz_marked_debug(img_curr,debug,root)

        # -- append --
        spix_st.append(spix_curr_st)
        spix_s.append(spix_curr_s)

        # -- stop condition --
        if spix_curr_st.min().item() < 0: break

    # -- read timer --
    print(timer)

    # -- view superpixels --
    marked = mark_spix_vid(vid,spix_st)
    tv_utils.save_image(marked,root / "marked_spacetime.png")
    marked = mark_spix_vid(vid,spix_s)
    tv_utils.save_image(marked,root / "marked_space.png")


def main():

    cfg = edict()
    cfg.name = "a"
    run_exp(cfg)

if __name__ == "__main__":
    main()
