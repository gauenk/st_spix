
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
from st_spix import flow_utils as futils
import torchvision.io as iio
from einops import rearrange,repeat
from skimage.segmentation import mark_boundaries
import torchvision.utils as tv_utils
import torch.nn.functional as th_f

# import seaborn as sns
# import matplotlib.pyplot as plt
from dev_basics.utils.metrics import compute_psnrs

try:
    import stnls
except:
    pass

from dev_basics import flow as flow_pkg
from dev_basics.utils.timer import ExpTimer,TimeIt

from easydict import EasyDict as edict

# -- colorwheel --
# import matplotlib.pyplot as plt
# import matplotlib as mpl
# from matplotlib.colors import LinearSegmentedColormap, ListedColormap

def stream_bass(vid,sp_size=80,alpha=0.001,beta=10.,nrefine=30,fflow=None):

    # -- config --
    npix_in_side = sp_size
    niters,inner_niters = 1,1
    # i_std,alpha,beta = 0.018,20.,100.
    i_std = 0.1

    # -- load images --
    vid = th.clip(255.*vid,0.,255.).type(th.uint8)
    T,F,H,W = vid.shape
    B,F,H,W = vid.shape

    # -- get flow --
    if fflow is None:
        # -- run flow [cv2] --
        # flows = flow_pkg.run(vid[None,:]/255.,sigma=0.0,ftype="cv2")
        # fflow = flows.fflow[0]
        # print(fflow.shape)

        # -- run flow [raft] --
        from st_spix.flow_utils import run_raft
        fflow,bflow = run_raft(vid)
        # print("fflow.shape: ",fflow.shape)
        # exit()

    # -- bass --
    img0 = img4bass(vid[None,0])
    bass_fwd = st_spix_cuda.bass_forward
    spix0,means,cov,counts,ids = bass_fwd(img0,npix_in_side,i_std,alpha,beta)
    ids = ids.unsqueeze(1).expand(-1, means.size(-1)).long()[None,:]

    # -- iterations --
    spix = [spix0]
    for ix in range(vid.shape[0]-1):

        # -- unpack --
        img_curr = img4bass(vid[None,ix+1])
        flow_curr = fflow[[ix]].contiguous()

        # -- run --
        outs = prop_seg(img_curr.clone(),spix[-1].clone(),flow_curr.clone(),
                        means.clone(),cov.clone(),counts.clone(),ids.clone(),
                        niters,inner_niters,npix_in_side,i_std,
                        alpha,beta,nrefine)
        spix_t,child,means,cov,counts = outs
        spix.append(spix_t)
    spix = th.stack(spix)[:,0]
    return spix,fflow

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


def shift_labels_v0(spix,means,flow):

    # -- unpack --
    B,H,W = spix.shape
    flow = flow.clone()

    # -- scatter --
    grid = futils.index_grid(H,W,dtype=spix.dtype,
                             device=spix.device,normalize=True)
    # grid = rearrange(grid,'b h w f -> b f h w')
    # print("grid.shape,flow.shape: ",grid.shape,flow.shape)
    grid = st_spix.sp_pool_from_spix(grid,spix)
    # print("grid.shape,flow.shape: ",grid.shape,flow.shape,means.shape)
    # print(".")
    # th.cuda.synchronize()
    gscatter,gcnts = st_spix.scatter.run(grid,flow,swap_c=True)
    # th.cuda.synchronize()
    # print("..")
    # print("[0] gscatter.shape: ",gscatter.shape)
    # print("[0] gcnts.shape: ",gcnts.shape)
    # exit()

    # -- invalidate --
    eps = 1e-13
    invalid = th.where(th.logical_or(gcnts<1-eps,1+eps<gcnts))
    for i in range(2):
        gscatter[:,i][invalid] = -100.
    # print("gscatter.shape: ",gscatter.shape)

    # -- all pairwise differences --
    locs = th.stack([means[:,-2]/(W-1),means[:,-1]/(H-1)],-1)
    # print("[2] gscatter.shape,locs.shape: ",gscatter.shape,locs.shape)
    # gscatter = rearrange(gscatter,'b h w f -> (b h w) f')
    gscatter = rearrange(gscatter,'b f h w -> (b h w) f')
    dists = th.cdist(gscatter,locs)
    # print("dists.shape: ",dists.shape)
    # print("means.shape: ",means.shape)

    # -- gather --
    spix_grid = th.arange(means.shape[0]).to(spix.device) #  I think "0" in invalid?
    # print("spix_grid.shape: ",spix_grid.shape)
    shifted_spix = spix_grid[dists.argmin(1)].int()
    # print(shifted_spix.min() >= -1)
    # print("shaped: ",means.shape,flow.shape,shifted_spix.shape)
    shifted_spix = rearrange(shifted_spix,'(b h w) -> b h w',h=H,w=W)
    shifted_spix[invalid] = -1

    return shifted_spix,gcnts

def prop_seg(img,spix,flow,means,cov,counts,ids,
             niters,inner_niters,npix_in_side,i_std,
             alpha,beta,refine_iters):

    # -- unpack --
    K = spix.max().item()+1
    max_SP = K-1
    eps = 1e-13

    # -- get superpixel flows and shift means --
    flow_sp,means_shift = st_spix.pool_flow_and_shift_mean(flow,means.clone(),spix,ids)

    # -- shift & mark overlaps/holes --
    spix_s,missing,invalid = shift_labels(spix,means[0],flow_sp)

    # -- edge case --
    if missing.numel() == 0:
        return spix_s,means,cov,counts

    # -- exec filling --
    niters_refine = refine_iters
    fill_debug,user_xfer = False,False
    fxn = st_spix_prop_cuda.spix_prop_dev
    outs = fxn(img,spix_s,missing,means,cov,counts,npix_in_side,
               i_std,alpha,beta,niters,inner_niters,niters_refine,
               K,max_SP,fill_debug,0,use_xfer)
    boarder,spix_s,db_spix,db_border,db_seg,_means,cov,counts,unique_ids = outs
    assert spix_s.max() <= means.shape[1],"Must be equal or less than."

    return spix_s,means,cov,counts


def refine_flow_sp(vid,spix,flow):
    flow_r = flow.clone()
    print("hi.")
    exit()
    return flow_r


def viz_marked_debug(img,debug,debug_border,missing,root):

    # -- check and create --
    if debug.numel() < 100: return
    if not root.exists():
        root.mkdir(parents=True)

    # -- debug info --
    # print("debug.")
    # print(th.mean(1.*(debug>=0),(-1,-2)))
    # print(th.mean(1.*(debug<0),(-1,-2)))

    # -- info --
    # print("Negs per debug: ",th.sum(debug==-1,dim=(-2,-1)))
    # print("Border: ",th.sum(debug_border,dim=(-2,-1)))
    negs = th.sum(debug==-1,dim=(-2,-1))
    deltas = negs[:-1] - negs[1:]
    # print("deltas: ",deltas)
    # print("negs: ",negs)
    any_neg = (th.sum((deltas == 0) * (negs[:-1] > 0))>0).item()
    # print("Any neg?",any_neg)
    missing = repeat(missing,'1 h w -> r h w',r=len(debug))
    # print("missing.shape: ",missing.shape)

    # -- [testing] negative inds --
    # debug[...,10:20,10:20] = -1

    # -- find negative locs --
    # print("debug.shape: ",debug.shape)
    # print("debug_border.shape: ",debug_border.shape)
    # print(debug_border[0])
    # print(img.shape)
    # print("img.min(),img.max(): ",img.min(),img.max())


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

    # print(" view that band area that keeps dissappearing.  ")
    # print(debug[0][165:184,212:220+4])
    # print(debug[1][165:184,212:220+4])

    #
    # -- Red Filter @ "-1" spix --
    #

    # -- alpha channel --
    alpha = th.ones_like(marks[:,0])
    alpha[th.where(debug==-1)] = 0.10
    alpha = alpha[:,None]
    marks = th.cat([marks,alpha],1)

    # -- fill --
    red = th.zeros_like(marks)
    red[:,2] = debug==-1
    red[:,3] = 1 - alpha[:,0]#1.*(debug==-1)
    view = alpha * marks + (1-alpha) * red

    # print(view.shape,debug_border.shape)
    # view[:,2][th.where(debug_border)] = 0.
    # view[:,1][th.where(debug_border)] = 1.
    # view[:,0][th.where(debug_border)] = 0.


    # view[:,2][th.where(debug_border)] = 0.
    # view[:,0][th.where(th.logical_and(debug_border,missing))] = 1.

    # view[:,0][th.where(th.logical_and(debug_border,missing))] = 1.

    # view[:,0][th.where(missing)] = 1.
    # view[:,1][th.where(missing)] = 0.
    # view[:,2][th.where(missing)] = 0.
    # print(missing.shape)
    # exit()

    # marks[:,[3]]*marks + red[:,[3]]*red
    # print(red.min(),red.max())
    # print(marks.min(),marks.max())
    # print(view.min(),view.max())
    # print("debug_border.shape: ",debug_border.shape)

    tv_utils.save_image(view,root / "marked_view.png")
    tv_utils.save_image(1.*debug_border[:,None],root / "border.png")
    th.save(debug,str(root/"spix.pth"))

    # exit(any_neg)

def mark_spix_vid(vid,spix):
    marked = []
    for ix,spix_t in enumerate(spix):
        img = rearrange(vid[:,ix],'b f h w -> b h w f')
        marked_t = mark_boundaries(img.cpu().numpy(),spix_t.cpu().numpy())
        marked_t = to_th(swap_c(marked_t))
        marked.append(marked_t)
    marked = th.cat(marked)
    return marked
