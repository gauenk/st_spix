
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
    child = [None]
    for ix in range(vid.shape[0]-1):

        # -- unpack --
        img_curr = img4bass(vid[None,ix+1])
        flow_curr = fflow[[ix]].contiguous()

        # -- run --
        outs = prop_seg(img_curr.clone(),spix[-1].clone(),flow_curr.clone(),
                        means.clone(),cov.clone(),counts.clone(),ids.clone(),
                        niters,inner_niters,npix_in_side,i_std,
                        alpha,beta,nrefine)
        spix_t,child_t,shift_st,dbs,dbp,missing,means = outs
        spix.append(spix_t)
        child.append(child_t)
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
    # print(spix.min(),spix.max())
    K = spix.max().item()+1
    max_SP = K-1
    eps = 1e-13
    # print("K: ",K)

    # -- get superpixel flows and shift means --
    flow_sp,means_shift = st_spix.pool_flow_and_shift_mean(flow,means.clone(),spix,ids)

    # -- shift & mark overlaps/holes --
    spix_s,missing,invalid = shift_labels(spix,means[0],flow_sp)

    # -- shift & mark overlaps/holes --
    # spix_s,cnts = shift_labels_v0(spix.clone(),means[0],flow_sp) # propogate labels
    # means = _means
    # invalid = th.logical_or(cnts>1+eps,cnts<1-eps)
    # missing = th.where(invalid.ravel())[0][None,:].type(th.int)
    # spix_s[th.where(invalid)] = -1

    # -- edge case --
    spix_s_0 = spix_s.clone()
    if missing.numel() == 0:
        # print("think of what to do.")
        empty = th.tensor([])
        return spix_s,spix_s_0,empty,empty,invalid,means

    # -- debug --
    # print("[prop] img.min(), img.max(): ",img.min(), img.max())
    # print("[info0] spix: ",spix_s.min().item(),spix_s.max().item())

    # -- exec filling --
    niters_refine = refine_iters
    fill_debug,user_xfer = False,False
    fxn = st_spix_prop_cuda.spix_prop_dev
    outs = fxn(img,spix_s,missing,means,cov,counts,npix_in_side,
               i_std,alpha,beta,niters,inner_niters,niters_refine,
               K,max_SP,fill_debug,0,use_xfer)
    boarder,spix_s,db_spix,db_border,db_seg,_means,cov,counts,unique_ids,child = outs
    # border,spix_s,db_spix,db_border = outs
    assert spix_s.max() <= means.shape[1],"Must be equal or less than."
    # print("[info1] spix: ",spix_s.min().item(),spix_s.max().item())

    # -- exec refine --
    # print("[prop] img.min(), img.max(): ",img.min(), img.max())

    return spix_s,child,spix_s_0,db_spix,db_border,invalid,means


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

def run_exp(cfg):

    # -- config --
    root = Path("./output/prop_seg")
    if not root.exists(): root.mkdir(parents=True)
    timer = ExpTimer()

    # -- config --
    npix_in_side = 80
    niters,inner_niters = 1,2
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
    spix_nr = [spix0]
    spix_st = [spix0]
    spix_s = [spix0]
    for ix in range(1):

        # -- unpack --
        #print("vid[:,ix+1].min(),vid[:,ix+1].max():",vid[:,ix+1].min(),vid[:,ix+1].max())
        img_curr = img4bass(vid[:,ix+1])
        flow_curr = flows.fflow[0,ix][None,:]
        # print("img_curr.min(),img_curr.max(): ",img_curr.min(),img_curr.max())

        # -- run --
        refine_iters = 8
        timer.sync_start("st_iter_%d"%ix)
        outs = prop_seg(img_curr.clone(),spix_st[-1].clone(),flow_curr.clone(),
                        means.clone(),cov.clone(),counts.clone(),ids.clone(),
                        niters,inner_niters,npix_in_side,i_std,
                        alpha,beta,refine_iters)
        spix_curr_st,shift_st,dbs,dbp,missing,_means = outs
        timer.sync_stop("st_iter_%d"%ix)
        viz_marked_debug(img_curr,dbs,dbp,missing,root/f"st_{ix}")
        tv_utils.save_image(mark_spix(img_curr[0],shift_st[0]),root/"init_shift_st.png")


        # -- run --
        refine_iters = 0
        timer.sync_start("nr_iter_%d"%ix)
        outs = prop_seg(img_curr,spix_st[-1],flow_curr,
                        means,cov,counts,ids,niters,
                        inner_niters,npix_in_side,i_std,
                        alpha,beta,refine_iters)
        spix_curr_nr,shift_nr,_dbs,_dbp,_missing,_means = outs
        timer.sync_stop("nr_iter_%d"%ix)
        viz_marked_debug(img_curr,_dbs,_dbp,_missing,root/f"nr_{ix}")
        tv_utils.save_image(mark_spix(img_curr[0],shift_nr[0]),root/"init_shift_nr.png")

        # -- run --
        timer.sync_start("s_iter_%d"%ix)
        spix_curr_s,_,_,_,_ = bass_fwd(img_curr,npix_in_side,i_std,alpha,beta)
        # spix0,means,cov,counts,ids = st_spix_original_cuda.bass_forward(
        # spix_curr_st,debug = prop_seg(img_curr,spix_st[-1],flow_curr,means,cov,counts,
        #                               niters,inner_niters,npix_in_side,i_std,alpha,beta)
        timer.sync_stop("s_iter_%d"%ix)

        # -- debug --
        # viz_marked_debug(img_curr,debug,root)

        # -- stop condition --
        if spix_curr_st.min().item() < 0:
            print("breaking early.")
            break
        spix_curr_st = spix_curr_st - spix_curr_st.min()

        # -- append --
        spix_st.append(spix_curr_st)
        spix_nr.append(spix_curr_nr)
        spix_s.append(spix_curr_s)


    # -- read timer --
    print(timer)

    # -- view superpixels --
    print("root: ",root)
    th.save(spix_st[0],root / "spix0_st.pth")
    th.save(spix_st[1],root / "spix1_st.pth")
    marked = mark_spix_vid(vid,spix_st)
    tv_utils.save_image(marked,root / "marked_spacetime.png")
    th.save(spix_nr[0],root / "spix0_nr.pth")
    th.save(spix_nr[1],root / "spix1_nr.pth")
    marked = mark_spix_vid(vid,spix_nr)
    tv_utils.save_image(marked,root / "marked_norefine.png")
    spix_noshift = [spix_st[0],]*len(spix_st)
    marked = mark_spix_vid(vid,spix_noshift)
    tv_utils.save_image(marked,root / "marked_noshift.png")
