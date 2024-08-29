

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

from skimage import io, color

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

    # -- all pairwise differences --
    locs = th.stack([means[:,-2]/(W-1),means[:,-1]/(H-1)],-1)
    gscatter = rearrange(gscatter,'b f h w -> (b h w) f')
    dists = th.cdist(gscatter,locs)

    # -- gather --
    spix_grid = th.arange(means.shape[0]).to(spix.device) #  I think "0" in invalid?
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

    # -- rigid shift --
    # flow_sp = st_spix.sp_pool_from_spix(flow,spix)
    flow_sp,_means = st_spix.pool_flow_and_shift_mean(flow,means.clone(),spix,ids)
    spix_s,cnts = shift_labels(spix.clone(),means[0],flow_sp) # propogate labels
    # print("spix_s.min(),spix_s.max(): ",spix_s.min(),spix_s.max())
    means = _means

    # -- mark overlapping and holes --
    invalid = th.logical_or(cnts>1+eps,cnts<1-eps)
    missing = th.where(invalid.ravel())[0][None,:].type(th.int)
    spix_s[th.where(invalid)] = -1
    spix_s_0 = spix_s.clone()

    # print("Comparing negatives: ",missing.shape,th.sum(spix_s==-1))
    # print(spix_s.shape,missing.shape,K,max_SP,img.shape)
    # th.cuda.synchronize()
    # exit()

    # -- exec filling --
    niters_refine = refine_iters
    fill_debug = True
    use_xfer = True
    fxn = st_spix_prop_cuda.spix_prop_dev
    # print("[prop] img.min(), img.max(): ",img.min(), img.max())
    # print("[info0] spix: ",spix_s.min().item(),spix_s.max().item())
    outs = fxn(img,spix_s,missing,means,cov,counts,npix_in_side,
               i_std,alpha,beta,niters,inner_niters,niters_refine,
               K,max_SP,fill_debug,0,use_xfer)
    boarder,spix_s,db_spix,db_border,db_seg,_means,cov,counts,unique_ids = outs
    # border,spix_s,db_spix,db_border = outs
    assert spix_s.max() <= means.shape[1],"Must be equal or less than."
    # print("[info1] spix: ",spix_s.min().item(),spix_s.max().item())

    # -- exec refine --
    # print("[prop] img.min(), img.max(): ",img.min(), img.max())

    return spix_s,spix_s_0,db_spix,db_border,db_seg,invalid,means

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
        img = rearrange(vid[ix],'f h w -> h w f')
        img = img.cpu().numpy()
        img = (255*img).astype(np.uint8)
        img_lab = color.rgb2lab(img)
        img_lab = (img_lab - img_lab.min())
        img_lab = img_lab / img_lab.max()
        marked_t = mark_boundaries(img_lab,spix_t.cpu().numpy())
        marked_t = to_th(swap_c(marked_t))
        marked.append(marked_t)
    marked = th.stack(marked)
    return marked

def run_exp(cfg):

    # -- config --
    root = Path("./output/anim_refine")
    if not root.exists(): root.mkdir(parents=True)
    timer = ExpTimer()

    # -- config --
    npix_in_side = 80
    niters,inner_niters = 1,1
    # i_std,alpha,beta = 0.018,20.,100.
    i_std,alpha,beta = 0.1,0.001,10.

    # -- load images --
    vid = st_spix.data.davis_example(isize=None,nframes=10)[:1,:10,:,:480,:480]
    # vid = vid + (25./255.)*th.randn_like(vid)
    vid[...,:10,:10] = 1.
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


    # -- run --
    ix = 0
    img_curr = img4bass(vid[:,ix+1])
    flow_curr = flows.fflow[0,ix][None,:]
    refines = []
    for refine_iters in range(9,10):
        outs = prop_seg(img_curr.clone(),spix_st[-1].clone(),flow_curr.clone(),
                        means.clone(),cov.clone(),counts.clone(),ids.clone(),
                        niters,inner_niters,npix_in_side,i_std,
                        alpha,beta,6*refine_iters)
        spix_curr_st,shift_st,dbs,dbp,db_seg,missing,_means = outs
        refines.append(spix_curr_st[0])
        # if refine_iters > 0:
        #     db_segs.append(db_seg)
        # viz_marked_debug(img_curr,dbs,dbp,missing,root/f"st_{ix}")
        # tv_utils.save_image(mark_spix(img_curr[0],shift_st[0]),root/"init_shift_st.png")
    means = _means

    # -- view superpixels --
    print("root: ",root)
    # 165:184,212:220+4]
    print(db_seg.shape)
    # db_seg = db_seg[:,167:184,65:75]
    # db_seg = db_seg[:,175:200,65:90]

    # db_seg = db_seg[:,165:210,55:100] # use this crop
    db_seg = db_seg[:,180:195,65:85] # use this crop
    spix = spix_curr_st
    print("spix.shape: ",spix.shape)
    spix = spix[0,180:195,65:85]

    # -- view the segmentation pixel values at top-left and bottom-right --
    # print(spix)
    # # print(means.shape)
    # TL = spix[0,0]
    # BR = spix[-1,-1]
    # print(means[0,int(TL)])
    # print(means[0,int(BR)])
    # print(db_seg.shape)
    # args = th.where(db_seg[53,:,:,-1]>0)
    # print(args)

    # # print(db_seg[50,:,:,5:8])
    # print(db_seg[53,:,:,8:11])
    # # print(db_seg[50,:,:,11+5:11+8])
    # print(db_seg[53,:,:,11+8:11+11])
    # exit()


    # for i in range(50):
    #     print(db_seg[i])
    # # print(db_seg.shape)

    # names = db_seg[...,[4,9,14,19]][:,None]
    # probs = probs - th.min(probs)
    # probs = probs / probs.max()
    # print(names)

    probs = db_seg[-1,...,[3,3+11,3+2*11,3+3*11]]
    probs[th.where(probs==0.)] = -100000000
    probs = th.exp((probs - th.max(probs))/100.)
    probs = probs / (probs.sum(-1,keepdim=True)+1e-10)
    print(probs.shape)
    tv_utils.save_image(th.stack([probs[...,0],probs[...,1]])[:,None],
                        root / "probs_NS.png",nrow=5)

    boundary = db_seg[:,:,:,-1][:,None]
    print(boundary.min(),boundary.max())
    tv_utils.save_image(boundary,root / "debug_seg.png",nrow=5)

    # tv_utils.save_image(1.*(probs[...,1]>probs[...,0]),root / "debug_SgtN.png",nrow=5)
    # tv_utils.save_image(1.*(probs[...,1]<probs[...,0]),root / "debug_NgtS.png",nrow=5)
    # check_geq = th.cat([1.*(probs[...,1]>probs[...,0]),
    #                     1.*(probs[...,2]>probs[...,0]),
    #                     db_seg[:,:,:,-1][:,None],
    #                     # th.zeros_like(probs[...,1]),
    #                     1.*(probs[...,1]<probs[...,0]),],1)
    # print(check_geq.shape)
    # tv_utils.save_image(check_geq,root / "debug_INEQ.png",nrow=5)
    # tv_utils.save_image(probs[...,0],root / "debug_N.png",nrow=5)
    # tv_utils.save_image(probs[...,1],root / "debug_S.png",nrow=5)
    # tv_utils.save_image(probs[...,2],root / "debug_W.png",nrow=5)
    # tv_utils.save_image(probs[...,3],root / "debug_E.png",nrow=5)

    s,sp = 0,11
    inds = [s,s+sp,s+2*sp,s+3*sp]
    sim = db_seg[...,inds][:,None].clone()
    s,sp = 1,11
    inds = [s,s+sp,s+2*sp,s+3*sp]
    sim += db_seg[...,inds][:,None]
    s,sp = 2,11
    inds = [s,s+sp,s+2*sp,s+3*sp]
    sim += db_seg[...,inds][:,None]
    sim[th.where(sim==0.)] = -100000000
    sim = th.exp((sim - th.max(sim))/100.)
    sim = sim / (sim.sum(-1,keepdim=True)+1e-10)
    tv_utils.save_image(sim[...,0],root / "sim_N.png",nrow=5)
    tv_utils.save_image(sim[...,1],root / "sim_S.png",nrow=5)

    sp = 11
    pixsim = db_seg[-1,...,[0,1*sp,2*sp,3*sp]][None,:]
    pixsim[th.where(pixsim==0.)] = -100000000
    pixsim = th.exp((pixsim - th.max(pixsim))/100.)
    pixsim = pixsim / (pixsim.sum(-1,keepdim=True)+1e-10)
    tv_utils.save_image(th.stack([pixsim[...,0],pixsim[...,1]]),
                        root / "pixsim_NS.png",nrow=5)


    s,sp = 1,11
    spacesim = db_seg[-1,...,[s,s+sp,s+2*sp,s+3*sp]][None,]
    spacesim[th.where(spacesim==0.)] = -100000000
    spacesim = th.exp((spacesim - th.max(spacesim))/100.)
    spacesim = spacesim / (spacesim.sum(-1,keepdim=True)+1e-10)
    tv_utils.save_image(th.stack([spacesim[...,0],spacesim[...,1]]),
                        root / "spacesim_NS.png",nrow=5)

    s,sp = 2,11
    potts = db_seg[-1,...,[s,s+sp,s+2*sp,s+3*sp]][None,]
    potts[th.where(potts==0.)] = -100000000
    potts = th.exp((potts - th.max(potts))/100.)
    potts = potts / (potts.sum(-1,keepdim=True)+1e-10)
    tv_utils.save_image(th.stack([potts[...,0],potts[...,1]]),
                        root / "potts_NS.png",nrow=5)


    # s,sp = 2,11
    # inds = [s,s+sp,s+2*sp,s+3*sp]
    # potts = -db_seg[...,inds][:,None]
    # # print(potts[50,0,...,0])
    # # print(potts[50,0,...,1])
    # potts[th.where(potts==0.)] = -100000000
    # potts = th.exp((potts - th.max(potts))/100.)
    # potts = potts / (potts.sum(-1,keepdim=True)+1e-10)
    # # print("potts.shape: ",potts.shape)
    # # print(potts[50,0,...,0])
    # # print(potts[50,0,...,1])
    # tv_utils.save_image(potts[...,0],root / "potts_N.png",nrow=5)
    # tv_utils.save_image(potts[...,1],root / "potts_S.png",nrow=5)
    # exit()

    # tv_utils.save_image(1.*(names[...,0]>0),root / "names_N.png",nrow=5)
    # tv_utils.save_image(1.*(names[...,1]>0),root / "names_S.png",nrow=5)
    # tv_utils.save_image(1.*(names[...,2]>0),root / "names_W.png",nrow=5)
    # tv_utils.save_image(1.*(names[...,3]>0),root / "names_E.png",nrow=5)

    print(" LAB values ")
    print(db_seg[...,8].min(),db_seg[...,8].max())
    print(db_seg[...,9].min(),db_seg[...,9].max())
    print(db_seg[...,10].min(),db_seg[...,10].max())
    # exit()
    imgs = [db_seg[50,...,[8+i*sp,9+i*sp,10+i*sp]].permute(2,0,1) for i in range(2)]
    imgN = imgs[0]
    # imgN[0] = imgN[0]*-100
    # imgN[1] = imgN[1]*-100
    imgS = imgs[1]
    print("imgN.min(), imgN.max(): ",imgN.min(),imgN.max())
    # imgN[0,:,:][th.where(imgN[0]==0)] = -1
    # imgS[0,:,:][th.where(imgS[0]==0)] = -1
    print("imgN.shape: ",imgN.shape)
    # imgN = color.lab2rgb(100*imgN.cpu().numpy().transpose(1,2,0)).transpose(2,0,1)
    # imgS = color.lab2rgb(100*imgS.cpu().numpy().transpose(1,2,0)).transpose(2,0,1)
    # imgN = color.lab2rgb(100*imgN.cpu().numpy().transpose(1,2,0)).transpose(2,0,1)
    # imgS = color.lab2rgb(100*imgS.cpu().numpy().transpose(1,2,0)).transpose(2,0,1)
    print("imgN.shape: ",imgN.shape)
    print("imgN.min(), imgN.max(): ",imgN.min(),imgN.max())
    # imgN = (imgN +1.)/2.
    # imgN = imgN / imgN.max()
    # imgS = imgS / imgS.max()

    # thc = lambda x: th.from_numpy(np.stack(x)).flip(-3
    thc = lambda x: th.stack(x).flip(-3)
    print("shaped: ",thc([imgN,imgS]).shape)
    tv_utils.save_image(thc([imgN,imgS]),root / "means_NS.png",nrow=5)
    # tv_utils.save_image(imgS,root / "means_S.png",nrow=5)

    imgs = [db_seg[50,...,[5+i*sp,6+i*sp,7+i*sp]].permute(2,0,1) for i in range(2)]
    imgN = imgs[0]
    imgS = imgs[1]
    imgN[0,:,:][th.where(imgN[0]==0)] = -1
    imgS[0,:,:][th.where(imgS[0]==0)] = -1
    # imgN = color.lab2rgb(100*imgN.cpu().numpy().transpose(1,2,0)).transpose(2,0,1)
    # imgS = color.lab2rgb(100*imgS.cpu().numpy().transpose(1,2,0)).transpose(2,0,1)
    # imgN = (imgN +1.)/2.
    # imgN = imgN / imgN.max()
    # imgS = (imgS +1.)/2.
    # imgN = imgN / imgN.max()
    # imgS = imgS / imgS.max()
    # print(imgN.shape,imgS.shape)
    tv_utils.save_image(thc([imgN,imgS]),root / "pix_NS.png",nrow=5)
    # tv_utils.save_image(imgS,root / "pix_S.png",nrow=5)

    # exit()
    # tv_utils.save_image(db_seg,root / "dbseg.png",nrow=5)
    vid = th.stack([vid[0,1],]*len(refines))/255.
    marked = mark_spix_vid(vid,refines)
    print(marked.shape)
    marked[:,0,180:195,65:85] = 1
    print(marked.shape)
    tv_utils.save_image(marked,root / "marked.png",nrow=5)


def main():

    print("PID: ",os.getpid())
    cfg = edict()
    cfg.name = "a"
    run_exp(cfg)

if __name__ == "__main__":
    main()
