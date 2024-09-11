
import os
import torch as th
import numpy as np
import pandas as pd
from pathlib import Path
import st_spix
from st_spix import flow_utils
import st_spix_cuda
import st_spix_prop_cuda
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
from scipy.stats import chi2

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
    missing = th.where(th.logical_or(gcnts<1-eps,1+eps<gcnts))
    # invalid = th.logical_or(cnts>1+eps,cnts<1-eps)
    #missing = th.where(invalid.ravel())[0][None,:].type(th.int)
    for i in range(2):
        gscatter[:,i][missing] = -100.

    # -- all pairwise differences --
    locs = th.stack([means[:,-2]/(W-1),means[:,-1]/(H-1)],-1)
    gscatter = rearrange(gscatter,'b f h w -> (b h w) f')
    dists = th.cdist(gscatter,locs)

    # -- gather --
    spix_grid = th.arange(means.shape[0]).to(spix.device)#+1
    shifted_spix = spix_grid[dists.argmin(1)].int()
    # print("shaped: ",means.shape,flow.shape,shifted_spix.shape)
    shifted_spix = rearrange(shifted_spix,'(b h w) -> b h w',h=H,w=W)
    shifted_spix[missing] = -1

    # -- get missing --
    invalid = th.logical_or(gcnts<1-eps,gcnts>1+eps)
    missing = th.where(invalid.ravel())[0][None,:].type(th.int)

    return shifted_spix,missing,gcnts

def prop_seg(img,spix,flow,means,cov,counts,ids,
             niters,inner_niters,npix_in_side,i_std,alpha,beta):

    # -- unpack --
    # print(spix.min(),spix.max())
    K = spix.max().item()+1
    max_SP = spix.max().item()+1
    eps = 1e-13
    print("K: ",K)

    # -- rigid shift --
    # flow_sp = st_spix.sp_pool_from_spix(flow,spix)
    flow_sp,_means = st_spix.pool_flow_and_shift_mean(flow,means.clone(),spix,ids,K)
    info = shift_labels(spix.clone(),means[0],flow_sp) # propogate labels
    spix_s,missing,cnts = info
    means = _means

    # -- mark overlapping and holes --
    # invalid = th.logical_or(cnts>1+eps,cnts<1-eps)
    # missing = th.where(invalid.ravel())[0][None,:].type(th.int)
    # spix_s[th.where(invalid)] = -1
    # print(spix_s.shape,missing.shape,K,max_SP,img.shape)
    # th.cuda.synchronize()


    # -- info --
    # print(spix)
    # print(spix[-20:,:20])
    # print(spix[:20,-20:])
    # print("-"*20)
    # print("-"*20)
    # print(spix_s)
    # print(spix.max(),spix_s.max())
    # print(spix_s[-20:,:20])
    # print(spix_s[:20,-20:])
    # exit()

    # -- exec filling --
    niters_refine = 40
    fill_debug = True
    fxn = st_spix_prop_cuda.spix_prop_dev
    border,spix_s,debug = fxn(img,spix_s,missing,means,cov,counts,
                              npix_in_side,i_std,alpha,beta,niters,
                              inner_niters,niters_refine,K,max_SP,fill_debug)
    # assert spix_s.max() <= means.shape[1],"Must be equal or less than."
    # print("[info] spix: ",spix_s.min().item(),spix_s.max().item())

    # -- exec refine --

    return spix_s,debug,means

def viz_marked_debug(img,debug,root):

    # -- debug info --
    # print("debug.")
    # print(th.mean(1.*(debug>=0),(-1,-2)))
    # print(th.mean(1.*(debug<0),(-1,-2)))

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

def get_params(img,spix,npix_in_side,i_std,alpha,beta):
    K = spix.max().item()+1
    fxn = st_spix_prop_cuda.get_params_forward
    means,cov,counts,ids = fxn(img,spix,npix_in_side,i_std,alpha,beta,K)
    return means,cov,counts,ids

# std::tuple<torch::Tensor,torch::Tensor,torch::Tensor,torch::Tensor>
# get_params_forward(const torch::Tensor imgs,
#                    const torch::Tensor in_spix,
#                    int nPixels_in_square_side, float i_std,
#                    float alpha, float beta, int in_K){



def viz_ellipsoids(root,name,means0,cov0,spix,img,H,spix_id=-1):


    spix_viz = spix[0].cpu().detach() /spix.max().item()
    spix_viz = spix_viz.unsqueeze(2).repeat(1,1,3)
    alpha = th.ones_like(spix_viz[:,:,[0]])
    # print("spix_viz.shape: ",spix_viz.shape)
    spix_viz = th.cat([spix_viz,0.5*alpha],-1)
    img_viz = img[0].cpu().detach()
    # print("img_viz.shape: ",img_viz.shape)
    alpha = (150.*alpha).type(th.uint8)
    img_viz = th.cat([img_viz,alpha],-1)
    # print("img_viz.shape: ",img_viz.shape)
    # print("spix_viz.shape: ",spix_viz.shape)
    # exit()
    means0 = means0[0].cpu().detach()
    ftrs0,locs0 = means0[...,:3],means0[...,-2:].clone()
    # locs0[:,1] = H-locs0[:,1]
    cov0 = cov0[0].cpu().detach().reshape(-1,2,2)

    # cov0[0,0,0] = 0.01
    # cov0[0,0,1] = 0.00
    # cov0[0,1,0] = 0.00
    # cov0[0,1,1] = 0.02

    # means1 = means1[0].cpu().detach()
    # ftrs1,locs1 = means1[...,:3],means1[...,-2:].clone()
    # locs1[:,1] = H-locs1[:,1]
    # cov1 = cov1[0].cpu().detach().reshape(-1,2,2)
    # print(cov0.shape)

    # -- info --
    # print(cov0[0])
    # print(cov0)
    # print(locs0)

    # print(cov0[-50])

    # Compute the eigenvalues and eigenvectors of the covariance matrix
    eigVals0,eigVecs0 = th.linalg.eigh(cov0)
    # eigVals1,eigVecs1 = th.linalg.eigh(cov1)
    # print(eigVals0.shape,eigVals1.shape)
    # # eigenvalues, eigenvectors = np.linalg.eigh(covariance)
    # print(eigVals0)

    # -- info --
    # tmp = cov0[-7]
    # print(tmp.shape)
    # print(tmp)
    # print(eigVecs0[-7])
    # print(eigVals0[-7])
    # print(tmp @ eigVecs0[-7,[0]].T,eigVals0[-7,0]*eigVecs0[-7,0])
    # print(tmp @ eigVecs0[-7,[1]].T,eigVals0[-7,1]*eigVecs0[-7,1])

    # Compute the angle of the ellipse's major axis in degrees
    # print("eigVecs0.shape: ",eigVecs0.shape)
    angle0 = th.atan2(eigVecs0[:, 0, -1],eigVecs0[:, 1, -1])
    # angle1 = th.atan2(eigVecs1[:, 0, -1],eigVecs1[:, 1, -1])
    # print(angle0[0])
    # print(angle0.shape)
    # print(angle1.shape)
    # exit()
    # print("angle0.shape: ",angle0.shape)
    # print(angle0)

    # Calculate the radius of the ellipse for the 90% confidence interval
    confidence_level = 0.90/2.
    chi2_val = chi2.ppf(confidence_level, 2)
    # print(chi2_val)
    # chi2_val = 10000.
    radii0 = th.sqrt(chi2_val * 1./eigVals0)
    # print(radii0.shape)
    # print(radii0)
    # radii1 = th.sqrt(chi2_val * 1./eigVals1)
    # print("radii0.shape: ",radii0.shape)

    # Define the ellipse points
    theta = th.linspace(0, 2 * th.pi, 100)
    # print("theta.shape: ",theta.shape)
    ellipse0 = th.stack([radii0[:,:1]*th.cos(theta),radii0[:,1:]*th.sin(theta)],-1)
    # ellipse1 = th.stack([radii1[:,:1]*th.cos(theta),radii1[:,1:]*th.sin(theta)],-1)

    # print("ellipse0.shape: ",ellipse0.shape)
    # print("ellipse1.shape: ",ellipse1.shape)

    rot0 = th.stack([th.stack([th.cos(angle0),-th.sin(angle0)],1),
                     th.stack([th.sin(angle0),th.cos(angle0)],1)],1)
    # print(rot0.shape)
    # rot1 = th.stack([th.stack([th.cos(angle1),-th.sin(angle1)],1),
    #                  th.stack([th.sin(angle1),th.cos(angle1)],1)],1)
    # print(rot1.shape)
    # print(rot0.shape,rot1.shape)
    # exit()
    ellipse0 = th.einsum('bij,bkj->bik', rot0, ellipse0)
    # ellipse1 = th.einsum('bij,bkj->bik', rot1, ellipse1)
    if ellipse0.shape[-1] == 2:
        ellipse0 = rearrange(ellipse0,'b n t -> b t n')
        # ellipse1 = rearrange(ellipse1,'b n t -> b t n')
    # print(ellipse0.shape)
    # print(means0.shape)
    # exit()
    valid_args = th.where(th.logical_and(locs0[:, 0]>=0,locs0[:, 1]>=0))

    # Plot the ellipses
    plt.figure()
    # print(spix_viz.shape,img.shape)
    plt.imshow(spix_viz)
    plt.imshow(img_viz)
    for i in range(means0.shape[0]):
        if (spix_id >= 0) and (spix_id != i): continue
        # print((ellipse0[i, 0] + locs0[i, 0],ellipse0[i, 1] + locs0[i, 1]))
        if locs0[i,0].item() < 0: continue
        if locs0[i,1].item() < 0: continue
        plt.plot((ellipse0[i, 0] + locs0[i, 0]),
                 (ellipse0[i, 1] + locs0[i, 1]),
                 label=f'90% Confidence Interval {i+1}')
        # if i >= 10: break
    plt.scatter(locs0[:,0][valid_args],locs0[:,1][valid_args])
    plt.xlabel('X-axis')
    plt.ylabel('Y-axis')
    plt.title('90% Confidence Interval of 2D Gaussians')
    # plt.legend()
    plt.grid(True)
    plt.axis('equal')
    plt.savefig(root/("%s.png"%name))

def inspect_rotation(m0,m1,c0,c1,H):
    print(m0.shape,m1.shape,c0.shape,c1.shape,H)
    print(m0,m1)
    exit()

def run_exp(cfg):

    # -- config --
    root = Path("./output/ellipses")
    if not root.exists(): root.mkdir(parents=True)
    timer = ExpTimer()

    # -- config --
    npix_in_side = 80
    # npix_in_side = 20
    niters,inner_niters = 1,25
    # i_std,alpha,beta = 0.018,20.,100.
    i_std,alpha,beta = 0.1,0.01,1000.


    # -- load images --
    vid = st_spix.data.davis_example(isize=None,nframes=10)[:1,:10,:,:480,:480]
    # vid = st_spix.data.davis_example(isize=None,nframes=10)[:1,:10,:,:64,:64]
    # vid = vid + (25./255.)*th.randn_like(vid)
    vid = th.clip(255.*vid,0.,255.).type(th.uint8)
    B,T,F,H,W = vid.shape
    tv_utils.save_image(vid[0]/255.,root/"vid.png")

    # -- bass --
    img0 = img4bass(vid[:,0])
    img1 = img4bass(vid[:,1])
    bass_fwd = st_spix_cuda.bass_forward
    # spix0,means,cov,counts,ids = bass_fwd(img0,npix_in_side,i_std,alpha,beta)
    # print(means[0,:,-2:])
    # exit()
    # print(len(th.unique(spix0)))
    # print(spix0.min(),spix0.max())
    # exit()
    timer.sync_start("bass")
    spix0,means,cov,counts,ids = bass_fwd(img0,npix_in_side,i_std,alpha,beta)
    timer.sync_stop("bass")
    params0 = get_params(img0,spix0,npix_in_side,i_std,alpha,beta)
    means0 = params0[0].clone()
    cov0 = params0[1].clone()
    # print(ids)
    # exit()

    # print(spix0.min(),spix0.max())
    # print(spix0)
    # print(spix0[0,-20:,:20])
    # exit()

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
    ix = 0

    # -- unpack --
    img_curr = img4bass(vid[:,ix+1])
    flow_curr = flows.fflow[0,ix][None,:]

    # -- run --
    timer.sync_start("st_iter_%d"%ix)
    spix_curr_st,debug,means = prop_seg(img_curr,spix_st[-1],flow_curr,
                                        means,cov,counts,ids,niters,
                                        inner_niters,npix_in_side,i_std,alpha,beta)
    timer.sync_stop("st_iter_%d"%ix)
    spix_st.append(spix_curr_st)

    print(".")
    # -- run --
    print(img_curr.shape)
    timer.sync_start("s_iter_%d"%ix)
    spix_curr_s = bass_fwd(img_curr,npix_in_side,i_std,alpha,beta)[0]
    # spix_curr_st,debug = prop_seg(img_curr,spix_st[-1],flow_curr,means,cov,counts,
    #                               niters,inner_niters,npix_in_side,i_std,alpha,beta)
    timer.sync_stop("s_iter_%d"%ix)
    spix_s.append(spix_curr_s)
    params_s = get_params(img_curr,spix_curr_s,npix_in_side,i_std,alpha,beta)

    # -- debug --
    # viz_marked_debug(img_curr,debug,root)

    # -- stop condition --
    spix_curr_st = spix_curr_st - spix_curr_st.min()

    # -- unpack cov --
    spix1 = spix_st[-1]
    params0 = get_params(img0,spix0,npix_in_side,i_std,alpha,beta)
    means0,cov0 = params0[0],params0[1]
    params1 = get_params(img1,spix1,npix_in_side,i_std,alpha,beta)
    means1,cov1 = params1[0],params1[1]


    # print(spix0.min(),spix0.max())
    # print(spix0)
    # print(spix0[0,-20:,:20])
    # print("-"*20)
    # print("-"*20)
    # print(spix1.min(),spix1.max())
    # print(spix1)
    # print(spix1[0,-20:,:20])
    # exit()

    # -- compute rotations --

    # params_st = get_params(img_curr,spix_curr_st,npix_in_side,i_std,alpha,beta)
    viz_ellipsoids(root,"ellipse0",means0,cov0,spix0,img0,H)
    viz_ellipsoids(root,"ellipse1",means1,cov1,spix1,img1,H)
    subroot = root / "anim"
    if not subroot.exists(): subroot.mkdir()
    for ix in range(10):
        viz_ellipsoids(subroot,"ellipse0_%d"%ix,means0,cov0,spix0,img0,H,ix)
        viz_ellipsoids(subroot,"ellipse1_%d"%ix,means1,cov1,spix1,img1,H,ix)

    # spix1 = spix_st[-1]
    # means1 = params_st[0]
    # cov1 = params_st[1]
    # viz_ellipsoids(root,means0,cov0,spix0,img0,H)
    # viz_ellipsoids(root,means0,cov0,means1,cov1,spix0,img0,H)

    # exit()

    # -- read timer --
    print(timer)

    # -- view superpixels --
    marked = mark_spix_vid(vid,spix_st)
    tv_utils.save_image(marked,root / "marked_spacetime.png")
    marked = mark_spix_vid(vid,spix_s)
    tv_utils.save_image(marked,root / "marked_space.png")

    inspect_rotation(means0[0,0],means1[0,0],cov0[0,0],cov1[0,0],H)


def main():

    print("PID: ",os.getpid())
    cfg = edict()
    cfg.name = "a"
    run_exp(cfg)

if __name__ == "__main__":
    main()
