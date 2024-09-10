
import torch as th
import numpy as np
from einops import rearrange,repeat
from pathlib import Path
from functools import reduce


# -- masked tensors --
from torch.masked import masked_tensor, as_masked_tensor
import warnings
warnings.filterwarnings(action='ignore', category=UserWarning)


from st_spix.spix_utils import mark_spix_vid,img4bass
import torchvision.utils as tv_utils

import st_spix
from st_spix import flow_utils as futils
from st_spix.prop_seg import stream_bass

from torchvision.transforms.functional import resize

import st_spix_cuda
from st_spix import scatter
from st_spix import deform
from st_spix.sp_pooling import pooling,SuperpixelPooling

import stnls
from dev_basics import flow as flow_pkg

import matplotlib.cm as cm
from matplotlib import colormaps
from matplotlib import patches, pyplot as plt
# import matplotlib.pyplot as plt

def run_stnls(nvid,acc_flows,ws):
    wt,ps,s0,s1,full_ws=1,1,1,1,False
    k = 1
    search_p = stnls.search.PairedSearch(ws,ps,k,
                                         nheads=1,dist_type="l2",
                                         stride0=s0,stride1=s1,
                                         self_action=None,use_adj=False,
                                         full_ws=full_ws,itype="float")
    _,flows = search_p.paired_vids(nvid,nvid,acc_flows,wt,skip_self=True)
    return flows

def viz_sample(pixels,locations,masks,spix_id,root):

    # -- sample --
    F = pixels.shape[-1]
    pix = pixels[:,spix_id]
    locs = locations[:,spix_id]
    mask = masks[:,spix_id].bool()

    # -- get extremes --
    # xmin,xmax,ymin,ymax = get_locs_extremes(locs,mask)
    # print(xmin,xmax,ymin,ymax)

    # # -- normalize --
    # locs[:,:,0][mask] = (locs[:,:,0][mask] - xmin)/(xmax-xmin)
    # locs[:,:,1][mask] = (locs[:,:,1][mask] - ymin)/(ymax-ymin)

    # -- plot --
    fig, axes = plt.subplots(1, 3, layout='constrained', figsize=(10, 4))
    for i in range(3):
        mask_i = mask[i].cpu().numpy()
        locs_i = locs[i].cpu().numpy().T
        axes[i].scatter(locs_i[0][mask_i],locs_i[1][mask_i])
        axes[i].set_aspect("equal","datalim")
        axes[i].yaxis.set_inverted(True)
    plt.savefig(root/"scatter.png")

def get_locs_extremes(locs,mask):
    x = locs[:,:,0][mask]
    xmin,xmax = x.min(),x.max()
    y = locs[:,:,1][mask]
    ymin,ymax = y.min(),y.max()
    return xmin,xmax,ymin,ymax

# -=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-
#
#
#          Batched Sinkhorn
#
#
# -=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-


def get_xfer_cost(regions,locations,masks,alpha):

    # -- center each frame's location --
    # print("locations.shape: ",locations.shape)
    masks = masks.bool()
    masks_e2 = masks[...,None].expand((-1,-1,-1,2))
    means = (locations * masks_e2).sum(-2,keepdim=True)
    means = means / (masks_e2.sum(-2,keepdim=True)+1e-10)
    locations = locations - means
    # locations[:,:,mask,0] -= locations[:,:,mask,0].mean(-1)
    # locations[:,:,mask,1] -= locations[:,:,mask,1].mean(-1)

    # -- cost --
    print("any na? ",th.any(th.isnan(locations)).item())
    costC = th.cdist(locations[:-1],locations[1:])
    print("any na? ",th.any(th.isnan(costC)).item())
    costC = alpha*costC + (1-alpha)*th.cdist(regions[:-1],regions[1:])
    print("any na? ",th.any(th.isnan(costC)).item())
    maskM = (masks[:-1,:,:,None]  * masks[1:,:,None])>0
    # maskM = (masks[:-1,:,None]  * masks[1:,:,:,None])>0
    # maskM = maskM.transpose(-2,-1)
    costC = costC * maskM
    # costC = th.cdist(locs[:-1],locs[1:])


    #
    # -- testing --
    #

    # -- checking --
    # mask0 = th.any(maskM,-1)
    # mask1 = th.any(maskM,-2)

    # -- skip the check if its empty --
    # empty0 = th.all(masks[:-1]==0,-1)
    # empty1 = th.all(masks[1:]==0,-1)

    # # -- viz sizes --
    # print("mask0.shape: ",mask0.shape)
    # print("mask1.shape: ",mask1.shape)
    # print("masks.shape: ",masks.shape)
    # print("empty0.shape: ",empty0.shape)
    # print("empty1.shape: ",empty1.shape)

    # -- run check --
    # print((th.sum(1.*masks[-1:] - 1.*mask0,-1)*empty0).sum())
    # print((th.sum(1.*masks[1:] - 1.*mask1,-1)*empty1).sum())
    # exit()

    # # -- compute lims --
    # midx0 = th.max(th.where(masks[0,10]>0)[-1])
    # midx1 = th.max(th.where(masks[1,10]>0)[-1])
    # print(midx0,midx1)
    # midx0 = th.max(th.where(maskM[0,10]>0)[-2])
    # midx1 = th.max(th.where(maskM[0,10]>0)[-1])
    # print(midx0,midx1)
    # exit()

    return costC,maskM

def remap_inds(inds,locations,H,W):
    """
        Convert indices from Intra-Superpixels to Image Space
    """
    B,NS,R,two = locations.shape
    Bm1,NS,K,R = inds.shape
    locations = locations[:,:,None].expand((-1,-1,K,-1,-1))
    inds = inds[...,None].expand((-1,-1,-1,-1,2)) # Bm1,NS,K,R,[2]
    # print(inds.shape)
    # print(locations.shape)
    # inds.shape # Bm1,NS,K(small),R
    img_inds = th.gather(locations[:-1],3,inds)
    # print(img_inds.max(),img_inds.min())
    img_inds[...,0] = img_inds[...,0]*(W-1)
    img_inds[...,1] = img_inds[...,1]*(H-1)
    img_inds = img_inds.long()
    # print(img_inds.shape)
    # print(img_inds.max(),img_inds.min())
    # exit()
    return img_inds

def run_sinkhorn(regions,locations,masks,spix,cost_alpha,root):

    # -- get a single sample for easier dev --
    device = regions.device
    costC,maskM = get_xfer_cost(regions,locations,masks,cost_alpha)
    # ot_scale = 1e3
    # K = th.exp(-ot_scale*costC)*maskM
    # print("masks.shape: ",masks.shape)
    # print("K.shape: ",K.shape)
    Bm1,NS,S,S = costC.shape

    # -- init sinkhorn params --
    # ot_scale = 5e3
    ot_scale = 1
    a,b = 1.*(masks[:-1]>0),1.*(masks[1:]>0)
    a,b = a.reshape(Bm1,NS,S,1),b.reshape(Bm1,NS,S,1)
    K = th.exp(-ot_scale*costC)*maskM

    # -- only one iter --
    # u = a / (K.mean(-1,keepdim=True)+1e-10)
    # v = b / (K.mean(-2,keepdim=True).transpose(-2,-1)+1e-10)
    # pi_est = u * K * v.reshape(Bm1,NS,1,S)

    # -- many iters --
    niters = 0
    v = b.clone()/b.sum(-2,keepdim=True)
    u = a.clone()/a.sum(-2,keepdim=True)
    for iter_i in range(niters):
        u = a / ((K @ v)+1e-10)
        v = b / ((K.transpose(-2,-1) @ u)+1e-10)
    if niters == 0:
        pi_est = K
    else:
        pi_est = u * K * v.reshape(Bm1,NS,1,S)

    # -- flows and weights from transport map --
    spix_idx = 3
    pi_vals,pi_inds = th.topk(pi_est,5,-2)
    beta = 2.
    pi_vals = th.softmax(beta*pi_vals,-2)
    # pi_vals,pi_inds = th.topk(pi_est[:,spix_idx],10,-2)
    viz_pi_sample(pi_vals[:,spix_idx],pi_inds[:,spix_idx],locations[:,spix_idx],
                  masks[:,spix_idx],root,"batching")

    return pi_vals,pi_inds

# -=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-
#
#
#          Single Superpixel Sinkhorn
#
#
# -=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-


def viz_pi_sample(pi_vals,pi_inds,locs,mask,root,prefix=""):

    # -- sample --
    beta = 1
    locs,mask = locs.cpu().numpy(),mask.bool().cpu().numpy()
    pi_vals = th.softmax(beta*pi_vals,-2).cpu().numpy()
    pi_inds = pi_inds.cpu().numpy()
    # print("pi_vals.shape: ",pi_vals.shape)
    # print("pi_inds.shape: ",pi_inds.shape)

    # -- init plot --
    fig, axes = plt.subplots(1, 3, layout='constrained', figsize=(10, 4))

    # -- plot scattering --
    for i in range(3):
        mask_i = mask[i]
        locs_i = locs[i].T
        axes[i].scatter(locs_i[0][mask_i],locs_i[1][mask_i])
        axes[i].set_aspect("equal","datalim")
        axes[i].yaxis.set_inverted(True)

    # -- colors --
    # prop_cycle = plt.rcParams['axes.prop_cycle']
    # colors = prop_cycle.by_key()['color']

    # -- create some quivers --
    print(pi_inds.shape,locs.shape)
    base_size = 2.
    no_red = True
    # for j in range(0,pi_vals.shape[-1],3):
    for j in [50]:
        color = "black"
        # color = "red" if j >= (pi_vals.shape[-1]-20) else "black"
        # if color == "red":
        #     print("red starting: ",j)
        # color = colormaps['prism'](j/(100.-1))
        # print(color)
        for i in range(pi_vals.shape[-2]):
            # quiver_size = base_size * pi[0,j,i]
            # quiver_size = base_size * pi_vals[0,j,i]
            quiver_size = base_size * pi_vals[0,i,j]
            quiver_start = locs[0,pi_inds[0,i,j]]
            # quiver_start = locs[0,j]
            # quiver_end = locs[1,i]
            # quiver_end = locs[1,pi_inds[0,j,i]]
            quiver_end = locs[1,j]
            arrow = patches.ConnectionPatch(
                quiver_start,
                quiver_end,
                coordsA=axes[0].transData,
                coordsB=axes[1].transData,
                # Default shrink parameter is 0 so can be omitted
                # color="black",
                color=color,
                arrowstyle="-|>",  # "normal" arrow
                mutation_scale=2*quiver_size,  # controls arrow head size
                linewidth=quiver_size,
            )
            fig.patches.append(arrow)

    # -- save --
    if prefix: fn = "%s_xfer_scatter.png" % prefix
    else: fn = "xfer_scatter.png"
    plt.savefig(root/fn)

def get_xfer_cost_single(regions,locations,masks,spix_idx,alpha):

    # -- config --
    pix = regions[:,spix_idx]
    locs = locations[:,spix_idx]
    mask = masks[:,spix_idx].bool()

    # -- center each frame's location --
    mask_e2 = mask[...,None].expand((-1,-1,2))
    means = (locs * mask_e2).sum(-2,keepdim=True)/(mask_e2.sum(-2,keepdim=True)+1e-10)
    locs = locs - means
    print("[locs is na?] : ",th.any(th.isnan(locs)).item())

    # -- shrink for easier dev --
    # print(mask.shape)
    # exit()
    # maskM0 = (mask[:-1,None]  * mask[1:,:,None])>0
    midx = th.max(th.where(mask>0)[-1])+1
    F = pix.shape[-1]
    pix = pix[:,:midx]
    locs = locs[:,:midx]
    mask = mask[:,:midx]
    B,S = mask.shape
    # print(maskM0[:,midx:,:].sum()+maskM0[:,:,midx:].sum())

    # -- sinkhorn pairs --
    # print(locs.shape)
    costC = th.cdist(locs[:-1],locs[1:])
    costC = alpha*costC + (1-alpha)*th.cdist(pix[:-1],pix[1:])
    # costC = th.cdist(locs[:-1],locs[1:])
    # print("costC.shape: ",costC.shape)
    dist0 = th.sum((locs[0,[0]] - locs[1,:])**2,-1).sqrt()
    # print(dist0[:10])
    # print("src]: ",costC[0,0,:10])
    # print("dest]: ",costC[0,:10,0])

    # maskM = (mask[:-1,:,None]  * mask[1:,None])>0
    # print("src] 0,-1: ",mask[0,-1])
    # print("dest] 1,-1: ",mask[1,-1])
    # exit()
    # maskM = (mask[:-1,None]  * mask[1:,:,None])>0
    maskM = (mask[:-1,:,None]  * mask[1:,None])>0
    # maskM[b,i,j] = if src i and dest j are valid
    # rows match src; cols match dest
    costC = costC * maskM

    # maskM = (masks[:-1,:,:,None]  * masks[1:,:,None])>0

    return costC,maskM

def run_sinkhorn_single(regions,locations,masks,spix_idx,cost_alpha,root):

    # -- get a single sample for easier dev --
    device = regions.device
    costC,maskM = get_xfer_cost_single(regions,locations,masks,spix_idx,cost_alpha)
    Bm1,S,S = costC.shape

    # -- init sinkhorn params --
    ot_scale = 5e2
    a,b = 1.*(masks[:-1,spix_idx,:S]>0),1.*(masks[1:,spix_idx,:S]>0)
    a,b = a.reshape(Bm1,S,1),b.reshape(Bm1,S,1)
    K = th.exp(-ot_scale*costC)*maskM
    # v = th.ones((Bm1,S,1),device=device)/S
    # u = th.ones((Bm1,S,1),device=device)/S
    v = b.clone()/b.sum(-2,keepdim=True)
    u = a.clone()/a.sum(-2,keepdim=True)

    niters = 0
    for iter_i in range(niters):

        # -- error --
        if iter_i % 20 == 0:
            # pi_est = th.diag_embed(u[:,:,0]) @ K @ th.diag_embed(v[:,:,0])
            pi_est = u * K.clone() * v.reshape(Bm1,1,S)
            a_est = pi_est.sum(-1,keepdim=True)
            b_est = pi_est.sum(-2,keepdim=True).reshape(Bm1,S,1)
            delta_a = th.mean((a_est - a)**2).item()
            delta_b = th.mean((b_est - b)**2).item()
            print(iter_i,delta_a,delta_b,ot_scale)
            # if (iter_i % 20) == 0 and (iter_i > 0):
            #     ot_scale = ot_scale * 2.
            #     K = th.exp(-ot_scale*costC)*maskM

        # -- updates --
        u = a / ((K @ v)+1e-10)
        v = b / ((K.transpose(-2,-1) @ u)+1e-10)
    if niters == 0:
        pi_est = K
    else:
        pi_est = u * K * v.reshape(Bm1,NS,1,S)

    # -- compute transport map --
    # pi_est = th.diag_embed(u[:,:,0]) @ K @ th.diag_embed(v[:,:,0])
    pi_est = u * K.clone() * v.reshape(Bm1,1,S)
    a_est = pi_est.sum(-1,keepdim=True)
    b_est = pi_est.sum(-2,keepdim=True).reshape(Bm1,S,1)
    delta_a = th.mean((a_est - a)**2).item()
    delta_b = th.mean((b_est - b)**2).item()

    # -- flows and weights from transport map --
    pi_vals,pi_inds = th.topk(pi_est,10,-2)
    viz_pi_sample(pi_vals,pi_inds,locations[:,spix_idx],masks[:,spix_idx],
                  root,prefix=str(spix_idx))

# -=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-
#
#
#                 Main
#
#
# -=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-

def viz_shrunk(vid,spix,marked,warped,root):

    delta = ((warped - vid[1:])**2).mean(-3)
    # img0 = vid[37:48,11:21:]
    # img1 = vid[37:48,158:168,:]
    # img2 = vid[37:48,298:308,:]
    # delta = delta[...,37:48,:]
    delta = delta[...,35:46,:]
    s0,s1 = 128+4,2*(128+4)
    delta = [delta[0,...,158-s0:168-s0],delta[1,...,298-s1:308-s1]]
    delta = th.stack(delta)
    print(delta.max(),th.quantile(delta,0.90))
    worst = delta>th.quantile(delta,0.90)
    print("WORST.shape: ",worst.shape)
    delta = th.clip(delta/(th.quantile(delta,0.90).item()+1e-10),0.,1.)
    print(delta.shape)
    tv_utils.save_image(delta[:,None],root / "shrunk_delta.png")

    print("warped.shape: ",warped.shape)
    warped = warped[...,35:46,:]
    s0,s1 = 128+4,2*(128+4)
    warped = [warped[0,...,158-s0:168-s0],
              warped[1,...,298-s1:308-s1]]
    print([m.shape for m in warped])
    warped = th.stack(warped)
    print(warped.shape)
    tv_utils.save_image(warped,root / "shrunk_warped.png")

    print("marked.shape: ",marked.shape)
    marked = marked[...,35:46,:]
    s0,s1 = 128+4,2*(128+4)
    marked = [marked[0,...,11:21],
              marked[1,...,158-s0:168-s0],
              marked[2,...,298-s1:308-s1]]
    print([m.shape for m in marked])
    marked = th.stack(marked)
    print(marked.shape)
    tv_utils.save_image(marked,root / "shrunk_marked.png")

    vid = vid[...,35:46,:]
    s0,s1 = 128+4,2*(128+4)
    vid = [vid[0,...,11:21],
              vid[1,...,158-s0:168-s0],
              vid[2,...,298-s1:308-s1]]
    print([m.shape for m in vid])
    vid = th.stack(vid)
    print(vid.shape)
    tv_utils.save_image(vid,root / "shrunk_vid.png")

    print("spix.shape: ",spix.shape)
    spix = spix[...,35:46,:]
    s0,s1 = 128+4,2*(128+4)
    spix = [spix[0,...,11:21],
              spix[1,...,158-s0:168-s0],
              spix[2,...,298-s1:308-s1]]
    print([m.shape for m in spix])
    spix = th.stack(spix)
    sshape = spix.shape
    # print(spix)
    spix = spix.reshape(-1)
    # print(spix.shape)
    spix = th.argmax(1.*(spix[:,None] == th.unique(spix)[None,:]),-1)
    spix = spix.reshape(sshape)
    # print(spix)
    spix = (spix+1) / (spix.max()+1)
    # print(spix.shape)
    tv_utils.save_image(spix[:,None],root / "shrunk_spix.png")

    # vid = vid[...,35:46,:]
    # s0,s1 = 128+4,2*(128+4)
    # vid = [vid[0,...,11:21],
    #        vid[1,...,158-s0:168-s0],
    #        vid[2,...,298-s1:308-s1]]
    # print([m.shape for m in vid])
    # vid = th.stack(vid)
    print(vid.shape)
    alpha = 0.0
    args = th.where(spix == th.mode(spix.ravel()).values.item())
    args_w = th.where(worst)
    print(args_w[1].max(),args_w[2].max())
    vid[:,0][args] = 0.
    vid[:,1][args] = 1.
    vid[:,2][args] = 0.
    # print("vid.shape: ",vid[1:,2].shape)
    vid[1:,2][args_w] = 1.
    tv_utils.save_image(vid,root / "shrunk_vid_f.png")


def main():

    # -- get root --
    root = Path("./output/create_deformable_regions/")
    if not root.exists(): root.mkdir()

    # -- config --
    sp_size = 15
    nrefine = 20
    niters,inner_niters = 1,1
    i_std,alpha,beta = 0.1,1.,10.

    # -- read img/flow --
    vid = st_spix.data.davis_example(isize=None,nframes=10,vid_names=['tennis'])
    # vid = vid[0,3:6,:,:256,:256]
    vid = vid[0,3:6,:,:128,290-128:290]
    # print("vid.shape: ",vid.shape)
    # vid = resize(vid,(156,156))

    # -- debug --
    def get_fstats(ftensor):
        min0 = ftensor.reshape(-1,2).min(0).values.round(decimals=4)
        max0 = ftensor.reshape(-1,2).max(0).values.round(decimals=4)
        min0 = min0.detach().cpu().numpy().tolist()
        max0 = max0.detach().cpu().numpy().tolist()
        return min0,max0

    # -- run flow [raft] --
    from st_spix.flow_utils import run_raft
    fflow,bflow = run_raft(vid)
    # print(get_fstats(fflow),get_fstats(bflow))
    # print(fflow.shape,bflow.shape,vid.shape)

    # -- run flow [cv2] --
    # flows = flow_pkg.run(vid,sigma=0.0,ftype="cv2")
    # fflow,bflow = flows.fflow,flows.bflow
    # print(get_fstats(fflow),get_fstats(bflow))
    # print(fflow.shape,bflow.shape,vid.shape)

    # -- shrink vid and flows --
    # vid = vid[...,32:-32,32:-32]
    # fflow = fflow[...,32:-32,32:-32]
    # print("vid.shape: ",vid.shape)
    # print("fflow.shape: ",fflow.shape)

    # -- save --
    B,F,H,W = vid.shape
    tv_utils.save_image(vid,root / "vid.png")

    # -- view --
    spix,fflow = stream_bass(vid,sp_size=sp_size,alpha=alpha,
                             beta=beta,nrefine=nrefine,fflow=fflow)
    B = spix.shape[0]
    th.cuda.empty_cache()

    # -- save --
    marked = mark_spix_vid(vid,spix)
    tv_utils.save_image(marked,root / "marked.png")

    # -- get largest superpixel size --
    mode = th.mode(spix.reshape(B,-1)).values[:,None]
    largest_spix = th.max(th.sum(spix.reshape(B,-1) == mode,-1)).item()
    R = largest_spix # "R" for "Radius"
    nspix = int(spix.max()+1)

    # -=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-
    #
    #      From Video to Superpixels
    #
    # -=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-

    # -- fill into contiguous tensor [very very silly] --
    regions,locs,inds = deform.get_regions(vid,spix,R)
    # inds = deform.get_scattering_field(spix,R)
    # print("[inds] is nan? ",th.any(th.isnan(inds)))
    # inds_e2 = inds[:,:,None].expand((-1,-1,2))
    # inds_e = inds[:,:,None].expand((-1,-1,3))
    # vid_r = rearrange(vid,'b f h w -> b (h w) f')
    # grid = futils.index_grid(H,W,normalize=True)
    # grid = repeat(grid,'1 f h w -> b (h w) f',b=B)

    # regions = th.zeros((B,nspix*R,F),device=spix.device)
    # regions = regions.scatter_(1,inds_e,vid_r)
    # regions = regions.reshape(B,nspix,R,F)

    # locs = th.zeros((B,nspix*R,2),device=spix.device)
    # locs = locs.scatter_(1,inds_e2,grid)
    # locs = locs.reshape(B,nspix,R,2)
    # print("[locs] is nan? ",th.any(th.isnan(locs)))

    masks = th.zeros((B,nspix*R),device=spix.device)
    masks = masks.scatter_(1,inds,th.ones_like(vid_r[:,:,0]))
    masks = masks.reshape(B,nspix,R)

    # print("inds.shape: ",inds.shape)
    # print(regions.shape)
    # th.save(regions,"regions.pth")

    # -=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-
    #
    #      From Superpixels to Video
    #
    # -=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-

    # -- gather --
    # npix = vid_r.shape[1]
    # vid_f = th.gather(regions.reshape(B,-1,F),1,inds_e,sparse_grad=True)
    vid_f = deform.regions_to_video(regions,inds,B,F)
    tv_utils.save_image(rearrange(vid_f,'b (h w) f -> b f h w',h=H),
                        root / "restored_img.png")
    print("Restore Video from Superpixels: ",th.mean( (vid_r - vid_f)**2 ).item())
    sp2vid_inds = inds_e

    # -=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-
    #
    #      Run Optimal Transport Examples
    #
    # -=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-

    # -- save --
    # cmap = cm.get_cmap('gist_rainbow')
    cmap = cm.get_cmap('jet')
    PICK = 4
    th.manual_seed(1)
    spix_idx_list = th.randperm(spix.max()+1)[:PICK]
    args = reduce(th.logical_or,[spix==idx for idx in spix_idx_list])
    args = th.where(args)
    marked = mark_spix_vid(vid,spix)
    print(marked.shape)
    for i in range(3): marked[:,i][args] = 0
    for i,spix_idx in enumerate(spix_idx_list):
        # print(i,spix_idx,cmap(i/1.*PICK))
        color = cmap(i/(1.*PICK))
        for c,col in enumerate(color[:3]):
            # print(c,col)
            marked[:,c][th.where(spix==spix_idx)] = col
    tv_utils.save_image(marked,root / "marked_fill.png")
    viz_sample(regions,locs,masks,spix_idx,root)

    # -- run sinkhorn --
    spix_idx = 4
    cost_alpha = 0.0
    run_sinkhorn_single(regions,locs,masks,spix_idx,cost_alpha,root)

    # -- test xfer cost --
    costC,maskM = get_xfer_cost(regions,locs,masks,cost_alpha)
    print("costC.shape: ",costC.shape)
    print("maskM.shape: ",maskM.shape)
    spix_idx_list = th.randperm(spix.max()+1)[:3]
    for spix_idx in spix_idx_list:
        costC_idx,maskM_idx = get_xfer_cost_single(regions,locs,masks,spix_idx,cost_alpha)
        S = costC_idx.shape[-1]
        delta_c = th.mean((costC[:,spix_idx,:S,:S] - costC_idx)**2).item()
        delta_m = th.mean((maskM[:,spix_idx,:S,:S]*1. - maskM_idx*1.)**2).item()
        out_c = costC[:,spix_idx,S:,:].abs().sum() + costC[:,spix_idx,:,S:].abs().sum()
        out_m = maskM[:,spix_idx,S:,:].abs().sum() + maskM[:,spix_idx,:,S:].abs().sum()
        print("delta [c,m,oc,om]: ",delta_c,delta_m,out_c.item(),out_m.item())


    # -- next... --
    vals,inds = run_sinkhorn(regions,locs,masks,spix,cost_alpha,root)

    # -- remap indices from Intra-Superpixel view to Image view --
    img_vals = rearrange(vals,'b ns k r -> b ns r k')
    # print(img_vals.shape)
    # exit()
    Bm1,NS,K,R = inds.shape
    img_inds = remap_inds(inds,locs,H,W)
    img_inds = rearrange(img_inds,'bm1 ns k r tw -> bm1 ns r k tw')
    # print(img_inds[0,0
    # print(img_inds[...,0].max(),img_inds[...,0].min())
    # print(img_inds[...,1].max(),img_inds[...,1].min())
    sp2vid_inds = sp2vid_inds[:,:,None,:1].expand((-1,-1,K,2))
    # print("img_inds.shape,sp2vid_inds.shape: ",img_inds.shape,sp2vid_inds.shape)
    img_inds = th.gather(img_inds.reshape(B-1,-1,K,2),1,sp2vid_inds[1:])
    # img_inds = img_inds.reshape(B-1,H,W,K,2)

    sp2vid_inds = sp2vid_inds[...,:1]
    # print("img_vals.shape,sp2vid_inds.shape: ",img_vals.shape,sp2vid_inds.shape)
    img_vals = th.gather(img_vals.reshape(B-1,-1,K,1),1,sp2vid_inds[1:])
    img_vals[th.where(th.isnan(img_vals))] = 0.
    img_vals = img_vals.reshape(B-1,H,W,1,K)
    if not(th.all(img_vals[...,0,0]>0).item()):
        print("There are probably holes in your output.")

    # print(img_inds[0,0,0])
    # vid_r = rearrange(vid,'b f h w -> b (h w) f')
    img_inds = img_inds.long()
    img_inds = img_inds[...,0] + img_inds[...,1]*W # rasterize; B-1,HW,K
    img_inds = img_inds[:,:,None].expand((-1,-1,F,-1)) # Bm1,HW,F,K
    # print(img_inds.min(),img_inds.max())

    # -- warp the image --
    # img_inds = img_inds[:,None].expand((-1,F,-1,-1,))
    # vid_f = th.gather(regions.reshape(B,-1,F),1,inds_e,sparse_grad=True)
    vid_r = vid_r[...,None].expand((-1,-1,-1,K)) # Bm1,HW,F,K
    # print(img_inds.shape)
    # print("vid_r.shape: ",vid_r.shape)
    warped_r = th.gather(vid_r[:-1],1,img_inds)
    warped = warped_r.reshape(B-1,H,W,F,K)
    # warped = th.mean(warped,-1)
    warped = th.sum(warped * img_vals,-1)
    warped = rearrange(warped,'b h w f -> b f h w')
    tv_utils.save_image(warped,root / "warped.png")

    delta = ((warped - vid[1:])**2).mean(-3)
    delta = delta.ravel()
    thresh = 0.001
    args_rm = th.where(delta>thresh)
    args_keep = th.where(delta<=thresh)
    print("Average difference: ",th.mean(delta[args_keep]))

    _warped = warped.clone()
    _warped[:,0].view(-1)[args_rm] = 1.
    _warped[:,1].view(-1)[args_rm] = 0.
    _warped[:,2].view(-1)[args_rm] = 0.
    tv_utils.save_image(warped,root / "warped_rmbig.png")

    viz_shrunk(vid,spix,marked,warped,root)

    # exit()
    # stack = th.gather(vid[0],img_inds)

    # -- get deformed image by first deforming each region --
    # def get_deform_region(reg0,reg1,vals,inds):
    #     print(reg0.shape,reg1.shape)
    #     print(vals.shape)
    #     print(inds.shape)
    #     exit()
    # deformed_region = get_deform_region(regions[0],regions[1],vals[0],img_inds[0],H,W)
    # deformed = th.gather(deformed_region.reshape(1,-1,F),1,sp2vid_inds[[1]])

    # # -- get deformed video from (vals,img_inds) --
    # def get_deformed_image(img0,img1,vals,inds,H,W):
    #     print(img0.shape,img1.shape)
    #     print(vals.shape)
    #     print(inds.shape)
    #     exit()
    # deformed = get_deformed_image(img[0],img1[1],vals[0],img_inds[0],H,W)
    # tv_utils.save_image(deformed,root / "marked.png")





if __name__ == "__main__":
    main()
