
import torch as th
import numpy as np
from einops import rearrange,repeat
from pathlib import Path

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
from st_spix.sp_pooling import pooling,SuperpixelPooling

import stnls
from dev_basics import flow as flow_pkg

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

def get_scattering_field(spix,R):

    # -- unpack --
    shape = spix.shape
    B = spix.shape[0]
    spix = spix.reshape(B,1,-1)
    npix = spix.shape[-1]

    # -- find matches --
    ids = th.arange(spix.max()+1)[None,:,None].to(spix.device)
    match = spix == ids

    # -- increment along pixels to get index within superpixel --
    csum = th.cumsum(match,-1)-1

    # -- fill index with superpixel coordinate within superpixel --
    inds = th.where(match)
    sinds = -th.ones((B,npix),device=spix.device,dtype=th.long)
    sinds[inds[0],inds[2]] = csum[inds[0],inds[1],inds[2]]
    assert (th.max(sinds) == (R-1))

    # -- offset sinds with superpixel label offset --
    sinds = sinds + spix.reshape(B,-1)*R
    return sinds

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


def get_xfer_cost(regions,locations,masks):

    # -- center each frame's location --
    # print("locations.shape: ",locations.shape)
    masks = masks.bool()
    masks_e2 = masks[...,None].expand((-1,-1,-1,2))
    means = (locations * masks_e2).sum(-2,keepdim=True)/masks_e2.sum(-2,keepdim=True)
    locations = locations - means
    # locations[:,:,mask,0] -= locations[:,:,mask,0].mean(-1)
    # locations[:,:,mask,1] -= locations[:,:,mask,1].mean(-1)

    # -- cost --
    costC = th.cdist(locations[:-1],locations[1:])
    maskM = (masks[:-1,:,:,None]  * masks[1:,:,None])>0
    # maskM = maskM.transpose(-2,-1)
    costC = costC * maskM
    # costC = th.cdist(locs[:-1],locs[1:])
    # maskM = (mask[:-1,:,None]  * mask[1:,None])>0

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

def run_sinkhorn(regions,locations,masks,spix,root):

    # -- get a single sample for easier dev --
    device = regions.device
    costC,maskM = get_xfer_cost(regions,locations,masks)
    ot_scale = 1e3
    K = th.exp(-ot_scale*costC)*maskM
    # print("masks.shape: ",masks.shape)
    # print("K.shape: ",K.shape)
    Bm1,NS,S,S = K.shape

    # -- init sinkhorn params --
    ot_scale = 5e2
    a,b = 1.*(masks[:-1]>0),1.*(masks[1:]>0)
    a,b = a.reshape(Bm1,NS,S,1),b.reshape(Bm1,NS,S,1)
    K = th.exp(-ot_scale*costC)*maskM

    # -- only one iter --
    # u = a / (K.mean(-1,keepdim=True)+1e-10)
    # v = b / (K.mean(-2,keepdim=True).transpose(-2,-1)+1e-10)
    # pi_est = u * K * v.reshape(Bm1,NS,1,S)

    # -- many iters --
    niters = 100
    v = b.clone()/b.sum(-2,keepdim=True)
    u = a.clone()/a.sum(-2,keepdim=True)
    for iter_i in range(niters):
        u = a / ((K @ v)+1e-10)
        v = b / ((K.transpose(-2,-1) @ u)+1e-10)
    pi_est = u * K * v.reshape(Bm1,NS,1,S)

    # -- flows and weights from transport map --
    spix_idx = 20
    pi_vals,pi_inds = th.topk(pi_est[:,spix_idx],10,-2)
    viz_pi_sample(pi_vals,pi_inds,locations[:,spix_idx],
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
    for j in [100]:
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

def get_xfer_cost_single(regions,locations,masks,spix_idx):

    # -- config --
    pix = regions[:,spix_idx]
    locs = locations[:,spix_idx]
    mask = masks[:,spix_idx].bool()

    # -- center each frame's location --
    mask_e2 = mask[...,None].expand((-1,-1,2))
    means = (locs * mask_e2).sum(-2,keepdim=True)/mask_e2.sum(-2,keepdim=True)
    locs = locs - means

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

    return costC,maskM

def run_sinkhorn_single(regions,locations,masks,spix_idx,root):

    # -- get a single sample for easier dev --
    device = regions.device
    costC,maskM = get_xfer_cost_single(regions,locations,masks,spix_idx)
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

    niters = 100
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
            if (iter_i % 20) == 0 and (iter_i > 0):
                ot_scale = ot_scale * 2.
                K = th.exp(-ot_scale*costC)*maskM

        # -- updates --
        u = a / ((K @ v)+1e-10)
        v = b / ((K.transpose(-2,-1) @ u)+1e-10)

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

def main():

    # -- get root --
    root = Path("./output/create_deformable_regions/")
    if not root.exists(): root.mkdir()

    # -- config --
    npix_in_side = 40
    niters,inner_niters = 1,2
    i_std,alpha,beta = 0.1,0.1,10.

    # -- read img/flow --
    vid = st_spix.data.davis_example(isize=None,nframes=10,vid_names=['tennis'])
    vid = vid[0,3:6,:,:480,:480]
    vid = resize(vid,(256,256))
    flows = flow_pkg.run(vid[None,:],sigma=0.0,ftype="cv2")
    B,F,H,W = vid.shape

    # -- view --
    spix,fflow = stream_bass(vid,sp_size=20,beta=5.)
    B = spix.shape[0]

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
    inds = get_scattering_field(spix,R)
    inds_e2 = inds[:,:,None].expand((-1,-1,2))
    inds_e = inds[:,:,None].expand((-1,-1,3))
    vid_r = rearrange(vid,'b f h w -> b (h w) f')
    grid = futils.index_grid(H,W,normalize=True)
    grid = repeat(grid,'1 f h w -> b (h w) f',b=B)

    regions = th.zeros((B,nspix*R,F),device=spix.device)
    regions = regions.scatter_(1,inds_e,vid_r)
    regions = regions.reshape(B,nspix,R,F)

    locs = th.zeros((B,nspix*R,2),device=spix.device)
    locs = locs.scatter_(1,inds_e2,grid)
    locs = locs.reshape(B,nspix,R,2)

    masks = th.zeros((B,nspix*R),device=spix.device)
    masks = masks.scatter_(1,inds,th.ones_like(vid_r[:,:,0]))
    masks = masks.reshape(B,nspix,R)

    print("inds.shape: ",inds.shape)
    print(regions.shape)
    th.save(regions,"regions.pth")

    # -=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-
    #
    #      From Superpixels to Video
    #
    # -=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-

    # -- gather --
    npix = vid_r.shape[1]
    vid_f = th.gather(regions.reshape(B,-1,F),1,inds_e,sparse_grad=True)
    print("Difference: ",th.mean( (vid_r - vid_r)**2 ).item())

    # -=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-
    #
    #      Run Optimal Transport Examples
    #
    # -=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-

    # -- save --
    spix_id = 20
    marked = mark_spix_vid(vid,spix)
    marked[:,0][th.where(spix==spix_id)] = 1.
    marked[:,1][th.where(spix==spix_id)] = 0.
    marked[:,2][th.where(spix==spix_id)] = 0.
    tv_utils.save_image(marked,root / "marked_fill.png")
    viz_sample(regions,locs,masks,spix_id,root)

    # -- run sinkhorn --
    spix_idx = 20
    run_sinkhorn_single(regions,locs,masks,spix_idx,root)

    # -- test xfer cost --
    costC,maskM = get_xfer_cost(regions,locs,masks)
    print("costC.shape: ",costC.shape)
    print("maskM.shape: ",maskM.shape)
    spix_idx_list = [20,50,100]
    for spix_idx in spix_idx_list:
        costC_idx,maskM_idx = get_xfer_cost_single(regions,locs,masks,spix_idx)
        S = costC_idx.shape[-1]
        delta_c = th.mean((costC[:,spix_idx,:S,:S] - costC_idx)**2).item()
        delta_m = th.mean((maskM[:,spix_idx,:S,:S]*1. - maskM_idx*1.)**2).item()
        out_c = costC[:,spix_idx,S:,:].abs().sum() + costC[:,spix_idx,:,S:].abs().sum()
        out_m = maskM[:,spix_idx,S:,:].abs().sum() + maskM[:,spix_idx,:,S:].abs().sum()
        print("delta [c,m,oc,om]: ",delta_c,delta_m,out_c.item(),out_m.item())


    # -- next... --
    run_sinkhorn(regions,locs,masks,spix,root)


if __name__ == "__main__":
    main()
