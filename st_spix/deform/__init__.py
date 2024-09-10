import torch as th
from einops import rearrange

def get_deformation(vid,spix,centers,covs,R):

    # -- region/locs --
    nspix = int(spix.max()+1)
    # regions,centers,inds = get_regions(vid,spix,R)
    regions,inds = get_regions(vid,spix,R)

    # -- masks --
    masks = th.zeros((B,nspix*R),device=spix.device)
    masks = masks.scatter_(1,inds,th.ones_like(vid_r[:,:,0]))
    masks = masks.reshape(B,nspix,R)

    # -- run sinkhorn --
    deform_vals,deform_inds = run_sinkhorn(regions,locs,masks,spix,cost_alpha)

    # -- remap --
    deform_vals,deform_inds = remap_deform(deform_vals,deform_inds,locs,inds)

    return deform_vals,deform_inds

def deform_vid(vid,vals,inds):
    warped_r = th.gather(vid_r[:-1],1,inds)
    warped = warped_r.reshape(B-1,H,W,F,K)
    # warped = th.mean(warped,-1)
    warped = th.sum(warped * vals,-1)
    warped = rearrange(warped,'b h w f -> b f h w')
    warped_r = th.gather(vid_r[:-1],1,inds)
    return warped_r

def remap_deform(deform_vals,deform_inds,locs,inds):

    # -- remap indices from Intra-Superpixel view to Image view --
    img_vals = rearrange(deform_vals,'b ns k r -> b ns r k')
    Bm1,NS,K,R = deform_inds.shape
    deform_inds = remap_inds(deform_inds,locs,H,W)
    deform_inds = rearrange(deform_inds,'bm1 ns k r tw -> bm1 ns r k tw')
    inds = inds[:,:,None,:1].expand((-1,-1,K,2))
    deform_inds = th.gather(deform_inds.reshape(B-1,-1,K,2),1,inds[1:])
    deform_inds = deform_inds.long()

    # -- remap values from intra-superpixel view to image view  --
    inds = inds[...,:1]
    deform_vals = th.gather(deform_vals.reshape(B-1,-1,K,1),1,inds[1:])
    deform_vals[th.where(th.isnan(deform_vals))] = 0.
    deform_vals = deform_vals.reshape(B-1,H,W,1,K)
    if not(th.all(deform_vals[...,0,0]>0).item()):
        print("There are probably holes in your output.")

    return deform_vals,deform_inds

def regions_to_video(regions,inds,B,F):
    B,F = regions.shape[0],regions.shape[-1]
    vid = th.gather(regions.reshape(B,-1,F),1,inds,sparse_grad=True)
    return vid

def get_regions(vid,spix,R):

    inds = get_scattering_field(spix,R)
    inds_e2 = inds[:,:,None].expand((-1,-1,2))
    inds_e = inds[:,:,None].expand((-1,-1,3))
    vid_r = rearrange(vid,'b f h w -> b (h w) f')
    grid = futils.index_grid(H,W,normalize=True)
    grid = repeat(grid,'1 f h w -> b (h w) f',b=B)

    regions = th.zeros((B,nspix*R,F),device=spix.device)
    regions = regions.scatter_(1,inds_e,vid_r)
    regions = regions.reshape(B,nspix,R,F)

    return regions

    # locs = th.zeros((B,nspix*R,2),device=spix.device)
    # locs = locs.scatter_(1,inds_e2,grid)
    # locs = locs.reshape(B,nspix,R,2)

    # return regions,locs

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


def run_sinkhorn(regions,centers,covs,masks,spix,cost_alpha):

    # -- get a single sample for easier dev --
    device = regions.device
    costC,maskM = get_xfer_cost(regions,centers,covs,masks,cost_alpha)
    Bm1,NS,S,S = costC.shape

    # -- init sinkhorn params --
    ot_scale = 1
    a,b = 1.*(masks[:-1]>0),1.*(masks[1:]>0)
    a,b = a.reshape(Bm1,NS,S,1),b.reshape(Bm1,NS,S,1)
    K = th.exp(-ot_scale*costC)*maskM

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
    beta = 2.
    pi_vals,pi_inds = th.topk(pi_est,5,-2)
    pi_vals = th.softmax(beta*pi_vals,-2)

    return pi_vals,pi_inds


def get_xfer_cost(regions,centers,covs,masks,alpha):

    # -- center each frame's location --
    masks = masks.bool()
    masks_e2 = masks[...,None].expand((-1,-1,-1,2))
    means = (centers * masks_e2).sum(-2,keepdim=True)
    means = means / (masks_e2.sum(-2,keepdim=True)+1e-10)
    centers = centers - means

    # -- cost --
    costC = th.cdist(centers[:-1],centers[1:])
    costC = alpha*costC + (1-alpha)*th.cdist(regions[:-1],regions[1:])
    maskM = (masks[:-1,:,:,None]  * masks[1:,:,None])>0
    costC = costC * maskM

    return costC,maskM

