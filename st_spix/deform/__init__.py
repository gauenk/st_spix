import torch as th
from einops import rearrange,repeat
from st_spix import flow_utils as futils

def get_deformation(vid,spix,centers,covs):

    # -- get largest superpixel size --
    device = vid.device
    B,F,H,W = vid.shape
    mode = th.mode(spix.reshape(B,-1)).values[:,None]
    largest_spix = th.max(th.sum(spix.reshape(B,-1) == mode,-1)).item()
    R = largest_spix # "R" for "Radius"
    nspix = int(spix.max()+1)

    # -- pix/locs --
    pix,locs,inds = get_regions(vid,spix,R,nspix)

    # -- masks --
    masks = th.zeros((B,nspix*R),device=device)
    masks = masks.scatter_(1,inds,th.ones((B,H*W),device=device,dtype=th.float32))
    masks = masks.reshape(B,nspix,R)

    # -- run sinkhorn --
    # [centers and cov are used here]
    cost_alpha = 0.5
    deform_vals,deform_inds = run_sinkhorn(pix,locs,masks,centers,covs,spix,cost_alpha)

    # -- remap --
    deform_vals,deform_inds = remap_deform(deform_vals,deform_inds,locs,inds,H,W)

    # print("deform_vals.shape: ",deform_vals.shape)
    # print("deform_inds.shape: ",deform_inds.shape)
    # exit()

    return deform_vals,deform_inds

def warp_video(vid,deform_vals,deform_inds):
    # vid = B,F,H,W
    B,F,H,W = vid.shape
    Bm1,_H,_W,K,_ = deform_vals.shape
    Bm1,_H,_W,K,two = deform_inds.shape
    assert (H ==_H) and (W == _W),"Must match."

    # -- prepare inds --
    deform_inds = rearrange(deform_inds,'b h w k a -> b (h w) k a')
    deform_inds = deform_inds.long()
    deform_inds = deform_inds[...,0] + deform_inds[...,1]*W # rasterize; B-1,HW,K
    deform_inds = deform_inds[...,None].expand((-1,-1,-1,F)) # Bm1,HW,F,K

    # -- prepare video --
    vid = rearrange(vid,'b f h w -> b (h w) 1 f')
    vid = vid.expand((-1,-1,K,-1)) # Bm1,HW,K,F
    # print("vid.shape: ",vid.shape,deform_inds.shape)
    # print(deform_inds.max(),deform_inds.min())
    # exit()

    # -- warp --
    # print("vid.min(),vid.max(): ",vid.min(),vid.max())
    # print(deform_vals.sum(1)
    warped = th.gather(vid[:-1],1,deform_inds)
    warped = warped.reshape(B-1,H,W,K,F)
    # print("deform_vals.hape: ",deform_vals.shape)
    # exit()
    warped = th.sum(warped * deform_vals,-2)
    warped = rearrange(warped,'b h w f -> b f h w')
    return warped

def inds2flow(inds,H,W):
    inds = rearrange(inds,'b h w k two -> b k two h w',h=H)
    # inds = inds.flip(2)
    grid = futils.index_grid(H,W,normalize=False)[:,None]
    # print(inds[:,:,:,10,20])
    # print(inds[:,:,:,20,10])
    # print(grid[:,:,:,10,20])
    # print(grid[:,:,:,20,10])

    # print(flow.shape,grid.shape)
    # print(flow.min(),flow.max())
    flow = inds - grid # ?
    # print(flow.shape,grid.shape)
    # print(flow.min(),flow.max())
    # exit()

    # flow[:,:,0] = flow[:,:,0] * (W-1) # ? -1
    # flow[:,:,1] = flow[:,:,1] * (H-1)
    # -- think in spix_utils.py [pool...] --
    # means[...,-2] = means[...,-2] + downsampled[...,0]
    # means[...,-1] = means[...,-1] + downsampled[...,1]
    return flow

def deform_vid(vid,vals,inds):
    warped_r = th.gather(vid_r[:-1],1,inds)
    warped = warped_r.reshape(B-1,H,W,F,K)
    # warped = th.mean(warped,-1)
    warped = th.sum(warped * vals,-1)
    warped = rearrange(warped,'b h w f -> b f h w')
    warped_r = th.gather(vid_r[:-1],1,inds)
    return warped_r

def remap_deform(deform_vals,deform_inds,locs,inds,H,W):

    # -- first inds augment --
    img_vals = rearrange(deform_vals,'b ns k r -> b ns r k')
    Bm1,NS,K,R = deform_inds.shape
    inds = inds[:,:,None].expand((-1,-1,3))
    inds = inds[:,:,None,:1].expand((-1,-1,K,2))

    # -- remap indices from Intra-Superpixel view to Image view --
    deform_inds = remap_inds(deform_inds,locs,H,W)
    deform_inds = rearrange(deform_inds,'bm1 ns k r tw -> bm1 ns r k tw')
    deform_inds = th.gather(deform_inds.reshape(Bm1,-1,K,2),1,inds[1:])
    # print("deform_inds.shape: ",deform_inds.shape)
    deform_inds = deform_inds.long()
    deform_inds = deform_inds.reshape(Bm1,H,W,K,2)

    # -- remap values from intra-superpixel view to image view  --
    inds = inds[...,:1]
    deform_vals = th.gather(deform_vals.reshape(Bm1,-1,K,1),1,inds[1:])
    deform_vals[th.where(th.isnan(deform_vals))] = 0.
    # print("deform_vals.shape: ",deform_vals.shape)
    # exit()
    deform_vals = deform_vals.reshape(Bm1,H,W,K,1)
    if not(th.all(deform_vals[...,0,0]>0).item()):
        print("There are probably holes in your output.")

    return deform_vals,deform_inds

def remap_inds(inds,locations,H,W):
    """
        Convert indices from Intra-Superpixels to Image Space
    """
    B,NS,R,two = locations.shape
    Bm1,NS,K,R = inds.shape
    locations = locations[:,:,None].expand((-1,-1,K,-1,-1))
    inds = inds[...,None].expand((-1,-1,-1,-1,2)) # Bm1,NS,K,R,[2]
    img_inds = th.gather(locations[:-1],3,inds)
    img_inds[...,0] = img_inds[...,0]*(W-1)
    img_inds[...,1] = img_inds[...,1]*(H-1)
    img_inds = img_inds.long()
    return img_inds

def regions_to_video(regions,inds,B,F):
    B,F = regions.shape[0],regions.shape[-1]
    vid = th.gather(regions.reshape(B,-1,F),1,inds,sparse_grad=True)
    return vid

def get_regions(vid,spix,R,nspix):

    B,F,H,W = vid.shape
    inds = get_scattering_field(spix,R).long()
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

    return regions,locs,inds

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


def run_sinkhorn(pix,locs,masks,centers,covs,spix,cost_alpha):

    # -- get a single sample for easier dev --
    device = pix.device
    costC,maskM = get_xfer_cost(pix,locs,masks,centers,covs,cost_alpha)
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


def get_xfer_cost(pix,locs,masks,centers,covs,alpha):

    # -- center each frame's location --
    masks = masks.bool()
    masks_e2 = masks[...,None].expand((-1,-1,-1,2))
    means = (locs * masks_e2).sum(-2,keepdim=True)
    means = means / (masks_e2.sum(-2,keepdim=True)+1e-10)
    locs = locs - means

    # -- cost --
    costSpace = th.cdist(locs[:-1],locs[1:])
    costPix = th.cdist(pix[:-1],pix[1:])
    costC = alpha*costSpace + (1-alpha)*costPix
    maskM = (masks[:-1,:,:,None]  * masks[1:,:,None])>0
    costC = costC * maskM

    return costC,maskM

