"""

    Investigate using a KD-tree

"""

import torch as th
import numpy as np
from einops import rearrange,repeat
# from scipy.spatial import KDTree
from pykdtree.kdtree import KDTree


import st_spix
import st_spix_cuda
from st_spix import flow_utils as futils

from dev_basics import flow as flow_pkg
from dev_basics.utils.timer import ExpTimer,TimeIt
from easydict import EasyDict as edict

def run_pwd_v0(means,cov,gscatter,invalid,H,W,cdist=False):

    # -- means --
    locs = th.stack([means[:,-2]/(W-1),means[:,-1]/(H-1)],-1)
    gscatter = rearrange(gscatter,'b f h w -> (b h w) f')
    if cdist:
        dists = th.cdist(gscatter,locs)
    else:
        # locs = rearrange(locs,'n f -> (n f) 1')
        # gscatter = rearrange(gscatter,'n f -> (n f) 1')
        # d = th.cdist(gscatter,locs)
        dx = th.cdist(gscatter[:,:1],locs[:,:1])
        dy = th.cdist(gscatter[:,1:],locs[:,1:])
        sx,sxy,sy,logDet = cov[:,0],cov[:,1],cov[:,2],cov[:,3]
        # print(dx.shape,dy.shape,sx.shape,sy.shape,logDet.shape)
        dists = th.exp(-(dx**2)*sx - (dy**2)*sy - 2*dx*dy*sxy - logDet)

    # -- gather --
    spix_grid = th.arange(means.shape[0]).to(means.device) #  I think "0" in invalid?
    shifted_spix = spix_grid[dists.argmin(1)].int()
    shifted_spix = rearrange(shifted_spix,'(b h w) -> b h w',h=H,w=W)
    shifted_spix[invalid] = -1
    return shifted_spix


def run_pwd_v1(kd,means,cov,gscatter,invalid,H,W):

    # -- get xy --
    # xy_s = rearrange(gscatter,'1 f h w -> (h w) f').cpu().numpy()
    kd = KDTree(means[0,:,-2:].cpu().numpy())
    xy_s = rearrange(gscatter,'1 f h w -> (h w) f').cpu().numpy()
    dd, ii = kd.query(xy_s,k=1)
    print(means.shape)
    print(np.unique(ii))
    print(ii.shape,xy_s.shape)
    # print(dd)
    # print(ii)
    # print(dd.shape)
    # print(ii.shape)

    # exit()

    # # -- means --
    # locs = th.stack([means[:,-2]/(W-1),means[:,-1]/(H-1)],-1)
    # gscatter = rearrange(gscatter,'b f h w -> (b h w) f')
    # dists = th.cdist(gscatter,locs)

    # # -- gather --
    # spix_grid = th.arange(means.shape[0]).to(means.device) #  I think "0" in invalid?
    # shifted_spix = spix_grid[dists.argmin(1)].int()
    # shifted_spix = rearrange(shifted_spix,'(b h w) -> b h w',h=H,w=W)
    # shifted_spix[invalid] = -1
    # return shifted_spix


def main():

    # -- load images --
    vid = st_spix.data.davis_example(isize=None,nframes=10)[:1,:10,:,:480,:480]
    # vid = vid + (25./255.)*th.randn_like(vid)
    vid = th.clip(255.*vid,0.,255.).type(th.uint8)
    B,T,F,H,W = vid.shape
    flows = flow_pkg.run(vid/255.,sigma=0.0,ftype="cv2")
    flow = flows.fflow[0,0][None,:]


    # -- get spix --
    npix_in_side = 80
    niters,inner_niters = 1,40
    i_std,alpha,beta = 0.1,0.001,100.
    img = st_spix.utils.img4bass(vid[:,0])
    bass_fwd = st_spix_cuda.bass_forward
    spix,means,cov,counts,ids = bass_fwd(img,npix_in_side,i_std,alpha,beta)
    nspix = spix.max().item()+1

    # -- init timer --
    timer = ExpTimer()

    # -- init shift labels --
    grid = futils.index_grid(H,W,dtype=spix.dtype,
                             device=spix.device,normalize=True)
    flow_sp,_means = st_spix.pool_flow_and_shift_mean(flow,means.clone(),spix,ids)
    grid0 = st_spix.sp_pool_from_spix(grid,spix)
    gscatter,gcnts = st_spix.scatter.run(grid0,flow_sp,swap_c=True) # moves puzzle pieces
    eps = 1e-13
    invalid = th.where(th.logical_or(gcnts<1-eps,1+eps<gcnts))
    for i in range(2):
        gscatter[:,i][invalid] = -100.


    # -- init tree --
    xy_s = rearrange(gscatter,'1 f h w -> (h w) f').cpu().numpy()
    kd = KDTree(means[0,:,-2:].cpu().numpy())


    for ix in range(3):

        # -- copy --
        means0 = means.clone()
        means1 = means.clone()
        cov0 = cov.clone()
        cov1 = cov.clone()
        gscatter0 = gscatter.clone()
        gscatter1 = gscatter.clone()

        timer.sync_start(f"v0_{ix}")
        st_spix0 = run_pwd_v0(means0[0],cov0[0],gscatter0,invalid,H,W)
        timer.sync_stop(f"v0_{ix}")

        timer.sync_start(f"v1_{ix}")
        st_spix1 = run_pwd_v1(kd,means1,cov1,gscatter1,invalid,H,W)
        timer.sync_stop(f"v1_{ix}")

        # -- info --
        # print("Differnce [spix]: ",th.mean((1.*(st_spix0 - st_spix1))**2).item())

    print(timer)



if __name__ == "__main__":
    main()


