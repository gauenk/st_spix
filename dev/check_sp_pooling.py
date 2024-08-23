


import torch as th
import numpy as np

import st_spix
import st_spix_cuda
from st_spix import flow_utils as futils
from dev_basics.utils.timer import ExpTimer,TimeIt
from easydict import EasyDict as edict

def main():

    # -- load images --
    vid = st_spix.data.davis_example(isize=None,nframes=10)[:1,:10,:,:480,:480]
    # vid = vid + (25./255.)*th.randn_like(vid)
    vid = th.clip(255.*vid,0.,255.).type(th.uint8)
    B,T,F,H,W = vid.shape

    # -- get spix --
    npix_in_side = 80
    niters,inner_niters = 1,40
    i_std,alpha,beta = 0.1,0.001,100.
    img = st_spix.utils.img4bass(vid[:,0])
    bass_fwd = st_spix_cuda.bass_forward
    spix,means,cov,counts,ids = bass_fwd(img,npix_in_side,i_std,alpha,beta)

    # -- init timer --
    timer = ExpTimer()

    # -- init cuda --
    grid = futils.index_grid(H,W,dtype=spix.dtype,
                             device=spix.device,normalize=True)
    grid = st_spix.sp_pool_from_spix(grid,spix,"v0")


    print(spix[0])
    for ix in range(3):

        timer.sync_start(f"v0_{ix}")
        grid0 = st_spix.sp_pool_from_spix(grid,spix,"v0")
        timer.sync_stop(f"v0_{ix}")

        timer.sync_start(f"v1_{ix}")
        grid1 = st_spix.sp_pool_from_spix(grid,spix,"v1")
        timer.sync_stop(f"v1_{ix}")

        print(grid0.shape,grid1.shape)
        print(grid0[0,:3,:3])
        print(grid1[0,:3,:3])
        print("Differnce: ",th.mean((grid0 - grid1)**2).item())

    print(timer)

if __name__ == "__main__":
    main()
