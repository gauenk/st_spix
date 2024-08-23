


import torch as th
import numpy as np

import st_spix
import st_spix_cuda
from st_spix import flow_utils as futils

from dev_basics import flow as flow_pkg
from dev_basics.utils.timer import ExpTimer,TimeIt
from easydict import EasyDict as edict

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

    # -- init cuda --
    grid = futils.index_grid(H,W,dtype=spix.dtype,
                             device=spix.device,normalize=True)
    grid = st_spix.sp_pool_from_spix(grid,spix,"v0")


    print(spix[0])
    for ix in range(3):

        # -- copy --
        means0 = means.clone()
        means1 = means.clone()
        flow0 = flow.clone()
        flow1 = flow.clone()

        timer.sync_start(f"v0_{ix}")
        # grid0 = st_spix.sp_pool_from_spix(grid,spix,"v0")
        flow0,means0 = st_spix.pool_flow_and_shift_mean(flow0,means0,spix,ids,version="v0")
        timer.sync_stop(f"v0_{ix}")

        timer.sync_start(f"v1_{ix}")
        flow1,means1 = st_spix.pool_flow_and_shift_mean(flow1,means1,spix,ids,version="v1")
        timer.sync_stop(f"v1_{ix}")

        print(flow0[0,0,:3,:3])
        print(flow1[0,0,:3,:3])
        # print(means0.shape,means1.shape)
        # print(means0[0,:3,:3])
        # print(means1[0,:3,:3])
        print("Differnce [flow]: ",th.mean((flow0 - flow1)**2).item())
        print("Differnce [means]: ",th.mean((means0 - means1)**2).item())

    print(timer)

if __name__ == "__main__":
    main()
