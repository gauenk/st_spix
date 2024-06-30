
# -- basics --
import torch as th
import numpy as np
from einops import rearrange,repeat
from easydict import EasyDict as edict
from torchvision.utils import save_image
from pathlib import Path

# -- data --
import data_hub

# -- import slic iterations --
import st_spix
# from st_spix import slic_iter

def main():

    # -- exp config --
    SAVE_ROOT = Path("output/results")
    if not SAVE_ROOT.exists():
        SAVE_ROOT.mkdir(parents=True)

    # -- spix config --
    spix_stride = 10
    n_iters = 10
    M = 2.5e-1
    spix_scale = 3.

    # -- data config --
    dcfg = edict()
    dcfg.dname = "davis"
    dcfg.dset = "train"
    dcfg.sigma = 1.
    dcfg.nframes = 0
    dcfg.isize = 480

    # -- load images --
    device = "cuda:0"
    data, loaders = data_hub.sets.load(dcfg)
    vid = (data.tr[1]['clean']/255.).to(device)
    # vid = vid[:2,:,256:256+256,128:128+256]
    vid = vid[:2,:,100:100+240,-240:]
    # flow = get_flow(vid[0],vid[1])

    # -- load slic iterations --
    sims_sp,sims,nsp,sftrs = st_spix.run_slic(vid,spix_stride,n_iters,M,spix_scale)
    spix = sims.argmax(1)
    spix = spix.reshape(len(vid),-1)

    # -- info --
    print(sims_sp.shape)
    print(sims.shape)
    print(nsp)
    print(sftrs.shape)

    # -- pooling --
    masked = st_spix.viz_spix(vid,spix,nsp)
    # pooled = st_spix.sp_pool(vid,spix,sims,spix_stride,nsp,"slic")
    pooled = st_spix.sp_pool(vid,sims)
    print(masked.shape)
    print(pooled.shape)

    # -- save info --
    save_fn = SAVE_ROOT / "vid.png"
    save_image(vid,save_fn)
    save_fn = SAVE_ROOT / "masked.png"
    print(masked.max(),pooled.max())
    save_image(masked,save_fn)
    save_fn = SAVE_ROOT / "pooled.png"
    save_image(pooled,save_fn)


if __name__ == "__main__":
    main()
