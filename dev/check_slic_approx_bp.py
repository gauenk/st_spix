
# -- basics --
import torch as th
import numpy as np
from einops import rearrange,repeat
from easydict import EasyDict as edict
from torchvision.utils import save_image,make_grid
from pathlib import Path

# -- data --
import data_hub

# -- import slic iterations --
import st_spix
# from st_spix import slic_iter

def get_fun_imgs():

    # -- data config --
    dcfg = edict()
    dcfg.dname = "davis"
    dcfg.dset = "train"
    dcfg.sigma = 1.
    dcfg.nframes = 1
    dcfg.isize = 480

    # -- load images --
    device = "cuda:0"
    data, loaders = data_hub.sets.load(dcfg)
    # vid_names = ["bear", "bmx-bumps", "boat", "boxing-fisheye", "breakdance-flare",
    #              "bus", "car-turn", "cat-girl", "classic-car", "color-run", "crossing",
    #              "dance-jump", "dancing", "disc-jockey", "dog-agility", "dog-gooses",
    #              "dogs-scale", "drift-turn", "drone"]
    # vid_names = ["bear", "bmx-bumps", "boat", "boxing-fisheye"]
    # vid_names = data.tr.vid_names[:3]
    # print(vid_names)
    vid_names = ["bmx-bumps","boxing-fisheye","dancing"]
    isel = {"bmx-bumps":[150,240],"boxing-fisheye":[200,200]}
    vid = []
    for name in vid_names:
        _index = data_hub.filter_subseq(data.tr,name,frame_start=0,frame_end=0)[0]
        _vid = data.tr[_index]['clean']/255.
        if name in isel: sh,sw = isel[name]
        else: sh,sw = 0,0
        _vid = _vid[:1,:,sh:sh+240,sw:sw+240].to(device)
        vid.append(_vid)
    vid = th.cat(vid)

    return vid

def main():

    # -- exp config --
    SAVE_ROOT = Path("output/results")
    if not SAVE_ROOT.exists():
        SAVE_ROOT.mkdir(parents=True)

    # -- spix config --
    spix_stride = 10
    n_iters = 10
    M = 2.5e-3
    spix_scale = 2.
    vid = get_fun_imgs()
    print("vid.shape: ",vid.shape)

    # -- load slic iterations --
    vid0 = vid.clone().requires_grad_(True)
    sims_sp,sims0,nsp,sftrs0 = st_spix.run_slic(vid0,spix_stride,n_iters,M,
                                                spix_scale,grad_type='full')
    spix = sims0.argmax(1).reshape(len(vid),-1)

    # -- fixed sim spix --
    sims1 = th.zeros_like(sims0)
    sims1[:,spix] = 1.
    vid1 = vid.detach().clone().requires_grad_(True)
    sftrs1,sims1 = st_spix.compute_slic_params(vid1,sims1,spix_stride,M,spix_scale)

    # -- fixed sim probs --
    sims2 = sims0.detach().clone()
    vid2 = vid.detach().clone().requires_grad_(True)
    sftrs2,sims2 = st_spix.compute_slic_params(vid2,sims2,spix_stride,M,spix_scale)

    # -- forward --
    pooled0 = st_spix.sp_pool(vid0,sims0)
    pooled1 = st_spix.sp_pool(vid1,sims1)
    pooled2 = st_spix.sp_pool(vid2,sims2)

    # -- backward --
    print(pooled0.max(),pooled0.min())
    print(pooled1.max(),pooled1.min())
    print(pooled2.max(),pooled2.min())
    tgt = th.rand_like(pooled0)-0.5
    th.mean((pooled0-tgt)**2).backward()
    th.mean((pooled1-tgt)**2).backward()
    th.mean((pooled2-tgt)**2).backward()

    # -- grads --
    grad0,grad1,grad2 = vid0.grad,vid1.grad,vid2.grad

    # -- prepare for saving --
    normz_fxn = lambda vid,smax,smin: (vid-smin)/(smax-smin)
    share_max = max([grad0.max(), grad1.max(), grad2.max()])
    share_min = min([grad0.min(), grad1.min(), grad2.min()])
    print(share_max,share_min)
    grad0_nmz = normz_fxn(grad0,share_max,share_min)
    grad1_nmz = normz_fxn(grad1,share_max,share_min)
    grad2_nmz = normz_fxn(grad2,share_max,share_min)

    print("normalized stats")
    print(grad0_nmz.min(),grad0_nmz.max())
    print(grad1_nmz.min(),grad1_nmz.max())
    print(grad2_nmz.min(),grad2_nmz.max())


    # -- save --
    save_fn = SAVE_ROOT / "vid.png"
    save_image(vid,save_fn)
    save_fn = SAVE_ROOT / "grad0.png"
    save_image(grad0_nmz,save_fn)
    save_fn = SAVE_ROOT / "grad1.png"
    save_image(grad1_nmz,save_fn)
    save_fn = SAVE_ROOT / "grad2.png"
    save_image(grad2_nmz,save_fn)

    # -- save grid --
    print(vid.shape,grad0_nmz.shape)
    grid = th.cat([vid,grad0_nmz,grad1_nmz,grad2_nmz])
    grid = rearrange(grid,'(group name) f h w -> (name group) f h w',group=4)
    grid = make_grid(grid,nrow=4)
    save_fn = SAVE_ROOT / "grid.png"
    save_image(grid,save_fn)

if __name__ == "__main__":
    main()
