
# -- basics --
import torch as th
import numpy as np
from einops import rearrange,repeat
from easydict import EasyDict as edict
from torchvision.utils import save_image,make_grid
from pathlib import Path

# -- import slic iterations --
import st_spix
from st_spix import utils
# from st_spix import slic_iter

def main():

    # -- exp config --
    SAVE_ROOT = Path("output/results")
    if not SAVE_ROOT.exists():
        SAVE_ROOT.mkdir(parents=True)
    utils.seed_everything(0)

    # -- spix config --
    device = "cuda:0"
    spix_stride = 10
    n_iters = 20
    M = 0.
    spix_scale = 2.
    vid = st_spix.data.davis_example()
    print("n_iters: ",n_iters)
    print("spix_scale: ",spix_scale)
    print("vid.shape: ",vid.shape)

    # -- load slic iterations --
    vid0 = vid.clone().requires_grad_(True)
    sims_sp,sims0,nsp,sftrs0 = st_spix.run_slic(vid0,spix_stride,n_iters,M,
                                                spix_scale,grad_type='full')
    spix = sims0.argmax(1).reshape(len(vid),-1)

    # -- fixed sim spix --
    B,NSP,NP = sims0.shape
    print(spix.max(),spix.min())
    print(spix.shape)
    print(sims0.shape)
    # exit()
    sims1 = th.zeros_like(sims0)
    # sims1[:,spix] = 1.
    batch_inds = th.arange(B).unsqueeze(-1)
    pix_inds = th.arange(NP).unsqueeze(0)
    sims1[batch_inds,spix,pix_inds] = 1.
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

    # -- quantiles --
    print("quantiles.")
    quants = th.tensor([0.1,0.2,0.5,0.75,0.99]).to(device)
    thresh = th.quantile(grad0.ravel(),quants)[3].item()
    print(th.quantile(grad0.ravel(),quants))
    print(th.quantile(grad1.ravel(),quants))
    print(th.quantile(grad2.ravel(),quants))
    print("\n")
    print("thresh: ",thresh)

    # -- compare --
    names = ["full","spix","sprobs"]
    grads = [grad0,grad1,grad2]
    for ix,grad_i in enumerate(grads):
        if ix == 0: continue
        args = th.where(grad0>thresh)
        # print(args)
        diff = th.sum(((grad_i[args] - grad0[args])/(grad0[args]+1e-15))**2).item()
        # diff = th.sum((grad_i - grad0)**2).item()
        print("%s: %2.3e" % (names[ix],diff))

    # -- prepare for saving --
    grads = [grad0,grad1,grad2]
    normz_fxn = lambda vid,smax,smin: (vid-smin)/(smax-smin)
    max_quant = lambda grad: th.quantile(grad,th.tensor([0.99]).to(device)).item()
    min_quant = lambda grad: th.quantile(grad,th.tensor([0.01]).to(device)).item()
    share_max = max([max_quant(grad) for grad in grads])
    share_min = min([min_quant(grad) for grad in grads])
    grads = [th.clamp(grad,share_min,share_max) for grad in grads]
    grad0,grad1,grad2 = grads
    # share_max = max([grad0.max(), grad1.max(), grad2.max()])
    # share_min = min([grad0.min(), grad1.min(), grad2.min()])
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
