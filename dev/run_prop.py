"""

      Execute the Algorithm's Pipeline

"""

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
import prop_cuda

from torchvision.transforms.functional import resize

from st_spix import scatter
from st_spix import deform
from st_spix.sp_pooling import pooling,SuperpixelPooling

import stnls
from dev_basics import flow as flow_pkg

import matplotlib as mpl
import matplotlib.cm as cm
from matplotlib import colormaps
from matplotlib import patches, pyplot as plt
# import matplotlib.pyplot as plt

from st_spix.prop import stream_bass,run_fwd_bwd

def draw_spix_vid(vid,spix):
    viz_seg = []
    nspix = spix.max().item()+1
    for t in range(vid.shape[0]):
        spix_t = spix[t].clone()
        spix_t[th.where(spix_t<0)] = 1000
        viz_seg.append(draw_spix(vid[t],spix_t,nspix))
    viz_seg = th.stack(viz_seg)
    return viz_seg/255.

def color_spix(vid,spix,spix_id,cidx=0):
    for t in range(vid.shape[0]):
        for ci in range(3):
            vid[t,ci][th.where(spix[t]==spix_id)] = 1.*(ci==cidx)
    return vid

def draw_spix(img,spix,nspix):
    masks = th.nn.functional.one_hot(spix.long(),num_classes=nspix).bool()
    masks = masks.permute(2,0,1)
    # nspix = spix.max().item()+1
    # viridis = mpl.colormaps['tab20'].resampled(nspix)
    viridis = mpl.colormaps['jet'].resampled(nspix)
    scolors = [list(255*a for a in viridis(ix/(1.*nspix))[:3]) for ix in range(nspix)]
    print(img.min(),img.max())
    img = th.clip(255*img,0.,255.).type(th.uint8)
    # print(img.shape,masks.shape)
    # print(masks[0])
    marked = tv_utils.draw_segmentation_masks(img,masks,colors=scolors)
    return marked

def inspect_means(vid,spix,params,sp_size):
    pix = []
    from st_spix.flow_utils import index_grid
    B,F,H,W = vid.shape
    grid = index_grid(H,W,dtype=th.float,device="cuda",normalize=False)
    grids = []
    for t in range(vid.shape[0]):

        pix_t = []
        for c in range(3):
            pix_t.append(vid[t,c][th.where(spix[t]==8)])
        pix_t = th.stack(pix_t)
        pix.append(pix_t)

        grids_t = []
        for c in range(2):
            grids_t.append(grid[0,c][th.where(spix[t]==8)])
        grids_t = th.stack(grids_t)
        grids.append(grids_t)

    print("pix[0].shape: ",pix[0].shape)
    print("pix[1].shape: ",pix[1].shape)
    print("grids[0].shape: ",grids[0].shape)
    print("grids[1].shape: ",grids[1].shape)

    m0 = params[0].mu_app[8]
    s0 = params[0].mu_shape[8]
    cov0 = params[0].sigma_shape[8]
    det0 = params[0].logdet_sigma_shape[8]
    c0 = params[0].counts[8]
    m1 = params[1].mu_app[8]
    c1 = params[1].counts[8]
    s1 = params[1].mu_shape[8]
    cov1 = params[1].sigma_shape[8]
    det1 = params[1].logdet_sigma_shape[8]
    print(params[0].prior_sigma_shape)

    # print(".")
    # print(params[0].prior_counts)
    # print(params[1].prior_counts)
    print("-"*10)
    sprior0 = params[0].prior_counts[8]
    sprior1 = params[1].prior_counts[8]
    x0,y0 = grids[0][0,:],grids[0][1,:]
    x1,y1 = grids[1][0,:],grids[1][1,:]
    mu_x0,mu_y0 = th.mean(x0),th.mean(y0)
    mu_x1,mu_y1 = th.mean(x1),th.mean(y1)
    xx0,yy0,xy0 = th.sum(x0*x0),th.sum(y0*y0),th.sum(x0*y0)
    xx1,yy1,xy1 = th.sum(x1*x1),th.sum(y1*y1),th.sum(x1*y1)

    print(sprior0,c0,sprior1,c1)
    # -- manually recompute distances --
    c00_0 = (sprior0**2 + xx0 - mu_x0 * mu_x0 * c0)/(c0 + sprior0 - 3.)
    c01_0 = (0       + xy0 - mu_x0 * mu_y0 * c0)/(c0 + sprior0 - 3.)
    c11_0 = (sprior0**2 + yy0 - mu_y0 * mu_y0 * c0)/(c0 + sprior0 - 3.)
    print("c00_0,c01_0,c11_0: ",c00_0.item(),c01_0.item(),c11_0.item())
    detC_0 = c00_0 * c11_0 - c01_0 * c01_0
    x0,y0,z0 = c11_0/detC_0,-c01_0/detC_0,c00_0/detC_0

    c00_1 = (sprior1**2 + xx1 - mu_x1 * mu_x1 * c1)/(c1 + sprior1 - 3.)
    c01_1 = (0       + xy1 - mu_x1 * mu_y1 * c1)/(c1 + sprior1 - 3.)
    c11_1 = (sprior1**2 + yy1 - mu_y1 * mu_y1 * c1)/(c1 + sprior1 - 3.)
    print("c00_1,c01_1,c11_1: ",c00_1.item(),c01_1.item(),c11_1.item())
    detC_1 = c00_1 * c11_1 - c01_1 * c01_1
    x1,y1,z1 = c11_1/detC_1,-c01_1/detC_1,c00_1/detC_1

    # -- info --
    print("-"*10 + "-- cov --" + "-"*10)
    print(x0.item(),y0.item(),z0.item(),detC_0.item())
    print(x1.item(),y1.item(),z1.item(),detC_1.item())
    print("-"*10)
    print("-"*10 + "-- sp cov --" + "-"*10)
    print(cov0)
    print(cov1)
    print(det0.item(),det1.item())
    print("-"*10)

    # print("-"*10)
    # print(" -- cov -- ")
    # print(th.linalg.pinv(th.cov(grids[0])))
    # print(th.linalg.pinv(th.cov(grids[1])))
    # print("-"*10)

    # print("-"*10)
    # print(pix[0].mean(-1),pix[1].mean(-1))
    # print(grids[0].mean(-1),grids[1].mean(-1))

    print("-"*10 + "-- means --" + "-"*10)
    print(s0)
    print(mu_x0,mu_y0)
    print(s1)
    print(mu_x1,mu_y1)
    print("^"*10)
    print("^"*10)

def main():

    # -- get root --
    root = Path("./output/run_prop/")
    if not root.exists(): root.mkdir()

    # -- config --
    niters = 80
    niters_seg = 4
    sm_start = 10
    sp_size = 15
    alpha_hastings,potts = 1.,10.
    pix_var = 0.09

    # -- read img/flow --
    vid = st_spix.data.davis_example(isize=None,nframes=10,vid_names=['tennis'])
    # vid = st_spix.data.davis_example(isize=None,nframes=10,vid_names=['baseball'])
    size = 256
    # vid = vid[0,5:7,:,50:50+size,300:300+size]
    vid = vid[0,2:4,:,50:50+size,200:200+size]
    vid = resize(vid,(128,128))
    vid_og = vid.clone()

    # -- run flow [raft] --
    from st_spix.flow_utils import run_raft
    fflow,bflow = run_raft(th.clip(255.*vid,0.,255.).type(th.uint8))
    # print(vid.shape,fflow.shape)
    # fflow,bflow = run_raft(vid)
    if fflow.shape[-1] != vid.shape[-1]:
        print("RAFT wants image size to be a multiple of 8.")
        exit()

    # -- resize again --
    # vid = resize(vid,(64,64))
    # fflow = resize(fflow,(64,64))/2. # reduced scale by 2
    size = 128
    vid = resize(vid,(size,size))
    fflow = resize(fflow,(size,size))/(128./size) # reduced scale by X


    # -- save --
    B,F,H,W = vid.shape
    tv_utils.save_image(vid,root / "vid.png")

    # -- propogate --
    outs = stream_bass(vid,flow=fflow,
                       niters=niters,niters_seg=niters_seg,
                       sp_size=sp_size,pix_var=pix_var,
                       alpha_hastings=alpha_hastings,
                       potts=potts,sm_start=sm_start)
    spix,params,children,missing,pmaps = outs
    print("[og] 8: ",params[0].mu_app[8])

    # -- view --
    marked = mark_spix_vid(vid,spix)
    marked_m = marked.clone()
    marked_m[1:] = (1-1.*missing.cpu())*marked_m[1:]
    marked_c = color_spix(marked.clone(),spix,2,cidx=1)
    marked_c = color_spix(marked_c,spix,8,cidx=0)
    # marked_c = color_spix(marked_c,spix,9,cidx=1)
    # # marked_c = color_spix(marked_c,spix,10,cidx=0)
    # # marked_c = color_spix(marked_c,spix,11,cidx=1)
    # marked_c = color_spix(marked_c,spix,12,cidx=2)

    # -- save --
    print("saving images.")
    # viz_seg = draw_spix_vid(vid,spix)
    futils.viz_flow_quiver(root/"flow.png",fflow[[0]],step=4)
    tv_utils.save_image(marked,root / "marked_fill.png")
    tv_utils.save_image(marked_m,root / "marked_missing.png")
    tv_utils.save_image(marked_c,root / "marked_colored.png")
    # tv_utils.save_image(viz_seg,root / "viz_seg.png")

    # -- vizualize the lab values with the means --
    vid_lab = st_spix.utils.vid_rgb2lab(vid,normz=False)
    print([(vid_lab[:,i].min().item(),vid_lab[:,i].max().item()) for i in range(3)])
    inspect_means(vid_lab,spix,params,sp_size)

    # -- copy before refinement --
    spix_og = spix.clone()
    params_og = [st_spix.copy_spix_params(p) for p in params]
    border_og = prop_cuda.find_border(spix_og)

    # -- run fwd/bwd --
    niters_ref = 15
    niters_fwd_bwd = 1
    pix_var = 0.1
    potts = 2.
    # print("8: ",params[0].mu_app[8],params[0].counts[8])
    spix,params = run_fwd_bwd(vid_og,spix,params,pmaps,sp_size,pix_var,
                              potts,niters_fwd_bwd,niters_ref)
    # print("8:" ,params[0].mu_app[8],params[0].counts[8])
    border_b = prop_cuda.find_border(spix)
    spix,params = run_fwd_bwd(vid_og,spix,params,pmaps,sp_size,pix_var,
                              potts,niters_fwd_bwd,niters_ref)
    border_c = prop_cuda.find_border(spix)

    # -- view --
    marked = mark_spix_vid(vid,spix)
    marked_m = marked.clone()
    marked_m[1:] = (1-1.*missing.cpu())*marked_m[1:]
    marked_c = color_spix(marked.clone(),spix,2,cidx=0)
    marked_c = color_spix(marked_c,spix,3,cidx=2)

    # -- save --
    print("saving images.")
    # viz_seg = draw_spix_vid(vid,spix)
    tv_utils.save_image(marked,root / "fwdbwd_marked_fill.png")
    tv_utils.save_image(marked_m,root / "fwdbwd_marked_missing.png")
    tv_utils.save_image(marked_c,root / "fwdbwd_marked_colored.png")
    # tv_utils.save_image(viz_seg,root / "viz_seg.png")

    mvid = st_spix.spix_utils.mark_border(vid,border_og,0)
    tv_utils.save_image(mvid,root / "double_border_a.png")
    mvid = st_spix.spix_utils.mark_border(vid,border_b,0)
    tv_utils.save_image(mvid,root / "double_border_b.png")
    mvid = st_spix.spix_utils.mark_border(vid,border_c,0)
    tv_utils.save_image(mvid,root / "double_border_c.png")
    # mvid = st_spix.spix_utils.mark_border(mvid,border_c,1)
    # mvid = st_spix.spix_utils.mark_border(mvid,border_og,2)


    # -- vizualize the lab values with the means --
    # vid_lab = st_spix.utils.vid_rgb2lab(vid)
    # print([(vid_lab[:,i].min().item(),vid_lab[:,i].max().item()) for i in range(3)])
    # inspect_means(vid_lab,spix,params)

    # -- view --
    vid_lab = vid_lab - vid_lab.min()
    vid_lab = vid_lab / vid_lab.max()
    print(vid.shape,vid.max(),vid.min())
    print(vid_lab.shape,vid_lab.max(),vid_lab.min())
    marked = mark_spix_vid(vid_lab,spix)
    marked_m = marked.clone()
    marked_m[1:] = (1-1.*missing.cpu())*marked_m[1:]
    marked_c = color_spix(marked.clone(),spix,2,cidx=0)
    marked_c = color_spix(marked_c,spix,3,cidx=2)

    # -- save --
    print("saving images.")
    tv_utils.save_image(marked,root / "lab_marked_fill.png")
    tv_utils.save_image(marked_m,root / "lab_marked_missing.png")
    tv_utils.save_image(marked_c,root / "lab_marked_colored.png")
    # tv_utils.save_image(viz_seg,root / "viz_seg.png")




if __name__ == "__main__":
    main()
