
import os
import shutil
from pathlib import Path


import pandas as pd
import numpy as np
import torch as th
from einops import rearrange,repeat

import matplotlib.pyplot as plt
from PIL import Image, ImageDraw
import torchvision.transforms.functional as tvf
from torchvision.transforms.functional import InterpolationMode
from torchvision.transforms.functional import resize


import st_spix
import prop_cuda
import torchvision.utils as tv_utils
from st_spix.spix_utils import mark_spix_vid,img4bass
from st_spix.prop import stream_bass,indepent_bass


def _zoomed_circle(img, center, radius, zoom_factor=2):

    # Get dimensions of the image
    device = img.device
    print(img.shape)
    img = rearrange(img,'f h w -> h w f').cpu().numpy()
    if img.shape[-1] == 1: img = img[...,0]
    h, w = img.shape[:2]
    img = np.clip(255*img,0,255).astype(np.uint8)
    print(img.shape)
    img = Image.fromarray(img)

    # Calculate the bounding box for the zoomed region
    x1 = max(center[0] - radius, 0)
    y1 = max(center[1] - radius, 0)
    x2 = min(center[0] + radius, w)
    y2 = min(center[1] + radius, h)

    # Crop and resize (zoom)
    cropped_img = img.crop((x1, y1, x2, y2))
    zoomed_img = cropped_img.resize((2 * radius, 2 * radius), Image.LANCZOS)

    # Create a circular mask
    mask = Image.new("L", (2 * radius, 2 * radius), 0)
    draw = ImageDraw.Draw(mask)
    draw.ellipse((0, 0, 2 * radius, 2 * radius), fill=255)

    # Apply the mask to the zoomed image to get the circular crop
    result = Image.new("RGBA", (2 * radius, 2 * radius))
    result.paste(zoomed_img, (0, 0), mask)
    H,W = result.size
    result = result.resize((zoom_factor*H,zoom_factor*W),Image.LANCZOS)

    # -- output --
    zoomed_circle_img = np.array(result)
    zoomed_circle_img = rearrange(zoomed_circle_img,'h w f -> f h w')/255.

    return th.from_numpy(zoomed_circle_img).to(device)


def zoomed_circle(img, center, radius, zoom_factor=2):

    # Calculate the bounding box for the zoomed region
    if img.ndim == 2: img = repeat(img,'h w -> f h w',f=3)
    _,h,w = img.shape
    x1 = max(center[0] - radius, 0)
    y1 = max(center[1] - radius, 0)
    x2 = min(center[0] + radius, w)
    y2 = min(center[1] + radius, h)

    # Crop and resize (zoom)
    cropped_img = img[:,y1:y2,x1:x2]
    zoomed_img = resize(cropped_img,(2*radius,2*radius),InterpolationMode.NEAREST)

    # Create a circular mask
    grid = th.arange(2 * radius).to(img.device)
    y, x = th.meshgrid(grid,grid)
    mask = ((x - radius) ** 2 + (y - radius) ** 2 <= radius ** 2).float()

    # Stack the alpha channel to the zoomed image
    zoomed = th.cat([zoomed_img, mask.unsqueeze(0)], dim=0)
    _,H,W = zoomed.shape
    zoomed = resize(zoomed,(zoom_factor*H,zoom_factor*W),InterpolationMode.NEAREST)

    return zoomed

def shade_circle_spix(marked,spix,k):

    # -- include yellow border --
    print(spix.shape)
    border = prop_cuda.find_border(spix[None,:].int())[0]
    for i in range(1,4):
        border[i:] = th.logical_or(border[i:],border[:-i])
        border[:,i:] = th.logical_or(border[:,i:],border[:,:-i])

    # -- remove all non-primary spix --
    spix = keep_only_topk(spix,marked[-1],k)
    print(th.unique(spix))

    # -- relabel spix --
    codes,_ = pd.factorize(spix.ravel().cpu().numpy())
    spix = th.from_numpy(codes).to(spix.device).reshape_as(spix)
    nspix = spix.max().item()+1
    print("num spix: ",nspix)

    # -- alpha --
    from torchvision.utils import draw_segmentation_masks
    alpha = marked[-1].clone()
    spix_m = th.nn.functional.one_hot(spix.long()).bool()
    spix_m = rearrange(spix_m,'h w m -> m h w')

    # Get the colormap from matplotlib and apply it
    # cmap = plt.get_cmap("Pastel2")
    # mask_color = cmap(spix.cpu().numpy()/(spix.max().item()+1))[:, :, :3]
    # mask_color = cmap(spix.cpu().numpy())[:, :, :3]
    # colors = (cmap(np.arange(nspix)/(1.*nspix-1))[:, :3]*255.).tolist()
    # colors = ["blue","purple","green"]
    # colors = ["#afeff9","#adb2fb","#78dfb9","#ffdfba"]

    # -- use these! --
    # colors = ["#f0f0f0","#adb2fb","#78dfb9","#ffdfba"]

    # -- dev --
    colors = ["#f0f0f0","#78dfb9","#adb2fb","#ffdfba"]

    smarked = draw_segmentation_masks(marked[:3],spix_m,alpha=1.,colors=colors)
    # smarked = draw_segmentation_masks(marked[:3],spix_m,alpha=1.)
    smarked = th.cat([smarked,alpha[None,]])

    args = th.where(border)
    for i in [0,1,2]:
        smarked[i][args] = i in [0,1]

    # marked[-1] *= alpha
    # marked[-1] = marked[-1] * mask
    return smarked


def get_spix(vid):
    vid = rearrange(vid,'t f h w -> t h w f')
    vid = vid.contiguous()
    sp_size = 30
    niters = sp_size
    niters_seg = 4
    alpha_hastings = 0.
    potts = 1.0
    sigma2_app = .01
    sigma2_size = 1.
    spix = indepent_bass(vid,niters=niters,niters_seg=niters_seg,
                         sp_size=sp_size,sigma2_app=sigma2_app,
                         alpha_hastings=alpha_hastings,
                         potts=potts,sm_start=0,rgb2lab=True)
    return spix

def get_split_spix(img):
    img = rearrange(img,'f h w -> 1 h w f')
    img = img.contiguous()
    sp_size = 30
    niters = sp_size
    niters_seg = 4
    # alpha_hastings = 110000.
    alpha_hastings = 0.
    potts = 2.0
    sigma2_app = .01
    sigma2_size = 1e10
    seg0 = prop_cuda.init_spix(img,sp_size)
    # seg1 = prop_cuda.refine(img,seg0,niters,sp_size,sigma2_app,sigma2_size,potts)
    seg0,params = prop_cuda.bass(img,0,0,0,sp_size,sigma2_app,sigma2_size,
                                 potts,alpha_hastings)
    seg1,params,_ = prop_cuda.refine(img,seg0,params,3,
                                     niters_seg,sp_size,sigma2_app,potts)
    print(seg1.min(),seg1.max())
    seg2,params = prop_cuda.split(img,seg1,params,1,
                           sp_size,sigma2_app,sigma2_size,alpha_hastings)
    seg2,params = prop_cuda.split(img,seg2,params,0,
                                  sp_size,sigma2_app,sigma2_size,alpha_hastings)
    print(seg2.min(),seg2.max())
    segs = [seg0,seg1,seg2]
    seg_ix = seg2
    for ix in range(5):
        seg_ix,params,_ = prop_cuda.refine(img,seg_ix,params,1,
                                           1,sp_size,sigma2_app,potts)
        segs.append(seg_ix)
    seg_ix,params,_ = prop_cuda.refine(img,seg_ix,params,5,
                                       niters_seg,sp_size,sigma2_app,potts)
    segs.append(seg_ix)
    segs = th.cat(segs)
    # exit()
    return segs

def keep_only_topk(spix_circ,alpha,k=2):
    topk = get_topk_freq(spix_circ,alpha,k)
    spix_circ[th.where(~th.isin(spix_circ,topk))] = 0 # always to zero
    return spix_circ

def get_topk_freq(spix_circ,alpha,k=2):
    args = th.where(alpha>0)
    hist = th.bincount(spix_circ[args].ravel())
    vals,inds = th.topk(hist,k=k)
    return inds

def main():

    # -- setup --
    print("PID: ",os.getpid())

    # -- output --
    root = Path("output/create_split_cls_figure/")
    if not root.exists(): root.mkdir(parents=True)

    # -- read data --
    vid = st_spix.data.davis_example(isize=None,nframes=5,vid_names=['kid-football'])[0]
    vid = vid[3:,:,:400,150:150+400]
    vid_lab = st_spix.utils.vid_rgb2lab(vid)
    tv_utils.save_image(vid,root / "vid.png")
    img = vid[0]
    img_l = vid_lab[0]
    spix = get_split_spix(img_l)
    img_r = repeat(img,'f h w -> r f h w',r=len(spix))
    print(spix.shape,img_r.shape)
    marked = mark_spix_vid(img_r,spix)
    tv_utils.save_image(marked,root / "marked.png")
    tv_utils.save_image(marked[[0]],root / "marked0.png")
    tv_utils.save_image(marked[[1]],root / "marked1.png")
    # exit()
    # center = (280,100)
    # center = (250,60)

    # -- back --
    # center = (340,250)
    center = (375,256)
    center = (372,226)
    center = (350,250)
    radius = 25
    zoom_factor = 8

    # -- show circle on image --
    # save_circle_on_image(vid,center,radius,root)

    # -- zoomed --
    circle = zoomed_circle(vid[0], center, radius, zoom_factor)
    tv_utils.save_image(circle[None,:],root / "circle.png")
    for i in range(len(marked)):
        marked_circle = zoomed_circle(marked[i], center, radius, zoom_factor)
        tv_utils.save_image(marked_circle[None,:],root / ("marked_circle%d.png"%i))

        # -- [alt] circular spix; confirm spix xform --
        # marked_circle = zoomed_circle(marked[i], center, radius, 1)
        spix_circ = zoomed_circle(spix[i], center, radius, zoom_factor)[0].int()
        # spix_circ = zoomed_circle(spix[i], center, radius, 1)[0].int()

        # codes,_ = pd.factorize(spix_circ.ravel().cpu().numpy())
        # spix_circ = th.from_numpy(codes).to(spix.device).reshape_as(spix_circ)
        # tv_utils.save_image(spix_circ[None,None]/spix_circ.max(),
        #                     root / ("spix_circle_%d.png"%i))
        # # exit()

        # spix_circ = keep_only_topk(spix_circ,_circle[3],k=3)
        alt_marked_circle = mark_spix_vid(circle[None,:3],spix_circ[None,])
        tv_utils.save_image(alt_marked_circle,root / ("alt_marked_circle_%d.png"%i))

        # -- shade regions of the marked circle --
        k = 1 if i < 2 else 2
        shaded = shade_circle_spix(marked_circle,spix_circ,k)
        tv_utils.save_image(shaded[None,:],root / ("shaded_circle_%d.png"%i))


if __name__ == "__main__":
    main()
