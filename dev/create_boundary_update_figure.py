
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

def shade_circle_spix(marked,spix):
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
    cmap = plt.get_cmap("coolwarm")

    # mask_color = cmap(spix.cpu().numpy()/(spix.max().item()+1))[:, :, :3]
    # mask_color = cmap(spix.cpu().numpy())[:, :, :3]
    # colors = (cmap(np.arange(nspix)/(1.*nspix-1))[:, :3]*255.).tolist()
    # colors = ["blue","purple","green"]
    # colors = ["#bae1ff","#eecbff","#d4ffea"]
    colors = ["#afeff9","#adb2fb","#78dfb9"]
    smarked = draw_segmentation_masks(marked[:3],spix_m,alpha=1.,colors=colors)
    smarked = th.cat([smarked,alpha[None,]])

    # -- include yellow border --
    args = th.where(th.logical_and(marked[0]>0.8,marked[1]>0.8))
    print(args)
    for i in [0,1,2]:
        smarked[i][args] = marked[i][args]

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

def main():

    # -- setup --
    print("PID: ",os.getpid())

    # -- output --
    root = Path("output/create_boundary_update_figure/")
    if not root.exists(): root.mkdir(parents=True)

    # -- read data --
    vid = st_spix.data.davis_example(isize=None,nframes=5,vid_names=['kid-football'])[0]
    vid = vid[3:,:,:400,150:150+400]
    tv_utils.save_image(vid,root / "vid.png")
    spix = get_spix(vid)
    marked = mark_spix_vid(vid,spix)
    tv_utils.save_image(marked,root / "marked.png")
    tv_utils.save_image(marked[[0]],root / "marked0.png")
    # center = (280,100)
    center = (250,60)
    radius = 12
    zoom_factor = 8

    # -- show circle on image --
    # save_circle_on_image(vid,center,radius,root)

    # -- zoomed --
    circle = zoomed_circle(vid[0], center, radius, zoom_factor)
    tv_utils.save_image(circle[None,:],root / "circle.png")
    marked_circle = zoomed_circle(marked[0], center, radius, zoom_factor)
    tv_utils.save_image(marked_circle[None,:],root / "marked_circle.png")

    # -- [alt] circular spix; confirm spix xform --
    spix_circ = zoomed_circle(spix[0], center, radius, zoom_factor)[0].int()
    alt_marked_circle = mark_spix_vid(circle[None,:3],spix_circ[None,])
    tv_utils.save_image(alt_marked_circle,root / "alt_marked_circle.png")

    # -- shade regions of the marked circle --
    shaded = shade_circle_spix(marked_circle,spix_circ)
    tv_utils.save_image(shaded[None,:],root / "shaded_circle.png")


if __name__ == "__main__":
    main()
