
"""

             Execute the Algorithm's Pipeline

   Fill the prior counts with "current counts" after shifting...
   .... this is good for smaller superpixels maybe...
   ...  but bigger superpixels are too big then...

"""

import os
import torch as th
import numpy as np
from einops import rearrange,repeat
from pathlib import Path
from functools import reduce
from st_spix.attn.sna import SuperpixelAttention
from st_spix.models import get_sims
from torchvision.transforms.functional import InterpolationMode


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
from st_spix.sp_pooling import pooling,SuperpixelPooling,sp_pooling

import stnls
from dev_basics import flow as flow_pkg

import matplotlib as mpl
import matplotlib.cm as cm
from matplotlib import colormaps
from matplotlib import patches, pyplot as plt
# import matplotlib.pyplot as plt

from st_spix.prop import stream_bass,run_fwd_bwd,indepent_bass

from dev_basics.utils.timer import ExpTimer,TimeIt
from dev_basics.utils.gpu_mem import GpuMemer,MemIt

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

def read_mnist():
    fn = "/home/gauenk/Documents/data/mnist/train_images.npy"
    import numpy as np
    images = np.load(fn)>0.5
    images = repeat(images[:3],'b h w -> b r h w',r=3)
    images = th.from_numpy(images)
    return images*1.

def read_figure_image():
    import torchvision.io as tvio

    # -- tiger --
    # fn = Path("/home/gauenk/Documents/packages/spix_paper/data/DIV2K/DIV2K_train_LR_bicubic/X4/0010x4.png") # tiger
    # img = tvio.read_image(fn)/255.
    # img = img[:,:408,:504]

    # -- crab --
    # fn = "/home/gauenk/Documents/packages/spix_paper/data/DIV2K/DIV2K_train_LR_bicubic/X4/0040x4.png" # crab
    # img = tvio.read_image(fn)/255.
    # img = img[:,135:320,70:350]

    # -- red-pandas --
    # fn = "/home/gauenk/Documents/packages/spix_paper/data/DIV2K/DIV2K_train_LR_bicubic/X4/0064x4.png"
    # img = tvio.read_image(fn)/255.
    # img = img[:,:336,:504]

    # -- catepillar --
    # fn = "/home/gauenk/Documents/packages/spix_paper/data/bsd500/HR/35028.jpg"
    # img = tvio.read_image(fn)/255.
    # img = img[:,180:308,60:252]

    # -- stack and return --
    vid = th.stack([img,img,img]).to("cuda")
    print("vid.shape: ",vid.shape)
    return vid


def read_elephants():
    import torchvision.io as tvio
    root = Path("/home/gauenk/Documents/packages/st_spix/data/figures/")
    name = "elephant_frame%d.png"
    vid = []
    # minH,minW = 704,1008
    minH,minW = 704,704
    for idx in range(3):
        if idx == 0: continue
        fn = root / (name % idx)
        img = tvio.read_image(fn)/255.
        print(img.shape)
        F,H,W = img.shape
        minH = H if H < minH else minH
        minW = W if W < minW else minW
        vid.append(img)
    for idx in range(len(vid)):
        vid[idx] = vid[idx][:,:minH,:minW]
    vid = th.stack(vid)
    vid = resize(vid,(352,504)).to("cuda")
    # vid = resize(vid,(352,352)).to("cuda")
    return vid

def rzs(vid):
    return resize(vid,(128,128)).to("cuda")

def read_otter():
    import torchvision.io as tvio
    fn = "../spix_paper/data/otters.png"
    img = tvio.read_image(fn)/255.
    img = th.stack([img,img])
    return img

def save_spix_parts(root,img,spix):
    uniq = spix.unique()
    nspix = len(uniq)
    import torchvision.io as tvio
    import torchvision.utils as tv_utils
    # print("img.min(),img.max(): ",img.min().item(),img.max().item())
    # exit()
    root = root /"parts"
    if not root.exists(): root.mkdir(parents=True)

    print(spix.shape)
    print("img.shape: ",img.shape)
    img = th.cat([img,th.zeros_like(img[:1])],0)
    print("[0] img.shape: ",img.shape)
    for ix in range(nspix):
        fn = root / ("%d.png"%ix)
        img_ix = img.clone()
        img[-1] = spix == uniq[ix]
        # tv_utils.save_image(img[None,].detach().cpu(),fn)

        # -- shrink box to nz alpha --
        fn = root / ("s_%d.png"%ix)
        inds_h,inds_w = th.where(spix == uniq[ix])
        min_h,max_h = inds_h.min().item(),inds_h.max().item()
        min_w,max_w = inds_w.min().item(),inds_w.max().item()
        # print(img.shape,min_h,max_h,min_w,max_w)
        crop = img[None,:,min_h:max_h,min_w:max_w].detach().cpu()
        tv_utils.save_image(crop,fn)

        # -- shrink box to nz alpha --
        # fn = root / ("b_%d.png"%ix)
        # inds_h,inds_w = th.where(spix == uniq[ix])
        # min_h,max_h = inds_h.min().item(),inds_h.max().item()
        # min_w,max_w = inds_w.min().item(),inds_w.max().item()
        # # print(img.shape,min_h,max_h,min_w,max_w)
        # crop = img[None,:,min_h:max_h,min_w:max_w].detach().cpu()
        # crop[:,:3] = 0
        # crop[:,2] = 1.
        # tv_utils.save_image(crop,fn)
    # exit()

def color_regions(marked,regions):
    from torchvision.utils import draw_bounding_boxes
    marked = th.clip(marked*255,0,255.).type(th.uint8)
    cmarked = []
    regions = th.tensor(regions).type(th.long)
    for mark in marked:
        cmark = draw_bounding_boxes(mark, regions[[0]], colors="blue",fill=True)
        cmark = draw_bounding_boxes(cmark, regions[[1]], colors="red",fill=True)
        cmarked.append(cmark)
    cmarked = th.stack(cmarked)/255.
    return cmarked

def save_zoom_vid(vid,region,name):
    def crop(img,region):
        sw,sh,ew,eh = region
        return img[:,sh:eh,sw:ew]
    zvid = []
    for img in vid:
        zvid.append(crop(img,region))
    print(zvid[0].shape)
    grid = tv_utils.make_grid(zvid)
    grid = grid[:,2:-2,2:-2]
    tv_utils.save_image(grid,name)
    # tv_utils.save_image(zvid,name)

def save_spix_img(root,img,spix):
    _img = rearrange(img,'f h w -> 1 h w f')
    pooled,down = sp_pooling(_img,spix[None,:])
    pooled = rearrange(pooled,'1 h w f -> 1 f h w')
    tv_utils.save_image(pooled,root/"spix.png")

    import torchvision
    import matplotlib.cm as cm
    import colorsys


    spix_e = th.nn.functional.one_hot(spix.long()+1).bool()
    spix_e = rearrange(spix_e,'h w m -> m h w')
    nspix = spix_e.shape[0]
    print("nspix: ",nspix)

    # -- get colors --
    # cmap = cm.get_cmap('hsv', nspix)
    cmap = mpl.colormaps['hsv'].resampled(nspix)
    colors = cmap(np.arange(0,cmap.N))
    # cmap = cm.get_cmap('tab20', nspix)
    # colors = th.tensor(cmap.colors[:, :3] * 255).to(th.uint8)
    colors = th.tensor(colors[:, :3])
    order = th.randperm(len(colors))
    colors = colors[order]
    # print(len(colors))

    # Apply the colormap to each superpixel mask
    colored_masks = th.zeros_like(img).float()
    for i in range(nspix):
        # Select the color for this segment
        color = colors[i].float()

        # Apply the color to the regions defined by the mask
        mask = spix_e[i]
        colored_masks[0][mask] = color[0]  # Red channel
        colored_masks[1][mask] = color[1]  # Green channel
        colored_masks[2][mask] = color[2]  # Blue channel

    # Convert the image back to the correct format if necessary
    # colored_masks = th.clip(colored_masks * 255, 0, 255).type(th.uint8)
    print(colored_masks.shape)

    # -- mute the vibrant colors --
    colored_masks = decrease_saturation(colored_masks, factor=0.5)

    tv_utils.save_image(colored_masks[None,:],root/"seg.png")

    # img = th.clip(img*255,0,255).type(th.uint8)
    # seg = torchvision.utils.draw_segmentation_masks(img,spix_e)
    # tv_utils.save_image(seg[None,:],root/"seg.png")
    # exit()

# Function to decrease saturation for a NumPy array
def decrease_saturation(image_array, factor=0.5):

    import colorsys
    image_array = rearrange(image_array,'c h w -> h w c')
    image_array = image_array.cpu().numpy()
    # Normalize RGB values to range [0, 1]
    # image_array = image_array / 255.0

    # Separate the RGB channels
    r, g, b = image_array[..., 0], image_array[..., 1], image_array[..., 2]

    # Convert RGB to HSV using vectorized approach
    hsv = np.vectorize(colorsys.rgb_to_hsv)(r, g, b)
    h, s, v = hsv[0], hsv[1], hsv[2]

    # Decrease saturation by factor
    s = s * factor
    v = v * 0.9

    # Convert back to RGB
    rgb = np.vectorize(colorsys.hsv_to_rgb)(h, s, v)

    # Stack the R, G, B channels and convert back to the range [0, 255]
    muted_image_array = np.stack(rgb, axis=-1)

    # Clip values to ensure they are within valid pixel range
    # muted_image_array = np.clip(muted_image_array, 0, 255).astype(np.uint8)

    muted_image_array = rearrange(muted_image_array,'h w c -> c h w')
    muted_image_array = th.from_numpy(muted_image_array)
    return muted_image_array


def zoomed_circle(img, center, radius, zoom_factor=2, fill_center=False):

    # Calculate the bounding box for the zoomed region
    if img.ndim == 2: img = repeat(img,'h w -> f h w',f=3)
    _,h,w = img.shape
    x1 = max(center[0] - radius, 0)
    y1 = max(center[1] - radius, 0)
    x2 = min(center[0] + radius, w)
    y2 = min(center[1] + radius, h)

    # Crop and resize (zoom)
    cropped_img = img[:,y1:y2,x1:x2].clone()
    if fill_center:
        cH,cW = cropped_img.shape[-2:]
        cropped_img[...,cH//2,cW//2] = 1.
    zoomed_img = resize(cropped_img,(2*radius-1,2*radius-1),InterpolationMode.NEAREST)
    # print(zoomed_img.shape)
    # exit()

    # Create a circular mask
    grid = th.arange(2 * radius-1).to(img.device)+1
    # print(grid)
    # exit()
    y, x = th.meshgrid(grid,grid)
    mask = ((x - radius) ** 2 + (y - radius) ** 2 <= radius ** 2).float()
    dists = (x - radius) ** 2 + (y - radius) ** 2
    # print(dists[:3,:3])
    # print(dists[13:16,13:16])
    # print(dists[-3:,-3:])
    mask[th.where(mask.sum(-1,keepdim=True)==1)] = 0
    mask[th.where(mask.sum(-2,keepdim=True)==1)] = 0

    # Stack the alpha channel to the zoomed image
    zoomed = th.cat([zoomed_img, mask.unsqueeze(0)], dim=0)
    _,H,W = zoomed.shape
    zoomed = resize(zoomed,(zoom_factor*H,zoom_factor*W),InterpolationMode.NEAREST)

    return zoomed

def create_gaussian_kernel(kernel_size=5, sigma=2.0):
    """Creates a 2D Gaussian kernel."""

    x, y = th.meshgrid(th.arange(-(kernel_size // 2),
                                       kernel_size // 2 + 1, dtype=th.float32),
                          th.arange(-(kernel_size // 2),
                                       kernel_size // 2 + 1, dtype=th.float32))
    kernel = th.exp(-(x**2 + y**2) / (2 * sigma**2))
    kernel = kernel / kernel.sum()

    return kernel

def main():

    # -- get root --
    print("PID: ",os.getpid())
    root = Path("./output/create_spixconv_figure/")
    if not root.exists(): root.mkdir()

    # -- config --
    niters_seg = 4
    sm_start = 0
    sp_size = 30
    niters = sp_size
    alpha_hastings = 0.
    potts = 1.0
    sigma2_app = .01
    sigma2_size = 1.
    K = 7

    # -- read img/flow --
    vid = st_spix.data.davis_example(isize=None,nframes=5,vid_names=['kid-football'])[0]
    vid = vid[3:,:,:400,150:150+400]
    B,F,H,W = vid.shape

    # -- run flow [raft] --
    from st_spix.flow_utils import run_raft,run_spynet
    fflow,bflow = run_raft(th.clip(255.*vid,0.,255.).type(th.uint8))

    # # -- save --
    # B,F,H,W = vid.shape
    # tv_utils.save_image(vid,root / "vid.png")
    # for t in range(B):
    #     tv_utils.save_image(vid[[t]],root / ("vid%d.png" % t))

    # -- prepare --
    vid_lab = st_spix.utils.vid_rgb2lab_th(vid.clone(),normz=False)
    vid_lab = rearrange(vid_lab,'b f h w -> b h w f').contiguous()
    fflow = rearrange(fflow,'b f h w -> b h w f').contiguous()

    # -- get spix --
    spix = stream_bass(vid_lab,flow=fflow,
                       niters=niters,niters_seg=niters_seg,
                       sp_size=sp_size,sigma2_app=sigma2_app,
                       sigma2_size=sigma2_size,
                       alpha_hastings=alpha_hastings,
                       potts=potts,sm_start=sm_start,rgb2lab=False)[0]

    # -- compute spix params --
    sims = get_sims(vid,spix,scale=10.)
    print("sims.shape: ",sims.shape)
    # sims[
    # exit()
    # sims[0,:,256,372] = 0.
    # sims[0,:,257,374] = 0.
    # sims[0,0,256,372] = 1.
    # sims[0,1,257,374] = 1.

    # print("sum: ",th.sum(sims[0,:,256,372]*sims[0,:,257,374]))
    # point0 = [256,372]
    # point1 = [257,374]

    # vid[...,256,372] = 1.
    vid = rearrange(vid,'b f h w -> b h w f').contiguous()

    # print(sims.shape)

    kwargs = {"kernel_size":K,"na_grid":"nat"}
    sna = SuperpixelAttention(3,**kwargs).to(vid.device)
    ws = sna.kernel_size
    attn,flows_k = sna.na_search(vid,fflow)
    # print(attn.shape)
    sim_attn = sna.get_sims_attn_map(sims)
    # print(sim_attn.shape)
    # print(th.where(sim_attn[0,0,0,:,12]==1))
    # print(th.where(sim_attn[0,0,0,:,13]==1))
    # print(sim_attn[0,0,0,256,372].reshape(5,5))
    # print(sim_attn[0,0,0,257,374].reshape(5,5))
    # exit()


    # print("-"*20)
    # print("sim_attn.shape: ",sim_attn.shape)
    # amax = sim_attn[0,0,0,50:-50,50:-50].std(-1).argmax()
    # # print(sim_attn[0,0,0,50:-50,50:-50].std(-1).argmax())
    # a = amax%300+10
    # b = amax//300+50
    # print(a,b)
    # center = (375,258)
    # a,b = 375,258

    # print(sim_attn[0,0,0,200,200])
    # print(sim_attn[0,0,0,100,200])
    # sim_attn = sim_attn / sim_attn.sum(-1,keepdim=True)
    sim_attn = sim_attn / (1e-5+sim_attn.max(-1,keepdim=True).values)

    # print(sim_attn[0,0,0,321:324,57:61])
    # exit()
    # sim_attn = th.softmax(-sim_attn,-1)
    # print("sa: ",sim_attn[0,0,0,200,200])
    # print("sim_attn.shape: ",sim_attn.shape)
    _gkernel = create_gaussian_kernel(K).reshape(K*K).to("cuda")
    gkernel = _gkernel.expand_as(sim_attn).to("cuda")
    # print("gkernel.shape: ",gkernel.shape)
    # print("-"*20)
    # print(sims[0,:,200,200])
    # print("sim_attn: ",sim_attn[0,0,0,a,b])
    # print("gkernel: ",gkernel[0,0,0,a,b])
    # skernel = th.softmax(sim_attn * gkernel,-1)
    skernel = sim_attn * gkernel
    # print("sim_attn.shape: ",sim_attn.shape)
    # print("gkernel.shape: ",gkernel.shape)
    # exit()
    # skernel = skernel / skernel.sum(-1,keepdim=True)
    # skernel = th.softmax(sim_attn,-1)
    # print("skernel: ",skernel[0,0,0,a,b])
    # # print(skernel)


    skernel = skernel[0,0,0]
    _gkernel = _gkernel.reshape(K,K)
    # skernel = skernel[198:202,198:202].reshape(-1,5,5)
    # skernel = skernel[321:324,57:61].reshape(-1,5,5)
    # skernel = skernel[a-2:a+2,b-2:b+2].reshape(-1,5,5)
    center = (250,75)
    # points = [[256,371],[256,372],[256,373],[256,374],[256,375]]
    N = 5
    points = [[center[1],center[0]-N//2+i] for i in range(N)]

    _skernel = []
    for point in points:
        print(sim_attn[0,0,0,point[0],point[1]])
        _skernel.append(skernel[point[0],point[1]].reshape(-1,K,K))
    _skernel.append(_gkernel[None,])
    # skernel = th.cat([skernel0,skernel1,_gkernel[None,]])
    skernel = th.cat(_skernel)[:,None]
    print(skernel.shape)
    skernel = resize(skernel,(18*7,18*7),InterpolationMode.NEAREST)
    skernel = skernel / skernel.max()
    mgrid = tv_utils.make_grid(skernel,nrow=len(skernel))
    # print(skernel.shape)
    # exit()
    mgrid = tv_utils.make_grid(skernel,nrow=len(skernel))
    mgrid = mgrid / mgrid.max()
    print(mgrid.min(),mgrid.max())
    print(mgrid.shape)
    tv_utils.save_image(mgrid,root / "skernel.png")
    for ix in range(len(skernel)):
        tv_utils.save_image(skernel[[ix]],root / ("skernel_%d.png"%ix))

    radius = 15
    zoom_factor = 8
    vid = rearrange(vid,'b h w f -> b f h w')
    # exit()
    # center = points[len(points)//2]
    # center = (250,75)
    circle = zoomed_circle(vid[0], center, radius, zoom_factor)
    tv_utils.save_image(circle[None,:],root / "circle.png")

    radius = 8
    ccircle = zoomed_circle(vid[0], center, radius, zoom_factor*2)
    tv_utils.save_image(ccircle[None,:],root / "ccircle.png")

    _center = (251,76)
    radius = 4
    cccircle = zoomed_circle(vid[0], _center, radius, zoom_factor*4)
    tv_utils.save_image(cccircle[None,:],root / "cccircle.png")


    # for hi in range(a-2,a+2):
    #     for wi in range(b-2,b+2):
    for (hi,wi) in points:
            center = (wi,hi)
            radius = 8
            zoom_factor = 15
            circle = zoomed_circle(vid[0], center, radius, zoom_factor, fill_center=True)
            sH,sW = circle.shape[-2:]
            # circle[...,sH//2,sW//2] = 1.
            tv_utils.save_image(circle[None,:],root / ("circle_%d_%d.png"%(hi,wi)))


if __name__ == "__main__":
    main()
