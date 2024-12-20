
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

def read_worm1():
    return read_segtrack_video("worm_1")

def read_bass_examples():
    from PIL import Image
    root = Path("/home/gauenk/Documents/packages/BASS_check/images/")
    vid = []
    for fn in root.iterdir():
        fname = fn.resolve()
        if fname.stem == "302003": continue
        fname = str(fname)
        img = np.array(Image.open(fname).convert("RGB"))/255.
        vid.append(img)
    vid = np.stack(vid)
    vid = th.from_numpy(vid).to("cuda")
    # print(vid.shape)
    vid = rearrange(vid,'t h w c -> t c h w').float()
    print("vid.shape: ",vid.shape)
    vid = vid[:,:,:320,:480]
    return vid


def read_segtrack_video(vname):
    from PIL import Image
    root = Path("/home/gauenk/Documents/packages/superpixel-benchmark/docker/in/")
    root = root /"SegTrackv2/PNGImages/" /vname
    nframes = len([f for f in root.iterdir() if str(f).endswith(".png")])
    vid = []
    for frame_ix in range(nframes):
        fname = root/("%05d.png" % (frame_ix+1))
        img = np.array(Image.open(fname).convert("RGB"))/255.
        vid.append(img)
    vid = np.stack(vid)
    vid = th.from_numpy(vid).to("cuda")
    # print(vid.shape)
    vid = rearrange(vid,'t h w c -> t c h w').float()
    return vid


def main():

    # -- get root --
    print("PID: ",os.getpid())
    root = Path("./output/run_prop/")
    if not root.exists(): root.mkdir()

    # kwargs = {"use_bass_prop":True,"niters":15,"niters_seg":4,
    #           "sp_size":15,"pix_var":0.01,"alpha_hastings":20.,
    #           "potts":1.,"sm_start":0,"rgb2lab":False}

    # -- config --
    niters_seg = 4
    sm_start = 0
    # sp_size = 20
    # sp_size = 30
    sp_size = 30
    # niters = min(sp_size*10,150)
    niters = sp_size
    # niters = sp_size # a fun default from the authors
    # alpha_hastings = 1.0
    # alpha_hastings = 0.1
    # alpha_hastings = 200.
    # alpha_hastings = 5.
    # alpha_hastings = 2.
    # potts = 10.0
    # potts = 5.0
    # alpha_hastings = 15.25
    # alpha_hastings = 23.
    # alpha_hastings = -10000.
    # alpha_hastings = 0.
    alpha_hastings = -0.693
    # potts = 1.0
    potts = 30.0
    # potts = 0.01
    # potts = 0.
    # pix_var = 0.001
    # sigma2_app = 0.02
    # sigma2_app = .01
    sigma2_app = 8e-5
    # sigma2_app = 0.001
    # sigma2_size = 0.02
    # sigma2_size = 5e-1
    # sigma2_size = 25.
    # sigma2_size = .1
    # sigma2_size = .001
    sigma2_size = 1.
    # sigma2_size = 500.
    # sigma2_size = 10.1
    # sigma2_app = 0.08
    # sigma2_app = 0.01

    # -- per-image bass --
    # potts = 10.
    # alpha_hastings = 1.
    # sigma2_app = 0.1

    # -- read img/flow --
    # vid = st_spix.data.davis_example(isize=None,nframes=-1,vid_names=['tennis'])
    # vid = st_spix.data.davis_example(isize="320_320",nframes=30,vid_names=['tennis'])
    vid = st_spix.data.davis_example(isize=None,nframes=10,vid_names=['tennis'])
    # vid = st_spix.data.davis_example(isize=None,nframes=30,vid_names=['tennis'])
    # vid = st_spix.data.davis_example(isize=None,nframes=10,vid_names=['baseball'])
    # vid = read_bass_examples()
    # vid = read_worm1()
    size = 256
    # vid = vid[0,5:7,:,50:50+size,300:300+size]
    # vid = vid[0,0:8,:,50:50+320,:480]
    # vid = vid[0,0:3,:,50:50+320,:480]
    vid = vid[0,0:3,:,20:20+320,:480]
    # vid = vid[0,0:30,:,50:50+320,:480]
    # vid = vid[0,0:30,:,50:50+320,:240]
    # vid = vid[0,0:,:,150+50:150+50+320,:240]
    # vid = vid[0,0:2,...,:800]
    # vid = vid[0,0:,...,:800]
    # vid = vid[0]

    # vid = vid[0,0:3,...,:800]
    # vid = vid[0,0:3,...,:480]
    # vid = vid[0,0:6,...,:800]
    # vid = vid[0,0:2,...,:400,:400]
    # vid = vid[0,0:10,...,:400,:400]
    # vid = vid[0,0:20,...,:480,:800]
    # vid = vid[0,2:6,...,:400,:400]

    # vid = vid[0,0:30,...,:800]
    # vid = vid[0,0:3,...,:800]
    # vid = vid[0,0:15,...,:800]
    # vid = vid[0,0:6,...,:400]
    # vid = vid[0,0:30,...,:400]
    # vid = vid[0,0:3,...,:800]
    # vid = vid[0,1:3+1,...,:800]
    # vid = th.cat([vid[:2],]*3)
    # vid = vid[0,0:5,...,:400]
    # vid = th.stack([th.randn_like(vid[0])*0.05+vid[0],]*5)

    # vid = vid[0,0:,:,50:50+320,:240]
    # vid = vid[0,0:,:,50:50+320,:800]
    # vid = vid[0,0:,:30,50:50+320,:800]
    # vid = vid[0,0:2,:,50:50+320,:240]
    # vid = vid[0,0:6,:,50:50+320,:240]
    # vid = vid[0,0:3,:,50:50+320,:240]

    # vid = vid[0,0:2,:,50:50+320,:240]
    # vid = vid.repeat(5,1,1,1)

    # vid = vid[0,0:2,:,50:50+320,480-320:480]

    # vid = vid[0,2:5,:,50:50+size,200:200+size]
    # vid = read_otter().to("cuda")
    # vid = read_mnist().to("cuda")
    # vid = resize(vid,(128,128))
    # vid = read_figure_image()
    # vid = read_figure_image()
    # vid = read_elephants()


    vid_og = vid.clone()
    # print(vid.shape)
    # vid = vid[:29,:,:320,:240].clone()

    # -- run flow [raft] --
    from st_spix.flow_utils import run_raft,run_spynet
    fflow,bflow = run_raft(th.clip(255.*vid,0.,255.).type(th.uint8))
    # fflow = th.clip(fflow,-10,10)
    # fflow[...] = 0.
    # fflow[...] = th.randn_like(fflow)*0.1

    # fflow,bflow = run_spynet(vid)
    B,F,H,W = vid.shape
    # fflow = th.zeros((B,2,H,W),device="cuda")
    # bflow = fflow.clone()

    # -- resize again --
    # vid = resize(vid,(56,56))
    # fflow = resize(fflow,(64,64))/2. # reduced scale by 2
    # size = 128
    # vid = resize(vid,(size,size))
    # fflow = resize(fflow,(size,size))/(128./size) # reduced scale by X

    # -- save --
    B,F,H,W = vid.shape
    tv_utils.save_image(vid,root / "vid.png")
    for t in range(B):
        tv_utils.save_image(vid[[t]],root / ("vid%d.png" % t))

    # vid = vid[:29,:,:320,:240].clone()
    # -- propogate --
    # outs = stream_bass(vid,flow=fflow,
    #                    niters=niters,niters_seg=niters_seg,
    #                    sp_size=sp_size,sigma2_app=sigma2_app,
    #                    alpha_hastings=alpha_hastings,
    #                    potts=potts,sm_start=sm_start)
    # spix,params,children,missing,pmaps = outs
    # print("[og] 8: ",params[0].mu_app[8])

    # -- debug --
    # data = th.randn(5,3,100,100)
    # mask = th.randn(5,3,100,100)>0
    # data1 = th.randn(5,3,100,100)
    # mask1 = th.randn(5,3,100,100)>0
    # data1.masked_fill_(mask1,1)
    # data.masked_fill_(mask, -1)

    # import time
    # th.cuda.synchronize()
    # start_time = time.perf_counter()
    # # Perform the operation
    # data.masked_fill_(mask, -1)

    # # Synchronize after operation to ensure all GPU tasks are done
    # # if th.cuda.is_available():
    # th.cuda.synchronize()

    # # End the timer
    # end_time = time.perf_counter()

    # # Calculate elapsed time
    # elapsed_time = end_time - start_time

    # # Output the elapsed time
    # print(f"Elapsed time: {elapsed_time:.6f} seconds")


    # # th.cuda.synchronize()

    # timer = ExpTimer()
    # memer = GpuMemer()

    # with TimeIt(timer,"main"):
    #     data.masked_fill_(mask, -1)
    #     # print(f"Elapsed time: {elapsed_time:.6f} seconds")

    # print(timer)
    # print(memer)


    # -- rgb 2 lab --
    # if rgb2lab:
    #     vid_lab = st_spix.utils.vid_rgb2lab_th(vid.clone(),normz=False)
    # else:
    #     vid_lab = vid
    vid_lab = st_spix.utils.vid_rgb2lab_th(vid.clone(),normz=False)
    # vid_lab = vid_lab+1.
    # vid_lab = (vid_lab + 1.)/2.
    # vid_lab = (vid_lab + 1.)/10.
    # print("vid_lab.min(),vid_lab.max(): ",vid_lab.min(),vid_lab.max())
    # exit()
    # vid_lab = vid.clone()
    vid_lab = rearrange(vid_lab,'b f h w -> b h w f').contiguous()
    fflow = rearrange(fflow,'b f h w -> b h w f').contiguous()

    print(".")
    th.cuda.synchronize()
    while True:
        timer = ExpTimer()
        memer = GpuMemer()
        with MemIt(memer,"main"):
            with TimeIt(timer,"main"):

                outs = stream_bass(vid_lab,flow=fflow,
                                   niters=niters,niters_seg=niters_seg,
                                   sp_size=sp_size,sigma2_app=sigma2_app,
                                   sigma2_size=sigma2_size,
                                   alpha_hastings=alpha_hastings,
                                   potts=potts,sm_start=sm_start,rgb2lab=False)
                spix,params,pre_fill,post_fill = outs

                # spix = indepent_bass(vid_lab,niters=niters,niters_seg=niters_seg,
                #                      sp_size=sp_size,sigma2_app=sigma2_app,
                #                      alpha_hastings=alpha_hastings,
                #                      potts=potts,sm_start=sm_start,rgb2lab=False)

                th.cuda.synchronize()
        print(timer)
        print(memer)
        break
    # exit()


    # -- info --
    print(spix[0][310:,0:10])
    print("spix.shape: ",spix.shape)
    # exit()

    # -- ... --
    for t in range(len(vid)):
        print("Num Uniq: ",len(th.unique(spix[t])))

    # print(spix[-1,150:170,20:40])
    # print(spix[-2,30:50,0:20])
    # print(spix[-1,30:50,0:20])
    print(spix[1,:10,40:60])
    # print(spix[2,:10,40:60])
    # spix[th.where(spix==128)] = 0
    # exit()

    # -- view --
    marked = mark_spix_vid(vid,spix,mode="subpixel")
    # save_zoom_vid(marked,[205,180,260,225],root/"nose.png")
    # save_zoom_vid(marked,[220,100,275,145],root/"eye.png")
    # cmarked = color_regions(marked,[[205,180,260,225],[220,100,275,145]])
    # print("cmarked.shape: ",cmarked.shape)
    # cmarked = cmarked[:,:,:250,35:380]
    # tv_utils.save_image(cmarked,root / "cmarked_fill.png")

    marked_m = marked.clone()
    # marked_m[1:] = (1-1.*missing.cpu())*marked_m[1:]
    # a,b,c = spix[-2,40,0].item(),spix[-1,40,0].item(),spix[-1,30,0].item()
    # a,b,c = spix[-2,40,0].item(),spix[-1,40,0].item(),spix[-1,30,0].item()
    # a,b,c = spix[1,0,55].item(),spix[2,8,55].item(),spix[2,15,55].item()
    a = 0
    b = 0
    c = 0
    a = spix[0,125,25].item()
    print(a,b,c)
    marked_c = color_spix(marked.clone(),spix,a,cidx=1)
    # marked_c = color_spix(marked_c,spix,b,cidx=0)
    marked_c = color_spix(marked_c,spix,b,cidx=0)
    marked_c = color_spix(marked_c,spix,c,cidx=2)
    # marked_c = color_spix(marked_c,spix,spix[0].max().item()+1,cidx=2)
    # marked_c = color_spix(marked_c,spix,b,cidx=2)
    # marked_c = color_spix(marked.clone(),spix,125,cidx=0)

    # marked_c = color_spix(marked_c,spix,125,cidx=0)
    # marked_c = color_spix(marked_c,spix,120,cidx=0)
    # marked_c = color_spix(marked_c,spix,44,cidx=2)
    # marked_c = color_spix(marked_c,spix,108,cidx=1)

    # marked_c = color_spix(marked_c,spix,9,cidx=1)
    # # marked_c = color_spix(marked_c,spix,10,cidx=0)
    # # marked_c = color_spix(marked_c,spix,11,cidx=1)
    # marked_c = color_spix(marked_c,spix,12,cidx=2)
    # tv_utils.save_image(vid[[0],...,70:150,80:180],root / "f0.png")

    # -- save --
    print("saving images.")
    # save_spix_parts(root/"otters",vid[0,...,70:150,80:180],spix[0,...,70:150,80:180])
    save_spix_img(root,vid[0],spix[0])
    # save_spix_parts(root/"elephant0",vid[0],spix[0])
    # save_spix_parts(root/"elephant1",vid[1],spix[1])
    # viz_seg = draw_spix_vid(vid,spix)
    futils.viz_flow_quiver(root/"flow.png",fflow[[0]],step=4)
    tv_utils.save_image(marked,root / "marked_fill.png")
    # tv_utils.save_image(marked[[0]],root / "marked0.png")
    tv_utils.save_image(marked[[0],...,70:150,80:180],root / "marked0.png")
    tv_utils.save_image(marked_m,root / "marked_missing.png")
    tv_utils.save_image(marked_c,root / "marked_colored.png")
    # tv_utils.save_image(viz_seg,root / "viz_seg.png")

    marked_pre_fill = mark_spix_vid(vid[1:],pre_fill)
    # marked_pre_fill = pre_fill # actually the flow_sp image
    marked_post_fill = mark_spix_vid(vid[1:],post_fill)
    tv_utils.save_image(marked_pre_fill,root / "marked_pre_fill.png")
    tv_utils.save_image(marked_post_fill,root / "marked_post_fill.png")
    return


    # -- elephant zoom --
    tv_utils.save_image(rzs(marked[[0],...,70-10:140-10,110+10:180+10]),
                        root / "eleph0.png")
    tv_utils.save_image(rzs(marked[[1],...,70+10:140+10,110+15:180+15]),
                        root / "eleph1.png")

    # -- vizualize the lab values with the means --
    vid_lab = st_spix.utils.vid_rgb2lab(vid,normz=False)
    print([(vid_lab[:,i].min().item(),vid_lab[:,i].max().item()) for i in range(3)])
    # inspect_means(vid_lab,spix,params,sp_size)

    # -- copy before refinement --
    spix_og = spix.clone()
    # params_og = [st_spix.copy_spix_params(p) for p in params]
    border_og = prop_cuda.find_border(spix_og)

    # -- view --
    vid_lab = vid_lab - vid_lab.min()
    vid_lab = vid_lab / vid_lab.max()
    print(vid.shape,vid.max(),vid.min())
    print(vid_lab.shape,vid_lab.max(),vid_lab.min())
    marked = mark_spix_vid(vid_lab,spix)
    marked_m = marked.clone()
    # marked_m[1:] = (1-1.*missing.cpu())*marked_m[1:]
    marked_c = color_spix(marked.clone(),spix,2,cidx=0)
    marked_c = color_spix(marked_c,spix,3,cidx=2)

    # -- save --
    print("saving images.")
    tv_utils.save_image(marked,root / "lab_marked_fill.png")
    tv_utils.save_image(marked_m,root / "lab_marked_missing.png")
    tv_utils.save_image(marked_c,root / "lab_marked_colored.png")
    # tv_utils.save_image(viz_seg,root / "viz_seg.png")

    exit()
    # -- run fwd/bwd --
    niters_ref = 15
    niters_fwd_bwd = 5
    sigma2_app = 0.005
    potts = 1.
    # print("8: ",params[0].mu_app[8],params[0].counts[8])
    spix,params = run_fwd_bwd(vid_og,spix,params,sp_size,sigma2_app,
                              potts,niters_fwd_bwd,niters_ref)
    # print("8:" ,params[0].mu_app[8],params[0].counts[8])
    border_b = prop_cuda.find_border(spix)
    spix,params = run_fwd_bwd(vid_og,spix,params,sp_size,sigma2_app,
                              potts,niters_fwd_bwd,niters_ref)
    border_c = prop_cuda.find_border(spix)

    # -- view --
    marked = mark_spix_vid(vid,spix)
    marked_m = marked.clone()
    # marked_m[1:] = (1-1.*missing.cpu())*marked_m[1:]
    marked_c = color_spix(marked.clone(),spix,115,cidx=0)
    marked_c = color_spix(marked_c,spix,126,cidx=2)

    # -- save --
    print("saving images.")
    # viz_seg = draw_spix_vid(vid,spix)
    tv_utils.save_image(marked,root / "fwdbwd_marked_fill.png")
    tv_utils.save_image(marked_m,root / "fwdbwd_marked_missing.png")
    tv_utils.save_image(marked_c,root / "fwdbwd_marked_colored.png")
    # tv_utils.save_image(viz_seg,root / "viz_seg.png")

    # -- elephant zoom --
    tv_utils.save_image(rzs(marked[[0],...,70-10:140-10,110+10:180+10]),
                        root / "eleph0.png")
    tv_utils.save_image(rzs(marked[[1],...,70+10:140+10,110+15:180+15]),
                        root / "eleph1.png")
    # tv_utils.save_image(marked[[0],...,70-10:140-10,110+10:180+10],root / "eleph0.png")
    # tv_utils.save_image(marked[[1],...,70+10:140+10,110+15:180+15],root / "eleph1.png")


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




if __name__ == "__main__":
    main()
