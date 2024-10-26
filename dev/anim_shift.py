
import os
import torch as th
import numpy as np
import pandas as pd
from pathlib import Path
import st_spix
from st_spix import flow_utils

import prop_cuda

from torchvision.transforms import InterpolationMode
from torchvision.transforms.functional import resize


import torchvision.io as tvio
import torchvision.io as iio
from einops import rearrange,repeat
from skimage.segmentation import mark_boundaries
import torchvision.utils as tv_utils
import torch.nn.functional as th_f

from st_spix.spix_utils import mark_spix_vid,fill_spix

from dev_basics.utils.metrics import compute_psnrs

import stnls

from dev_basics import flow as flow_pkg
from dev_basics.utils.timer import ExpTimer,TimeIt

from easydict import EasyDict as edict

from st_spix.prop import stream_bass,run_fwd_bwd,indepent_bass

import imageio

# -- colorwheel --
import matplotlib.pyplot as plt
import matplotlib as mpl
from matplotlib.colors import LinearSegmentedColormap, ListedColormap

def img_for_bass(img,device="cuda"):
    img= (th.clip(img,0.,1.)*255.).type(th.uint8)
    img = rearrange(img,'... f h w -> ... h w f').to(device)
    if img.ndim == 3: img = img[None,:]
    return img
def to_th(tensor):
    return th.from_numpy(tensor)
def swap_c(img):
    return rearrange(img,'... h w f -> ... f h w')
def viz_spix(spix,N):
    B,H,W = spix.shape
    viridis = mpl.colormaps['viridis'].resampled(N)
    spix = spix / N
    cols = viridis(spix.cpu().numpy())
    cols = rearrange(to_th(cols),'b h w f -> b f h w')
    return cols

def filenames2gif(fnames,nstart,nend,name,root):
    # -- stop at start and end --
    fnames_gif = []
    for ix in range(nstart):
        fnames_gif.append(fnames[0])
    for fname in fnames:
        fnames_gif.append(fname)
    for ix in range(nend):
        fnames_gif.append(fnames[-1])

    # -- save filenames --
    with imageio.get_writer(root/("%s.gif"%name), mode='I', duration=0.5) as writer:
        for fname in fnames_gif:
            img = imageio.imread(fname)
            if img.shape[-1] == 4:
                img = transparent_to_white(img)
            writer.append_data(img)

def transparent_to_white(img):
    # print(img.max())
    # -- make it white --
    args = np.where(img[:,:,-1] < 1.)
    for i in range(3):
        img[:,:,i][args] = 255
    img = img[:,:,:3]
    return img

def read_image(fn):
    return rearrange(iio.read_image(str(fn)),'c h w -> 1 c h w')/255.

def load_background(name):
    fn = Path("data/%s.png"%name)
    img = read_image(str(fn))
    # print("img.shape: ",img.shape)
    # img[0,-1][th.where(img[0,:-1].sum(1)>0)] = 1.
    return img

def read_elephants():
    # root = Path("/home/gauenk/Documents/packages/st_spix/data/figures/")
    root = Path("/home/gauenk/Documents/packages/st_spix/data/figures/baby_elephants/")
    # name = "elephant_frame%d.png"
    name = "img%d.png"
    vid = []
    # minH,minW = 704,1008
    minH,minW = 704,704
    for idx in range(4):
        if idx in [1,2]: continue
        fn = root / (name % idx)
        img = tvio.read_image(fn)/255.
        F,H,W = img.shape
        minH = H if H < minH else minH
        minW = W if W < minW else minW
        print(img.shape)
        vid.append(img)
    for idx in range(len(vid)):
        # vid[idx] = vid[idx][:,100:minH,300:minW]
        vid[idx] = vid[idx][:,100:800,700:1400]

    # for idx in range(len(vid)):
    #     vid[idx] = vid[idx][:,:minH,:minW]
    vid = th.stack(vid)
    # vid = resize(vid,(352,504)).to("cuda")
    # vid = resize(vid,(352,352)).to("cuda")
    # vid = resize(vid,(352//2,352//2),interpolation=InterpolationMode.BICUBIC).to("cuda")
    vid = resize(vid,(352,352),interpolation=InterpolationMode.BICUBIC).to("cuda")
    return vid

def get_transparent_background(H,W):
    xparent = read_image("data/figures/transparent_png.jpg")
    xparent = xparent.repeat(1,3,1,1)
    xparent = th.cat([xparent,xparent],-1)
    xparent = th.cat([xparent,xparent],-2).cuda()
    xparent = xparent[0,...,:H,:W]
    # print("img0.shape: ",img0.shape)
    # print("xparent.shape: ",xparent.shape)
    assert xparent.shape[-2:] == (H,W),"Must be equal."
    # exit()
    print("[v] xparent.shape: ",xparent.shape)
    return xparent

def animate_filling(img0,img1,flow,spix0,spix1,params0,root):

    # -- init --
    root_fill = root/"fill_frames"
    if not root_fill.exists():
        root_fill.mkdir(parents=True)
    # for fn in root_fill.iterdir(): os.remove(str(fn))

    # -- [pool flow] --
    mu_shape,spix_ids = params0.mu_shape[None,:],params0.ids
    fxn = st_spix.pool_flow_and_shift_mean
    flow_sp,flow_down,mu_shift = fxn(flow[None,:],mu_shape,spix0,spix_ids)

    # -- [shift_labels] --
    flow_down = flow_down.round().int()

    _spix,_flow = spix0[None,:],flow_down
    _spix_toshift = spix0[None,:,:,None].clone().float()
    _sizes = th.bincount(_spix.ravel())[None,:].int()
    ishift,counts,contrib = prop_cuda.shift_tensor(_spix_toshift,_spix,_flow)
    # print(contrib.shape,_sizes.shape)
    # th.cuda.synchronize()
    # print(ishift[0,100,100])
    # print(".")
    select = prop_cuda.shift_order(contrib,_sizes)
    spix_prop,_bdcounts = prop_cuda.shift_tensor_ordered(_spix_toshift,_spix,_flow,select)
    spix_prop = spix_prop[:,:,:,0].int()
    print(spix_prop.shape)


    # spix_prop,counts = prop_cuda.shift_labels(spix0[None,:],flow_down)
    # spix_prop[counts!=1] = -1
    # missing_mask = counts != 1
    spix_prop[counts==0] = -1
    overlaps = counts[0]>1
    holes = counts[0]==0
    missing_mask = counts == 0
    missing = th.where(missing_mask.ravel())[0][None,:].type(th.int)
    nspix0 = spix0.max().item()+1
    # print(th.logical_and(spix_prop>0,overlaps).sum())
    # print(th.logical_and(spix_prop>0,holes).sum())
    # exit()

    # -- non-transparent, transparent background --
    H,W = img0.shape[-2:]
    xparent = get_transparent_background(H,W)

    # -- [create animation; run "fill_missing" only a bit] --
    fnames_fill = []
    for ix in range(10):
        if ix < 5: inner_niters = ix
        else: inner_niters = 2*ix

        # -- [fill_missing] partially run --
        if ix > 0:
           spix_prop = prop_cuda.fill_missing(spix_prop,mu_shift,
                                              missing,inner_niters)
        holes_ix = th.logical_and(holes,spix_prop[0]==-1)
        overlaps_ix = th.logical_and(overlaps,spix_prop[0]==-1)
        border = prop_cuda.find_border(spix_prop)
        spix_ix = spix_prop[0]

        # -- fillin new information with frame t+1 info --
        anim_frame = th.zeros_like(img1)
        for i in range(3): # --> color channels

            # -- mark hole and overlaps --
            # anim_frame[i][th.where(overlaps_ix)] = i==0
            # anim_frame[i][th.where(holes_ix)] = i==2
            anim_frame[i][th.where(holes_ix)] = xparent[i][th.where(holes_ix)]

            # -- fill valid images with img1 --
            anim_frame[i][th.where(spix_ix!=-1)] = img1[i][th.where(spix_ix!=-1)]

            # -- along the boarder --
            anim_frame[i][th.where(border[0])] = i==1

        # -- save --
        fn_ix = str(root_fill/("frame_%d.png"%ix))
        fnames_fill.append(fn_ix)
        tv_utils.save_image(anim_frame,fn_ix)

    # -- filenames to gif --
    nstart = 3
    nend = 5
    name = "fill"
    filenames2gif(fnames_fill,nstart,nend,name,root)

def animate_shifting(img0,img1,flow,spix0,spix1,params0,root):

    # -- init --
    root_shift = root/"shift_frames"
    if not root_shift.exists():
        root_shift.mkdir(parents=True)
    # for fn in root_shift.iterdir(): os.remove(str(fn))
    marked = mark_spix_vid(th.stack([img0,img1]),th.stack([spix0,spix1]))
    marked0,marked1 = marked[0],marked[1]
    img0 = rearrange(img0,'f h w -> h w f').contiguous()
    marked0 = rearrange(marked0,'f h w -> h w f').contiguous().cuda()


    # -- get spix0 border --
    border0 = prop_cuda.find_border(spix0[None,:])

    # -- [pool flow] --
    # img1 = mark_spix_vid(img1[None,:].cpu(),spix1[None,:].cpu())[0].cuda()
    mu_shape,spix_ids = params0.mu_shape[None,:],params0.ids
    fxn = st_spix.pool_flow_and_shift_mean
    flow_sp,flow_down,means_shift = fxn(flow[None,:],mu_shape,spix0,spix_ids)

    # -- non-transparent, transparent background --
    H,W,F = img0.shape[-3:]
    # print("img0.shape: ",img0.shape)
    xparent = get_transparent_background(H,W)

    # -- only shift partially across each animation --
    nsteps = 10
    fnames_shift = []
    for ix in range(nsteps):

        # -- [shift_labels] (so we know the holes and overlaps) --
        _flow_down = ((ix/(nsteps-1.)) * flow_down.contiguous().round()).int()
        spix_prop,counts = prop_cuda.shift_labels(spix0[None,:],_flow_down)
        # print(counts)
        # exit()
        spix_prop[counts!=1] = -1
        overlaps = counts[0]>1
        holes = counts[0]==0
        missing_mask = counts != 1
        # print(counts.min(),counts.max(),missing_mask.shape)
        valid = th.where(missing_mask[0]==0)
        # print(valid[0].min(),valid[0].max())
        # print(valid[1].min(),valid[1].max())
        # print("-"*20)
        nspix0 = spix0.max().item()+1

        # -- shift superpixel boundary --
        # print("border0.shape: ",border0.shape)
        # bprop,counts = prop_cuda.shift_labels(border0,_flow_down)
        # print("bprop.shape: ",bprop.shape)

        # new function: using (spix,_flow_down) shift an input image
        # this is the "puzzle piece" function you've been looking for...
        # print("img0.shape: ",img0.shape)
        # print("spix0.shape: ",spix0.shape)
        # print("_flow_down.shape: ",_flow_down.shape)

        _sizes = th.bincount(spix0.ravel())[None,:].int()
        # print(valid[0].min(),valid[0].max())
        # print(valid[1].min(),valid[1].max())
        # print("-"*20)

        _img,_spix,_flow = marked0[None,:],spix0[None,:],_flow_down
        ishift,icounts,contrib = prop_cuda.shift_tensor(_img,_spix,_flow)
        select = prop_cuda.shift_order(contrib,_sizes)
        ishift,_bdcounts = prop_cuda.shift_tensor_ordered(_img,_spix,_flow,select)
        # print(_bdcounts.min(),_bdcounts.max())
        # print(ishift.shape)
        # print(th.all(_bdcounts<=1))
        ishift,icounts = ishift[0],icounts[0]
        # print("ishift.shape: ",ishift.shape)
        # print("icounts.shape: ",icounts.shape)
        # print("icounts.shape: ",icounts.shape)
        # for i in range(3): ishift[...,i][icounts!=1] = 0

        ioverlaps = icounts>1
        # iholes = icounts[0]==0
        iholes = icounts==0
        mshift,_,_ = prop_cuda.shift_tensor(marked0[None,:],spix0[None,:],_flow_down)
        mshift = mshift[0]
        # for i in range(3): mshift[...,i][icounts!=1] = 0
        # print("mshift.shape: ",mshift.shape)

        # -- create animation frame --
        anim_frame = th.zeros_like(img1)
        # print("overlaps.shape: ",overlaps.shape)
        # print("anim_frame.shape: ",anim_frame.shape)
        # print("missing_mask[0].shape: ",missing_mask[0].shape)
        # print("img1.shape: ",img1.shape)
        # exit()
        for i in range(3):
            anim_frame[i][th.where(overlaps)] = i==0
            # anim_frame[i][th.where(holes)] = i==2
            anim_frame[i][th.where(holes)] = xparent[i][th.where(iholes)]
            anim_frame[i][valid] = img1[i][valid]
            # spix_prop

        # -- create animation frame --
        anim_mframe = rearrange(ishift,'h w f -> f h w')
        # anim_mframe = rearrange(img0,'h w f -> f h w')
        for i in range(3):
            anim_mframe[i][th.where(holes)] = xparent[i][th.where(iholes)]


        # -- save --
        fn_ix = str(root_shift/("frame_%d.png"%ix))
        fnames_shift.append(fn_ix)
        tv_utils.save_image(anim_frame,fn_ix)

        # -- save --
        fn_ix = str(root_shift/("mframe_%d.png"%ix))
        fnames_shift.append(fn_ix)
        tv_utils.save_image(anim_mframe,fn_ix)

    # -- save --
    name,nstart,nend = "shift",3,5
    filenames2gif(fnames_shift,nstart,nend,name,root)

def get_anim_frames(root):
    root_fill = root/"fill_frames"
    root_shift = root/"shift_frames"
    N = len(list(root_fill.iterdir()))
    fnames_shift = []
    fnames_fill = []
    for ix in range(N):
        # fn_shift_ix = str(root_shift/("frame_%d.png"%ix))
        # fnames_shift.append(fn_shift_ix)
        fn_shift_ix = str(root_shift/("mframe_%d.png"%ix))
        fnames_shift.append(fn_shift_ix)
        fn_fill_ix = str(root_fill/("frame_%d.png"%ix))
        fnames_fill.append(fn_fill_ix)
    # fnames_shift = sorted(list(root_shift.iterdir()))
    # fnames_fill = sorted(list(root_fill.iterdir()))
    return fnames_shift,fnames_fill

def animate_group(root,img0,img1):

    # -- read root --
    root_grouped = root/"grouped_frames"
    if not root_grouped.exists():
        root_grouped.mkdir(parents=True)
    for fn in root_grouped.iterdir(): os.remove(str(fn))

    # -- read file names --
    fnames_shift,fnames_fill = get_anim_frames(root)

    # -- read groups --
    ix = 0
    fnames_group = []
    for fn_shift,fn_fill in zip(fnames_shift,fnames_fill):
        shift = read_image(fn_shift)[0]
        fill = read_image(fn_fill)[0]
        grid = th.stack([img0.cpu(),shift,fill,img1.cpu()])
        grid = tv_utils.make_grid(grid,nrow=4)
        fn_ix = str(root_grouped/("frame_%d.png"%ix))
        fnames_group.append(fn_ix)
        tv_utils.save_image(grid,fn_ix)
        ix += 1

    # -- save --
    nstart = 0
    nend = 0
    name = "grouped"
    filenames2gif(fnames_group,nstart,nend,name,root)

def main():

    # -- config --
    root = Path("./output/anim_shift")
    if not root.exists(): root.mkdir(parents=True)
    timer = ExpTimer()

    # -- config --
    niters = 30
    niters_seg = 4
    sm_start = 0
    sp_size = 20
    # alpha_hastings = 10.
    # alpha_hastings = 1.
    alpha_hastings = -5
    potts = 5.0
    pix_var = 0.01

    # sp_size = 20
    # niters,inner_niters = 1,25
    # # i_std,alpha,beta = 0.018,20.,100.
    # i_std,alpha,beta = 0.1,0.001,10.
    # cnt_eps = 0.#1e-8
    # niters_seg = 4

    # -- load images --
    # vid = st_spix.data.davis_example(isize=None,nframes=10)[0,:3,:,:480,:480]
    # vid = st_spix.data.davis_example(isize=None,nframes=10)[0,:3,:,:480,:480]
    vid = st_spix.data.davis_example(isize=None,nframes=10,vid_names=['kid-football'])
    vid = vid[0,4:6,:,50:50+320,200:480]
    # vid = vid + (25./255.)*th.randn_like(vid)
    # vid = read_elephants()
    T,F,H,W = vid.shape
    tv_utils.save_image(vid,root/"vid.png")

    # -- flow --
    from st_spix.flow_utils import run_raft,run_spynet
    fflow,bflow = run_raft(th.clip(255.*vid,0.,255.).type(th.uint8))

    # -- get streaming superpixels --
    vid_lab = st_spix.utils.vid_rgb2lab_th(vid.clone(),normz=False)
    vid_lab = rearrange(vid_lab,'b f h w -> b h w f').contiguous()
    fflow = rearrange(fflow,'b f h w -> b h w f').contiguous()
    outs = stream_bass(vid_lab,flow=fflow,
                       niters=niters,niters_seg=niters_seg,
                       sp_size=sp_size,pix_var=pix_var,
                       alpha_hastings=alpha_hastings,
                       potts=potts,sm_start=sm_start,rgb2lab=False)
    spix,params,children,missing,pmaps = outs

    # -- vizualize --
    marked = mark_spix_vid(vid,spix)
    tv_utils.save_image(marked,root / "marked.png")

    # -=-=-=-=-=-=-=-=-=-=-=-
    #
    #         ....
    #
    # -=-=-=-=-=-=-=-=-=-=-=-

    # -- iterative shift superpixels --
    animate_shifting(vid[0],vid[1],fflow[0],spix[0],spix[1],params[0],root)
    # -- iterative fill missing superpixels --
    animate_filling(vid[0],vid[1],fflow[0],spix[0],spix[1],params[0],root)
    # -- combine both animations --
    animate_group(root,vid[0],vid[1])
    return

    # -=-=-=-=-=-=-=-=-=-=-=-
    #
    #      Sequential
    #
    # -=-=-=-=-=-=-=-=-=-=-=-

    root_seq = root/"seq_frames"
    if not root_seq.exists():
        root_seq.mkdir(parents=True)
    for fn in root_seq.iterdir(): os.remove(str(fn))

    img0 = vid[None,0].cpu()/255.
    img1 = vid[None,1].cpu()/255.
    N = len(list(root_shift.iterdir()))
    fnames_shift = []
    fnames_fill = []
    for ix in range(N):
        fn_shift_ix = str(root_shift/("frame_%d.png"%ix))
        fnames_shift.append(fn_shift_ix)
        fn_fill_ix = str(root_fill/("frame_%d.png"%ix))
        fnames_fill.append(fn_fill_ix)

    fnames_seq = []
    ix = 0
    bkg_h = 50
    bkg_w = 20
    background_seq = load_background("background_seq_1")
    for fn_shift in fnames_shift:
        shift = read_image(fn_shift)
        fill = read_image(fnames_fill[0])
        grid = th.cat([img0,shift,fill,img1])
        grid = tv_utils.make_grid(grid,nrow=4)
        gH,gW =grid.shape[-2:]
        background_seq[:,:3,bkg_h:bkg_h+gH,bkg_w:bkg_w+gW] = grid
        background_seq[:,-1,bkg_h:bkg_h+gH,bkg_w:bkg_w+gW] = 1.
        # background_seq[:,-1] = 1.
        # print(grid.shape)
        # print(background_seq.shape)
        # exit()

        fn_ix = str(root_seq/("frame_%d.png"%ix))
        fnames_seq.append(fn_ix)
        # print(fn_ix)
        # tv_utils.save_image(grid,fn_ix)
        tv_utils.save_image(background_seq,fn_ix)
        ix+=1
        # exit()

    background_seq = load_background("background_seq_2")
    for fn_fill in fnames_fill:
        # print(fn_shift,fn_fill)
        shift = read_image(fnames_shift[-1])
        fill = read_image(fn_fill)
        # print(img0.shape,img1.shape,shift.shape,fill.shape)
        grid = th.cat([img0,shift,fill,img1])
        # print(grid.shape)
        grid = tv_utils.make_grid(grid,nrow=4)

        gH,gW =grid.shape[-2:]
        background_seq[:,:3,bkg_h:bkg_h+gH,bkg_w:bkg_w+gW] = grid
        background_seq[:,-1,bkg_h:bkg_h+gH,bkg_w:bkg_w+gW] = 1.
        # print(grid.shape)
        # print(background_seq.shape)
        # exit()
        fn_ix = str(root_seq/("frame_%d.png"%ix))
        fnames_seq.append(fn_ix)
        # tv_utils.save_image(grid,fn_ix)
        tv_utils.save_image(background_seq,fn_ix)
        ix += 1

    nstart = 0
    nend = 0
    name = "seq"
    filenames2gif(fnames_seq,nstart,nend,name,root)

if __name__ == "__main__":
    main()
