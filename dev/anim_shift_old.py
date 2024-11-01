
import os
import torch as th
import numpy as np
import pandas as pd
from pathlib import Path
import st_spix
from st_spix import flow_utils
import st_spix_cuda
import st_spix_prop_cuda
from st_spix import flow_utils as futils
import torchvision.io as iio
from einops import rearrange,repeat
from skimage.segmentation import mark_boundaries
import torchvision.utils as tv_utils
import torch.nn.functional as th_f

from st_spix.spix_utils import mark_spix_vid,fill_spix

import seaborn as sns
import matplotlib.pyplot as plt
from dev_basics.utils.metrics import compute_psnrs

import stnls

from dev_basics import flow as flow_pkg
from dev_basics.utils.timer import ExpTimer,TimeIt

from easydict import EasyDict as edict


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

def viz_marked_debug(img,debug,root):

    # -- debug info --
    print("debug.")
    print(th.mean(1.*(debug>=0),(-1,-2)))
    print(th.mean(1.*(debug<0),(-1,-2)))

    # -- marked regions from debug iterations --
    # _img1 = rearrange(img1,'b f h w -> b h w f')
    _img1 = img
    marks = []
    for _debug in debug:
        marked_ = mark_boundaries(_img1.cpu().numpy(),_debug[None,:].cpu().numpy())
        marked_ = to_th(swap_c(marked_))
        # if len(marks) > 0:
        #     marked_[0] = th.abs(marked_[0] - marks[0])
        #     marked_[0] = marked_[0]/marked_[0].abs().max()
        marks.append(marked_[0])
    marks = th.stack(marks)
    tv_utils.save_image(marks,root / "marked_debug.png")

# def mark_spix_vid(vid,spix):
#     marked = []
#     for ix,spix_t in enumerate(spix):
#         img = rearrange(vid[:,ix],'b f h w -> b h w f')
#         marked_t = mark_boundaries(img.cpu().numpy(),spix_t.cpu().numpy())
#         marked_t = to_th(swap_c(marked_t))
#         marked.append(marked_t)
#     marked = th.cat(marked)
#     return marked

def img4bass(img):
    img = rearrange(img,'... f h w -> ... h w f')
    img = img.contiguous()
    if img.ndim == 3: img = img[None,:]
    return img

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

def run_exp(cfg):

    # -- config --
    root = Path("./output/anim_shift")
    if not root.exists(): root.mkdir(parents=True)
    timer = ExpTimer()

    # -- config --
    npix_in_side = 40
    niters,inner_niters = 1,25
    # i_std,alpha,beta = 0.018,20.,100.
    i_std,alpha,beta = 0.1,0.001,10.
    cnt_eps = 0.#1e-8
    niters_seg = 4

    # -- load images --
    # vid = st_spix.data.davis_example(isize=None,nframes=10)[0,:3,:,:480,:480]
    # vid = st_spix.data.davis_example(isize=None,nframes=10)[0,:3,:,:480,:480]
    # vid = vid + (25./255.)*th.randn_like(vid)
    vid = read_elephant()
    vid = th.clip(255.*vid,0.,255.).type(th.uint8)
    T,F,H,W = vid.shape
    tv_utils.save_image(vid/255.,root/"vid.png")

    # -- bass --
    img0 = img4bass(vid[None,0])
    bass_fwd = prop_cuda.bass
    # spix0,means,cov,counts,ids = bass_fwd(img0,npix_in_side,i_std,alpha,beta)
    spix_t,params_t = prop_cuda.bass(img0,niters,niters_seg,sm_start,
                                     sp_size,pix_var,potts,alpha_hastings)
    # print(len(th.unique(spix0)))
    # print(spix0.min(),spix0.max())
    # exit()
    timer.sync_start("bass")
    # spix0,means,cov,counts,ids = bass_fwd(img0,npix_in_side,i_std,alpha,beta)
    spix_t,params_t = prop_cuda.bass(img0,niters,niters_seg,sm_start,
                                     sp_size,pix_var,potts,alpha_hastings)

    timer.sync_stop("bass")

    # -- run flow --
    timer.sync_start("flow")
    flows = flow_pkg.run(vid[None,:]/255.,sigma=0.0,ftype="cv2")
    # flow = flows.bflow[0,1][None,:]
    timer.sync_stop("flow")
    ids = ids.unsqueeze(1).expand(-1, means.size(-1)).long()[None,:]


    # -- iterations --
    spix_st = [spix0]
    spix_s = [spix0]

    # -- unpack --
    ix = 0
    img_curr = img4bass(vid[None,ix+1])
    flow_curr = flows.fflow[0,ix][None,:]
    # print(spix0.min(),spix0.max())

    # -- run --
    th.cuda.synchronize()
    timer.sync_start("st_iter_%d"%ix)
    spix_curr_st,cnts,debug,means,_ = stream_spix(img_curr,spix_st[-1],
                                                  flow_curr,means,cov,counts,
                                                  ids,niters,inner_niters,
                                                  npix_in_side,i_std,alpha,beta)
    timer.sync_stop("st_iter_%d"%ix)
    print(cnts.shape)
    print(cnts[0,23:28,16:20])
    print(cnts[0,63:68,16:20])
    print(cnts[0,73:80,16:20])
    th.save(cnts,"cnts.pth")



    spix_g = th.stack([spix0[0],spix_curr_st[0]])
    marked = mark_spix_vid(vid,spix_g)
    fill_spix(marked,spix_g,10)
    print(marked.shape)
    tv_utils.save_image(marked,root / "marked.png")

    # -=-=-=-=-=-=-=-=-=-=-=-
    #
    #   Animate the Filling
    #
    # -=-=-=-=-=-=-=-=-=-=-=-

    # # print(debug.shape)
    # # for ix in range(nsteps):
    # #     print(debug[ix].shape)
    root_fill = root/"fill_frames"
    if not root_fill.exists():
        root_fill.mkdir(parents=True)
    for fn in root_fill.iterdir(): os.remove(str(fn))
    fnames_fill = []
    spix = spix_st[-1]
    K = spix_st[-1].max().item()+1
    flow_sp,_ = st_spix.pool_flow_and_shift_mean(flow_curr,means.clone(),
                                                 spix_st[-1],ids)
    nsteps = 10
    negative_spix = True
    # while negative_spix:
    for ix in range(nsteps):
        # ix = nsteps
        # inner_niters = nsteps
        if ix < 5: inner_niters = ix
        else: inner_niters = 2*ix
        flow_sp,_ = st_spix.pool_flow_and_shift_mean(flow_curr,means.clone(),
                                                     spix_st[-1],ids)
        scatter,cnts = st_spix.scatter.run(vid[None,0].contiguous()/255.,flow_sp)
        cmp1 = scatter.clone()
        img_curr = img4bass(vid[None,1])
        spix1,_,debug,_,border = prop_seg(img_curr,spix,flow_curr,
                                          means,cov,counts,ids,niters,
                                          inner_niters,npix_in_side,
                                          i_std,alpha,beta)

        # -- fillin new information with frame t+1 info --
        scatter = th.clamp(scatter,0,1.)
        img1 = vid[None,1]/255.
        for i in range(3):
            # -- still invalid --
            bool0 = th.logical_and(cnts>(1+cnt_eps),spix1==-1)
            bool1 = th.logical_and(cnts<(1-cnt_eps),spix1==-1)
            scatter[:,i][th.where(bool0)] = i==0
            scatter[:,i][th.where(bool1)] = i==2

            # -- valid filled with img1 --
            scatter[:,i][th.where(spix1!=-1)] = img1[:,i][th.where(spix1!=-1)]

            # -- 'filled-in' region --
            bool_f = th.logical_and(cnts>(1+cnt_eps),spix1!=-1)
            bool_f = th.logical_or(bool_f,th.logical_and(cnts<(1-cnt_eps),spix1!=-1))
            scatter[:,i][th.where(bool_f)] = img1[:,i][th.where(bool_f)]

            # -- along the boarder --
            scatter[:,i][th.where(border)] = i==1

        # -- compute diff to reference frame --
        delta = th.mean((img1 - scatter)**2,1,keepdim=True).repeat(1,3,1,1)
        # delta = delta / (1e-8+delta.max())
        # print(spix1.shape,delta.shape,scatter.shape)
        # print(delta.max(),delta.min())
        args = th.where(spix1==-1)
        # for i in range(3):
        #     delta[:,i][args] = 0.
        # # print(delta.max(),delta.min())
        # delta = delta / (1e-8+delta.max())
        for i in range(3):
            delta[:,i][args] = scatter[:,i][args]

        # -- create frame --
        # print(scatter.shape,delta.shape)
        frame_ix = scatter
        # frame_ix = th.cat([scatter,delta])
        # frame_ix = tv_utils.make_grid(frame_ix,nrow=2)
        # frame_ix = scatter

        # -- save --
        fn_ix = str(root_fill/("frame_%d.png"%ix))
        fnames_fill.append(fn_ix)
        tv_utils.save_image(frame_ix,fn_ix)

        # -- update --
        negative_spix = th.any(spix1==-1)
        nsteps = nsteps + 1
        if nsteps > 30: break


    # -- filenames to gif --
    nstart = 3
    nend = 5
    name = "fill"
    filenames2gif(fnames_fill,nstart,nend,name,root)


    # -=-=-=-=-=-=-=-=-=-=-=-
    #
    #   Animate the Shifting
    #
    # -=-=-=-=-=-=-=-=-=-=-=-

    # # # nsteps = 10 # from above
    root_shift = root/"shift_frames"
    if not root_shift.exists():
        root_shift.mkdir(parents=True)
    # for fn in root_shift.iterdir(): os.remove(str(fn))
    # fnames_shift = []
    # K = spix_st[-1].max().item()+1
    # flow_sp,_ = st_spix.pool_flow_and_shift_mean(flow_curr,means.clone(),
    #                                              spix_st[-1],ids)
    # for ix in range(nsteps):
    #     timer.sync_start("shift_img_%d"%ix)
    #     _flow_sp = (ix/(nsteps-1.)) * flow_sp.contiguous()
    #     scatter,cnts = st_spix.scatter.run(vid[:,0].contiguous()/255.,_flow_sp)
    #     timer.sync_stop("shift_img_%d"%ix)
    #     scatter = th.clamp(scatter,0,1.)
    #     for i in range(3):
    #         scatter[:,i][th.where(cnts>(1.+cnt_eps))] = i==0
    #         scatter[:,i][th.where(cnts<(1.-cnt_eps))] = i==2
    #     fn_ix = str(root_shift/("frame_%d.png"%ix))
    #     fnames_shift.append(fn_ix)
    #     tv_utils.save_image(scatter,fn_ix)

    # name = "shift"
    # filenames2gif(fnames_shift,nstart,nend,name,root)


    # # -=-=-=-=-=-=-=-=-=-=-=-
    # #
    # #        Grouped
    # #
    # # -=-=-=-=-=-=-=-=-=-=-=-

    # # -- all together --
    # img0 = vid[:,0].cpu()/255.
    # img1 = vid[:,1].cpu()/255.
    N = len(list(root_shift.iterdir()))
    fnames_shift = []
    fnames_fill = []
    for ix in range(N):
        fn_shift_ix = str(root_shift/("frame_%d.png"%ix))
        fnames_shift.append(fn_shift_ix)
        fn_fill_ix = str(root_fill/("frame_%d.png"%ix))
        fnames_fill.append(fn_fill_ix)
    # fnames_shift = sorted(list(root_shift.iterdir()))
    # fnames_fill = sorted(list(root_fill.iterdir()))

    # # print(fnames_shift)
    # # print(fnames_fill)
    # root_grouped = root/"grouped_frames"
    # if not root_grouped.exists():
    #     root_grouped.mkdir(parents=True)
    # for fn in root_grouped.iterdir(): os.remove(str(fn))
    # fnames_group = []
    # ix = 0
    # for fn_shift,fn_fill in zip(fnames_shift,fnames_fill):
    #     # print(fn_shift,fn_fill)
    #     shift = read_image(fn_shift)
    #     fill = read_image(fn_fill)
    #     # print(img0.shape,img1.shape,shift.shape,fill.shape)
    #     grid = th.cat([img0,shift,fill,img1])
    #     # print(grid.shape)
    #     grid = tv_utils.make_grid(grid,nrow=4)
    #     fn_ix = str(root_grouped/("frame_%d.png"%ix))
    #     fnames_group.append(fn_ix)
    #     tv_utils.save_image(grid,fn_ix)
    #     ix += 1

    # nstart = 0
    # nend = 0
    # name = "grouped"
    # filenames2gif(fnames_group,nstart,nend,name,root)

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


    # # -- run --
    # timer.sync_start("s_iter_%d"%ix)
    # spix_curr_s,_,_,_,_ = bass_fwd(img_curr,npix_in_side,i_std,alpha,beta)
    # # spix_curr_st,debug = prop_seg(img_curr,spix_st[-1],flow_curr,means,cov,counts,
    # #                               niters,inner_niters,npix_in_side,i_std,alpha,beta)
    # timer.sync_stop("s_iter_%d"%ix)

    # -- debug --
    # viz_marked_debug(img_curr,debug,root)

    # -- stop condition --
    # if spix_curr_st.min().item() < 0: break
    # spix_curr_st = spix_curr_st - spix_curr_st.min()

    # -- read timer --
    print(timer)

    # # -- view superpixels --
    # marked = mark_spix_vid(vid,spix_st)
    # tv_utils.save_image(marked,root / "marked_spacetime.png")
    # marked = mark_spix_vid(vid,spix_s)
    # tv_utils.save_image(marked,root / "marked_space.png")


def main():

    print("PID: ",os.getpid())
    cfg = edict()
    cfg.name = "a"
    run_exp(cfg)

if __name__ == "__main__":
    main()
