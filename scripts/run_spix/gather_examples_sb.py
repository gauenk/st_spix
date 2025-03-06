"""

    Create formatted results from superpixel-benchmark results

"""


import os
import tqdm
import shutil
import numpy as np
import pandas as pd
from pathlib import Path
from collections import OrderedDict as odict

import torch as th
import torchvision.io as tvio
import torchvision.utils as tv_utils

import bist_cuda

from einops import rearrange
from st_spix.spix_utils import mark_spix_vid,img4bass

def read_spix(root,frames):
    spix = []
    for frame in frames:
        fn = root / ("%05d.csv"%frame)
        spix_f = np.asarray(pd.read_csv(fn,header=None))
        # spix_f = rearrange(spix_f,'w h -> h w')
        spix.append(th.from_numpy(spix_f))
    spix = th.stack(spix)
    return spix

def get_kstr(method,k):
    if method in ["bass","bist"]:
        kstr = "param0"
    elif method in ["etps","seeds","ers"]:
        # kstr = "sp%d"%(k//100)
        kstr = "%02dsp"%(k//100)
    else:
        kstr = "sp%d"%k
        # if method in ["mbass","st_spix"]:
        #     kstr = "sp%d"%400
    return kstr

def get_base(method):
    if method in ["etps","seeds","ers",'tsp']:
        base = Path("/home/gauenk/Documents/packages/superpixel-benchmark/docker/out")
        return base
    else:
        base = Path("/home/gauenk/Documents/packages/st_spix_refactor/result/")
        return base

def read_all_spix(dname,vid_name,frames,k,methods):

    # -- run script --
    # dname = "segtrackerv2"
    # base = Path("/home/gauenk/Documents/packages/superpixel-benchmark/docker/")
    # vid_names = get_video_names(dname)
    # methods = ["etps","seeds","ers","mbass","st_spix","tsp"]
    spix = {}
    for method in methods:
        base = get_base(method)
        kstr = get_kstr(method,k)
        sp_dir = base/Path("%s/%s/%s/%s/" % (dname,method,kstr,vid_name))
        if method == "bist":
            print(method,sp_dir)
        #     exit()
        spix_m = read_spix(sp_dir,frames)
        spix[method] = spix_m.to("cuda")
    return spix

def read_video(root,vid_name,frames):

    # -- read all image files --
    root = Path(root)/vid_name
    files = []
    for fn in root.iterdir():
        suffix = fn.suffix
        if not(suffix in [".jpg",".jpeg",".png"]):
            # print("skipping: ",fn)
            continue
        files.append(fn.name)

    # -- sort by number --
    files = sorted(files, key=lambda x: int(os.path.splitext(x)[0]))

    # -- read to video --
    vid = []
    for fn in files:
        if not(int((root/fn).stem) in frames): continue
        img = tvio.read_image(str(root/fn))/255.
        vid.append(img)
    vid = th.stack(vid).cuda()
    return vid

def read_anno(droot,vname,frames):
    # -- read to anno --
    anno = []
    for fidx in frames:
        fname = Path(droot).parents[0]/"GroundTruth"/vname/("%05d.png"%fidx)
        if not fname.exists():
            fname = Path(droot).parents[0]/"GroundTruth"/vname/"1"/("%05d.png"%fidx)
        img = 1.0*(tvio.read_image(str(fname))[0]>0)
        anno.append(img)
    anno = th.stack(anno).cuda()
    return anno


def compare_single_video():
    # -- execute summary for each sequence --
    root = Path("output/run_spix/gather_examples_sb/")
    k = 800
    # data_root = "/home/gauenk/Documents/packages/LIBSVXv4.0/Data/SegTrackv2/PNGImages/"
    if not root.exists(): root.mkdir(parents=True)
    # dname = "segtrackerv2"
    # vname = "penguin" # a good limitation of tracking
    # vname = "monkey"
    dname = "davis"
    # vname = "blackswan"
    # frames = [1,2,3,4,5,6,7]
    # frames = [20,30,40]
    # frames = [1,10,20]

    # -- idk; great example --
    # vname = "mbike-trick" # great!
    # frames = [1,5,10]

    # -- struggle in dynamic scenes --
    # vname = "kite-surf"
    # frames = [1,5,10]

    # -- both immediately fail for temporal coherence --
    # vname = "parkour"
    # frames = [30,35,40]

    # -- bist does slightly better than TSP --
    vname = "parkour"
    frames = [30,31,32]

    # -- bist does slightly better than TSP --
    vname = "bike-packing"
    frames = [1,5,10]

    # -- read --
    data_root = get_data_root(dname)
    vid = read_video(data_root,vname,frames)#[...,20:-20,100:-100]
    anno = read_anno(data_root,vname,frames)#[...,20:-20,100:-100]
    # methods = ["tsp","st_spix"]
    methods = ["tsp","bist","bass"]
    spix = read_all_spix(dname,vname,frames,k,methods)
    spix_tsp = spix['tsp'].contiguous().int()
    spix_bist = spix['bist'].contiguous().int()
    spix_bass = spix['bass'].contiguous().int()
    trip = format_triplet(vid,anno,spix_tsp,spix_bist,spix_bass)
    vid,marked_bist,marked_tsp,marked_bass = trip
    mgrid = th.cat([vid,marked_bist,marked_tsp,marked_bass],0).cpu()

    # -- save --
    mgrid = tv_utils.make_grid(mgrid,nrow=len(vid))
    fname = root/"compare_single_video.png"
    print(f"Saving to {str(fname)}")
    tv_utils.save_image(mgrid,fname)

def save_triplet_collection():
    entry0 = {"save_name":"temporal_consistency",
              "dname":"davis",
              "vname":"bike-packing",
              "frames":[1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20]}
    entry1 = {"save_name":"bike_long",
              "dname":"davis",
              "vname":"bike-packing",
              "frames":np.arange(20)+1,
              "alpha":0.5}
    entry2 = {"save_name":"mbike-trick_seq",
              "dname":"davis",
              "vname":"mbike-trick",
              "frames":np.arange(20)+20,
              "alpha":1.0}
    entry3 = {"save_name":"blackswan_seq",
              "dname":"davis",
              "vname":"blackswan",
              "frames":np.arange(25)+1,
              "alpha":0.5}
    entry4 = {"save_name":"parkour",
              "dname":"davis",
              "vname":"parkour",
              "frames":np.arange(20)+25,
              "alpha":1.0}
    entry5 = {"save_name":"breakdance",
              "dname":"davis",
              "vname":"breakdance",
              "frames":np.arange(20)+1,
              # "frames":np.arange(5)+1,
              "alpha":1.0}
    entry6 = {"save_name":"bmx-trees",
              "dname":"davis",
              "vname":"bmx-trees",
              "frames":np.arange(20)+1,
              # "frames":np.arange(5)+1,
              "alpha":1.0}
    # entry_list = [entry0,entry1]
    # entry_list = [entry1,entry2,entry3,entry4,entry5]
    entry_list = [entry1,entry2,entry3,entry4,entry5,entry6]
    # entry_list = [entry6]
    # entry_list = [entry4,entry5]
    # entry_list = [entry5]
    # entry_list = [entry2]
    entries = {}
    for entry in entry_list:
        save_triplet_seq(entry['save_name'],entry['dname'],
                         entry['vname'],entry['frames'],entry['alpha'])


def save_triplet_seq(save_name,dname,vname,frames,gif_alpha):

    # -- execute summary for each sequence --
    root = Path("output/run_spix/gather_examples_sb/save_triplet_seq/")/save_name
    if not root.exists(): root.mkdir(parents=True)

    def setup_trip(vid,marked_bist,marked_tsp,marked_bass):
        _mgrid = th.stack([vid,marked_bass,marked_bist,marked_tsp],1).cpu()
        mgrid = []
        for ti in range(vid.shape[0]):
            mgrid.append(tv_utils.make_grid(_mgrid[ti],nrow=2))
        mgrid = th.stack(mgrid)
        return mgrid

    # -- read --
    data_root = get_data_root(dname)
    vid = read_video(data_root,vname,frames)#[...,20:-20,100:-100]
    anno = read_anno(data_root,vname,frames)#[...,20:-20,100:-100]
    methods = ["tsp","bist","bass"]
    k = 800
    spix = read_all_spix(dname,vname,frames,k,methods)
    spix_tsp = spix['tsp'].contiguous().int()
    spix_bist = spix['bist'].contiguous().int()
    spix_bass = spix['bass'].contiguous().int()
    trip = format_triplet(vid,anno,spix_tsp,spix_bist,spix_bass,0.5)
    _vid,marked_bist,marked_tsp,marked_bass = trip
    mgrid = setup_trip(_vid,marked_bist,marked_tsp,marked_bass)

    # -- save frames --
    print(f"Saving frames to {str(root)}")
    for frame_index in range(len(mgrid)):
        frame = mgrid[frame_index]
        fname = root / ("%05d.png"%frame_index)
        tv_utils.save_image(frame,fname)

    # -- save gif --
    from PIL import Image
    trip = format_triplet(vid,anno,spix_tsp,spix_bist,spix_bass,gif_alpha)
    _vid,marked_bist,marked_tsp,marked_bass = trip
    mgrid = setup_trip(_vid,marked_bist,marked_tsp,marked_bass)

    root = Path("output/run_spix/gather_examples_sb/save_triplet_seq/gif/")
    if not root.exists(): root.mkdir(parents=True)
    fname = root / ("%s.gif"%save_name)
    gif_list = []
    for img in mgrid:
        img = np.clip(255.*img.cpu().numpy(),0.,255.).astype(np.uint8)
        img = rearrange(img,'c h w -> h w c')
        gif_list.append(Image.fromarray(img))
    gif_list[0].save(fname,
                     save_all=True,
                     append_images=gif_list[1:],
                     duration=300.0,
                     loop=5,
    )
    # kwargs = { 'duration': 1 }
    # imageio.mimsave(fname, gif_list, **kwargs)

def format_triplet(vid,anno,spix_tsp,spix_bist,spix_bass,alpha=0.5):

    # -- mark --
    # color = th.tensor([0.0,0.0,0.8])
    color = th.tensor([1.0,1.0,1.0])*0.7
    _vid = rearrange(vid,'t c h w -> t h w c').contiguous()
    # print("_vid.shape,spix_tsp.shape :",_vid.shape,spix_tsp.shape)
    _marked_tsp = bist_cuda.get_marked_video(_vid,spix_tsp,color)
    marked_tsp = rearrange(_marked_tsp,'t h w c -> t c h w').contiguous()
    _marked_bist = bist_cuda.get_marked_video(_vid,spix_bist,color)
    marked_bist = rearrange(_marked_bist,'t h w c -> t c h w').contiguous()
    _marked_bass = bist_cuda.get_marked_video(_vid,spix_bass,color)
    marked_bass = rearrange(_marked_bass,'t h w c -> t c h w').contiguous()
    # marked_tsp = mark_spix_vid(vid,spix_tsp)
    # marked_bist = mark_spix_vid(vid,spix_bist)


    # # -- Color a Small Region --
    # R = 32
    # for rh in range(R):
    #     for rw in range(R):
    #         h_r = H//2 - R//2 + rh + 80
    #         w_r = W//2 - R//2 + rw - 80
    #         spix_id_tsp = spix_tsp[0,h_r,w_r]
    #         spix_id_bist = spix_bist[0,h_r,w_r]
    #         marked_tsp = color_spix(marked_tsp,spix_tsp,spix_id_tsp,cidx=2)
    #         marked_bist = color_spix(marked_bist,spix_bist,spix_id_bist,cidx=2)

    # -- color based on annotation --
    # color = th.tensor([1.0,0,0])
    color = th.tensor([0.0,0.0,1.0])
    spix_ids_bist = th.unique(spix_bist[0][th.where(anno[0]>0)])
    marked_bist = color_spix(marked_bist,spix_bist,spix_ids_bist,color,alpha)
    # color = th.tensor([0.8,0,0])
    # color = th.tensor([255/255.,20/255.,147/255.])
    color = th.tensor([1.0,0.0,0.0])
    spix_ids_tsp = th.unique(spix_tsp[0][th.where(anno[0]>0)])
    marked_tsp = color_spix(marked_tsp,spix_tsp,spix_ids_tsp,color,alpha)
    # color = th.tensor([255/255., 165/255., 0])
    # color = th.tensor([1.0,0.0,0])
    color = th.tensor([1.0,1.0,1.0])*0.7
    alpha = 0.5
    for i in range(len(spix_bass)):
        spix_ids_bass = th.unique(spix_bass[i][th.where(anno[i]>0)])
        marked_bass[[i]]=color_spix(marked_bass[[i]],spix_bass[[i]],
                                    spix_ids_bass,color,alpha)

    # -- anno mask --
    alpha = 0.5
    mask_rgb = th.tensor([1.0,1.0,1.0])
    mask_rgb = th.tensor(mask_rgb, device=vid.device).view(1, 3, 1, 1) * anno[:,None]
    vid = th.where(mask_rgb > 0, (1 - alpha) * vid + alpha * mask_rgb, vid)

    # -- save --
    # # mgrid = th.stack([vid,marked_bist,marked_tsp,marked_bass],1).cpu()
    # mgrid = th.stack([vid,marked_bist,marked_tsp,marked_bass],1).cpu()
    return vid,marked_bist,marked_tsp,marked_bass

def color_spix(vid,spix,spix_ids,color,alpha):

    mask = th.zeros_like(spix)
    args = th.where(th.isin(spix,spix_ids))
    mask[args] = 1
    mask = mask[:,None]

    mask_rgb = th.tensor(color, device=vid.device).view(1, 3, 1, 1) * mask
    vid = th.where(mask_rgb > 0, (1 - alpha) * vid + alpha * mask_rgb, vid)

    return vid

def get_data_root(dname):
    if "segtrack" in dname:
        data_root="/home/gauenk/Documents/packages/LIBSVXv4.0/Data/SegTrackv2/PNGImages/"
        return data_root
    else:
        data_root="/home/gauenk/Documents/packages/LIBSVXv4.0/Data/DAVIS/PNGImages/"
        return data_root

def compare_many_images():
    # -- execute summary for each sequence --
    root = Path("output/run_spix/gather_examples_sb/")
    if not root.exists(): root.mkdir(parents=True)
    dname = "segtrackerv2"
    vnames = ["frog_3","penguin","cheetah"]
    frames = [10]
    H,W = 200,250
    k = 800
    # methods = ["seeds","ers","tsp","mbass","st_spix"]
    methods = ["seeds","ers","tsp","bass","bist"]

    # -- read --
    vid,spix = [],{}
    for vname in vnames:
        vid_v = read_video(data_root,vname,frames)[...,:H,:W]


        spix_v = read_all_spix(dname,vname,frames,k,methods)
        vid.append(vid_v)
        if len(spix) == 0:
            spix = {_k:[v[...,:H,:W]] for _k,v in spix_v.items()}
        else:
            for _k,v in spix_v.items(): spix[_k].append(v[...,:H,:W])
    for _k in spix: spix[_k] = th.cat(spix[_k])
    vid = th.cat(vid)

    # -- mark --
    _vid = rearrange(vid,'t h w c -> t c h w').contiguous()
    marked = {}
    for method in spix:
        print(spix[method].shape,vid.shape)
        _marked = bist_cuda.get_marked_video(_vid,spix)
        marked[method] = rearrange(_marked,'t h w c -> t c h w').contiguous()
        # marked[method] = mark_spix_vid(vid,spix[method])

    # -- save --
    names = list(marked.keys())
    mgrid = th.cat([vid.cpu(),]+list(marked.values()))
    mgrid = tv_utils.make_grid(mgrid,nrow=len(vid))
    tv_utils.save_image(mgrid,root/"compare_many_images.png")
    print(names)


def main():

    # -- execute summary for each sequence --
    root = Path("output/run_spix/gather_examples_sb/")
    # data_root = "/home/gauenk/Documents/packages/LIBSVXv4.0/Data/SegTrackv2/PNGImages/"
    if not root.exists(): root.mkdir(parents=True)
    dname = "segtrackerv2"
    vname = "monkey"
    frames = [10,11,12]
    k = 800
    H,W = 200,250
    # methods = ["seeds","ers","tsp","mbass","st_spix"]
    methods = ["seeds","ers","tsp","bass","bist"]

    # -- read --
    data_root = get_data_root(dname)
    vid = read_video(data_root,vname,frames)
    spix = read_all_spix(dname,vname,frames,k,methods)

    vid = vid[...,:H,:W]
    for _k,_v in spix.items(): spix[_k] = _v[...,:H,:W]


    # -- mark --
    marked = {}
    for method in spix:
        print(spix[method].shape,vid.shape)
        _vid = rearrange(vid,'t h w c -> t c h w').contiguous()
        _marked = bist_cuda.get_marked_video(_vid,spix)
        marked[method] = rearrange(_marked,'t h w c -> t c h w').contiguous()
        # marked[method] = mark_spix_vid(vid,spix[method])

    # -- save --
    names = list(marked.keys())
    mgrid = th.cat([vid.cpu(),]+list(marked.values()))
    mgrid = tv_utils.make_grid(mgrid,nrow=len(vid))
    tv_utils.save_image(mgrid,root/"marked.png")
    print(names)

    # metrics = df['metric'].unique().tolist()
    # df = df.pivot(index=["k", "method", "vid_name"],
    #               columns="metric", values="mean").reset_index()
    # df.drop("vid_name",axis=1,inplace=True)
    # df = df.groupby(["k", "method"]).mean().reset_index()
    # df = df.sort_values("sp")
    # print(df['method'].unique())


    # # -- fig --
    # dpi = 300
    # ginfo = {'wspace':0.01, 'hspace':0.1,
    #          "top":0.99,"bottom":0.09,"left":.125,"right":0.99}
    # fig,axes = plt.subplots(3,1,figsize=(6,5),gridspec_kw=ginfo,dpi=dpi)

    # # -- metrics --
    # # metrics = ["asa","ue","rec","sse_xy","cd","co","ev"]
    # # metrics = ["asa","ev","cd"]
    # metrics = ["asa","ue","cd"]

    # # -- plots --
    # i = 0
    # for metric in metrics:
    #     plot_metric(axes[i],df,root,metric)
    #     if i < 2:
    #         axes[i].set_xticks([])
    #         axes[i].set_xticklabels([])
    #     i+=1
    # axes[0].legend(ncols=3,framealpha=0.0)
    # axes[-1].set_xlabel("Number of Superpixels",fontsize=12)

    # # Add an arrow to the right of the plots
    # # fig_width = fig.get_figwidth()
    # # arrow_x = fig_width * 0.98  # Position at the right edge
    # # fig.gca().add_patch(FancyArrow(x=arrow_x, y=0.5, dx=0.1, dy=0,
    # #                                width=0.02, color='black',
    # #                                transform=fig.transFigure, clip_on=False))

    # plot_arrows(fig,axes)
    # plt.savefig(root/("single_image_summary.png"),transparent=True)

if __name__ == "__main__":
    # main()
    # compare_single_video()
    save_triplet_collection()
    # compare_many_images()
