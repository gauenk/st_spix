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
    if method in ["etps","seeds","ers"]:
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

def read_all_spix(vid_name,frames,k,methods):

    # -- run script --
    dname = "segtrackerv2"
    # base = Path("/home/gauenk/Documents/packages/superpixel-benchmark/docker/")
    # vid_names = get_video_names(dname)
    # methods = ["etps","seeds","ers","mbass","st_spix","tsp"]
    spix = {}
    for method in methods:
        base = get_base(method)
        kstr = get_kstr(method,k)
        sp_dir = base/Path("%s/%s/%s/%s/" % (dname,method,kstr,vid_name))
        spix_m = read_spix(sp_dir,frames)
        spix[method] = spix_m
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


def compare_single_video():
    # -- execute summary for each sequence --
    root = Path("output/run_spix/gather_examples_sb/")
    data_root = "/home/gauenk/Documents/packages/LIBSVXv4.0/Data/SegTrackv2/PNGImages/"
    if not root.exists(): root.mkdir(parents=True)
    dname = "segtrackerv2"
    vname = "monkey"
    frames = [1,2,3,4,5,6,7]
    k = 800

    # -- read --
    vid = read_video(data_root,vname,frames)[...,20:-20,100:-100]
    # methods = ["tsp","st_spix"]
    methods = ["tsp","bist"]
    spix = read_all_spix(vname,frames,k,methods)
    spix_tsp = spix['tsp'][...,20:-20,100:-100]
    spix_stb = spix['bist'][...,20:-20,100:-100]
    T,F,H,W = vid.shape

    # -- mark --
    marked_tsp = mark_spix_vid(vid,spix_tsp)
    marked_stb = mark_spix_vid(vid,spix_stb)

    # -- color a fun spix --
    R = 32
    for rh in range(R):
        for rw in range(R):
            h_r = H//2 - R//2 + rh
            w_r = W//2-20 - R//2 + rw
            spix_id_tsp = spix_tsp[0,h_r,w_r].item()
            spix_id_stb = spix_stb[0,h_r,w_r].item()
            marked_tsp = color_spix(marked_tsp,spix_tsp,spix_id_tsp,cidx=2)
            marked_stb = color_spix(marked_stb,spix_stb,spix_id_stb,cidx=2)

    # -- save --
    mgrid = th.cat([vid.cpu(),marked_tsp,marked_stb])
    mgrid = tv_utils.make_grid(mgrid,nrow=len(vid))
    print(f"Saving to {str(root)}")
    tv_utils.save_image(mgrid,root/"compare_single_video.png")


def color_spix(vid,spix,spix_id,cidx=0):
    for t in range(vid.shape[0]):
        for ci in range(3):
            vid[t,ci][th.where(spix[t]==spix_id)] = 1.*(ci==cidx)
    return vid

def compare_many_images():
    # -- execute summary for each sequence --
    root = Path("output/run_spix/gather_examples_sb/")
    data_root = "/home/gauenk/Documents/packages/LIBSVXv4.0/Data/SegTrackv2/PNGImages/"
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


        spix_v = read_all_spix(vname,frames,k,methods)
        vid.append(vid_v)
        if len(spix) == 0:
            spix = {_k:[v[...,:H,:W]] for _k,v in spix_v.items()}
        else:
            for _k,v in spix_v.items(): spix[_k].append(v[...,:H,:W])
    for _k in spix: spix[_k] = th.cat(spix[_k])
    vid = th.cat(vid)

    # -- mark --
    marked = {}
    for method in spix:
        print(spix[method].shape,vid.shape)
        marked[method] = mark_spix_vid(vid,spix[method])

    # -- save --
    names = list(marked.keys())
    mgrid = th.cat([vid.cpu(),]+list(marked.values()))
    mgrid = tv_utils.make_grid(mgrid,nrow=len(vid))
    tv_utils.save_image(mgrid,root/"compare_many_images.png")
    print(names)


def main():

    # -- execute summary for each sequence --
    root = Path("output/run_spix/gather_examples_sb/")
    data_root = "/home/gauenk/Documents/packages/LIBSVXv4.0/Data/SegTrackv2/PNGImages/"
    if not root.exists(): root.mkdir(parents=True)
    dname = "segtrackerv2"
    vname = "monkey"
    frames = [10,11,12]
    k = 800
    H,W = 200,250
    # methods = ["seeds","ers","tsp","mbass","st_spix"]
    methods = ["seeds","ers","tsp","bass","bist"]

    # -- read --
    vid = read_video(data_root,vname,frames)
    spix = read_all_spix(vname,frames,k,methods)

    vid = vid[...,:H,:W]
    for _k,_v in spix.items(): spix[_k] = _v[...,:H,:W]


    # -- mark --
    marked = {}
    for method in spix:
        print(spix[method].shape,vid.shape)
        marked[method] = mark_spix_vid(vid,spix[method])

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
    compare_single_video()
    # compare_many_images()
