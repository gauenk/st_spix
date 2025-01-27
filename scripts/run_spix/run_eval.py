"""

   Run superpixel eval

"""

import os
import tqdm
import h5py
import numpy as np
import pandas as pd
from pathlib import Path
from PIL import Image
from einops import rearrange

from st_spix.spix_utils.evaluate import computeSummary

# import matplotlib.pyplot as plt
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from matplotlib.patches import FancyArrow


def read_video(vname):
    root = Path("/home/gauenk/Documents/packages/superpixel-benchmark/docker/in/")
    root = root /"SegTrackv2/PNGImages/" /vname
    nframes = len([f for f in root.iterdir() if str(f).endswith(".png")])
    vid = []
    for frame_ix in range(nframes):
        fname = root/("%05d.png" % (frame_ix+1))
        img = np.array(Image.open(fname).convert("RGB"))/255.
        vid.append(img)
    vid = np.stack(vid)
    return vid

def read_seg_loop(root):
    nframes = len([f for f in root.iterdir() if str(f).endswith(".png")])
    vid = []
    for frame_ix in range(nframes):
        fname = root/("%05d.png" % (frame_ix+1))
        img = 1.*(np.array(Image.open(fname).convert("L")) >= 128)
        vid.append(img)
    vid = np.stack(vid)
    return vid

def read_seg(vname):
    root = Path("/home/gauenk/Documents/packages/superpixel-benchmark/docker/in/")
    root = root /"SegTrackv2/GroundTruth/" /vname
    has_subdirs = np.all([f.is_dir() for f in root.iterdir()])
    if has_subdirs:
        seg = None
        for ix,subdir in enumerate(root.iterdir()):
            if seg is None:
                seg = read_seg_loop(subdir)
            else:
                tmp = read_seg_loop(subdir)
                seg[np.where(tmp>0)] = ix+1
                # tmp[np.where(tmp)>0] = ix
                # print(ix,np.unique(tmp))
                # seg = seg + (ix+1)*read_seg_loop(subdir)
    else:
        seg = read_seg_loop(root)
    # print(np.unique(seg))
    # exit()
    return seg

def get_segtrackerv2_videos():
    root = Path("/home/gauenk/Documents/packages/superpixel-benchmark/docker/in/")
    root = root /"SegTrackv2/GroundTruth/"
    vid_names = list([v.name for v in root.iterdir()])
    return vid_names


def get_sp_grid(group,root,method):
    if group == "spix-bench":
        # path = root/"superpixel-benchmark/docker/out/segtrackerv2/"/method
        path = root / method
        ids = [int(str(f.name).split("sp")[0]) for f in path.iterdir()]
        return ids
    elif group == "libsvx":
        vname = "birdfall"
        path = root / method / "Segments" / vname
        check = lambda f: str(f.name).endswith(".mat")
        proc = lambda f: int(str(f.name).split(".")[0])
        ids = [proc(f) for f in path.iterdir() if check(f)]
        return ids
    elif group == "stspix":
        # vname = "birdfall"
        path = root / method
        # path = root /"output/run_segtrackerv2_spix/"/method
        # check = lambda f: str(f.name).endswith("sp")
        check = lambda f: str(f.name).startswith("sp")
        proc = lambda f: int(str(f.name).split("sp")[1])
        ids = [proc(f) for f in path.iterdir() if check(f)]
        return ids
    elif group == "gbass":
        # vname = "cheetah"
        path = root / method
        check = lambda f: str(f.name).startswith("sp")
        proc = lambda f: int(str(f.name).split("sp")[1])
        ids = [proc(f) for f in path.iterdir() if check(f)]
        return ids
    elif group == "bist":
        path = root / method
        check = lambda f: str(f.name).startswith("sp")
        proc = lambda f: int(str(f.name).split("sp")[1])
        ids = [proc(f) for f in path.iterdir() if check(f)]
        return ids
    else:
        raise ValueError("")

def read_csv(root,nframes,offset_fidx=0):
    # nframes = len([f for f in root.iterdir() if str(f).endswith(".csv")])
    # nframes = len([f for f in root.iterdir() if str(f).endswith(".csv")])
    spix = []
    for fidx in range(nframes):
        fname = str(root/("%05d.csv"%(fidx+offset_fidx)))
        spix.append(pd.read_csv(fname,header=None))
    spix = np.stack(spix)
    return spix

def read_mat(fname):
    spix = np.array(h5py.File(fname)['svMap'])
    spix = rearrange(spix,'t w h -> t h w')
    return spix

def read_spix(group,root,method,vname,sp,nframes):
    if group == "spix-bench":
        return read_csv(root / method / ("%02dsp"%sp) / vname,nframes,1)
    elif group == "libsvx":
        return read_mat(root / method / "Segments" / vname / ("%02d.mat"%sp))
    elif group == "stspix":
        return read_csv(root / method / ("sp%d"%sp) / vname,nframes)
    elif group == "bist":
        return read_csv(root / method / ("sp%d"%sp) / vname,nframes,1)
    elif group == "gbass":
        return read_csv(root / method / ("sp%d"%sp) / vname,nframes,1)
    else:
        raise ValueError("")

def read_cache(cache_root,group,method):
    cache_fn = cache_root / ("%s_%s.csv"%(group,method))
    # print(cache_fn)
    if not cache_fn.exists(): return None
    else: return pd.read_csv(cache_fn)

def save_cache(summs,cache_root,group,method):
    if not cache_root.exists():cache_root.mkdir(parents=True)
    cache_fn = cache_root / ("%s_%s.csv"%(group,method))
    print(summs)
    print(pd.DataFrame(summs))
    # exit()
    pd.DataFrame(summs).to_csv(cache_fn)

def get_group_root(group):
    root = Path("/home/gauenk/Documents/packages/")
    if group == "stpix":
        base = root/"st_spix/output/run_segtrackerv2_spix/"
    elif group == "spix-bench":
        base = root/"superpixel-benchmark/docker/out/segtrackerv2/"
    elif group == "libsvx":
        base = root/"LIBSVXv4.0/Results/SegTrackv2/"
    elif group == "bist":
        base = root/"st_spix_refactor/result/bist"
    else:
        raise ValueError("")
    return base


def process_group(group,base,methods,refresh=False):


    # -- init --
    cache_root = Path("./output/run_eval/cache")
    # refresh = False

    # -- run --
    summs_agg = []
    vnames = get_segtrackerv2_videos()
    # vnames = ["frog_2"]
    for method in tqdm.tqdm(methods,position=0):

        # -- reading cache --
        summs = read_cache(cache_root,group,method)
        if not(summs is None) and (refresh is False):
            summs_agg.append(summs)
            continue
        else:
            summs = []

        # base = get_group_root(group)
        spgrid = get_sp_grid(group,base,method)
        for vname in tqdm.tqdm(vnames,position=1,leave=False):
            vid = read_video(vname)
            seg = read_seg(vname)
            nframes = len(vid)
            for sp in tqdm.tqdm(spgrid,position=2,leave=False):
                spix = read_spix(group,base,method,vname,sp,nframes)
                # print(vname,method,sp)
                # print(vid.shape,seg.shape,spix.shape)
                # exit()
                # print(seg.min(),seg.max())
                _summ = computeSummary(vid,seg,spix)
                _summ.name = vname
                _summ.method = method
                _summ.nspix = len(np.unique(spix))
                _summ.param = sp
                summs.append(_summ)

        # -- caching --
        # print(summs)
        save_cache(summs,cache_root,group,method)
        summs_agg.append(pd.DataFrame(summs))

    # print(summs_agg)
    return pd.concat(summs_agg)


def plot_metric(ax,df,root,metric):

    # -- fig --
    dpi = 200
    # ginfo = {'wspace':0.01, 'hspace':0.01,
    #          "top":0.92,"bottom":0.16,"left":.07,"right":0.98}
    # fig,ax = plt.subplots(1,1,figsize=(5,4),gridspec_kw=ginfo,dpi=200)
    ymin = 100000
    ymax = -1
    # print("\n"*20)


    methods = ["bist","st_spix","TSP","streamGBH",
               "mbass","bass","ers","etps","seeds","slic"]
    # for method,dfm in df.groupby("method"):
    for method in methods:
        dfm = df[df['method'] == method]
        if len(dfm) == 0: continue
        # x = dfm['nspix'].to_numpy()
        x = dfm['ave_nsp'].to_numpy()
        y = dfm[metric].to_numpy()
        # args = np.where(np.logical_and(x>200,x<1400))
        # args = np.where(np.logical_and(x>100,x<1400))
        # x,y = x[args],y[args]
        args = np.argsort(x)
        if method == "mbass":
            method = "BASS"
        elif method == "st_spix":
            method = "BIST-v0"
        elif method == "bist":
            method = "BIST"
        elif method == "streamGBH":
            method = "sGBH"
        else:
            method = method.upper()
        if metric in ["tex","szv","ue3d","sa3d"] and not(method in ["BIST","BIST-v0","TSP","sGBH"]):
            continue
        # if metric in ["ue2d","sa2d"
        colors = {"BIST":"blue", "BIST-v0":"black","BASS":"red","TSP":"green", "sGBH":"pink", "ETPS":"orange",  "SLIC":"grey", "ERS":"purple", "SEEDS":"brown"}
        # print(x[args],y[args])
        if len(x) == 1:
            ax.plot(np.r_[x[0]-10,x[0]],np.r_[y[0],y[0]],label=method,color=colors[method])
        else:
            ax.plot(x[args],y[args],label=method,color=colors[method])
        _ymin,_ymax = y[args].min(),y[args].max()
        ymin = _ymin if _ymin < ymin else ymin
        ymax = _ymax if _ymax > ymax else ymax
        # print(x[args],y[args],method,ymin,ymax)


    # Set three y-ticks
    yticks = np.linspace(ymin*0.95,ymax*1.1,3)
    # ax.grid(True)
    ax.set_yticks(yticks)
    ax.set_yticklabels("%1.3f"%y for y in yticks)
    if metric == "asa":
        metric = "SA"
    if metric.lower() in ["sa2d","sa3d","pooling","ev"]:
        arrow_s = r"$\uparrow$"
    elif metric.lower() in ["tex","szv"]:
        arrow_s = ""
    else:
        arrow_s = r"$\downarrow$"
    ylabel = metric.upper() + arrow_s
    ax.set_ylabel(ylabel,fontsize=12,fontweight='bold',labelpad=6)

    # ax.legend()
    # plt.savefig(root/("%s.png"%metric))
    # plt.close("all")

def plt_arrows(fig,ax,a,b):
    bbox = ax.get_position()  # Get the position of the axis
    alpha = 0.45
    arrow_x = bbox.x0-0.05
    arrow_y = (1-alpha)*bbox.y0 + alpha*bbox.y1
    # arrow_y = arrow_y - 0.025/4.
    fig.patches.append(FancyArrow(x=arrow_x, y=arrow_y, dx=0, dy=0.025,
                                  width=0.0025, color='black',
                                  transform=fig.transFigure,clip_on=False))

def plot_arrows(fig,axes):
    # Add a vertical arrow to the right of each subplot
    for ix,ax in enumerate(axes):
        bbox = ax.get_position()  # Get the position of the axis
        # arrow_x = bbox.x1 - 0.00  # Position slightly to the right of each axis
        # arrow_x = 0.120
        # alpha = 0.75
        alpha = 0.45
        arrow_x = 0.033
        arrow_y = (1-alpha)*bbox.y0 + alpha*bbox.y1

        if ix == 0:
            arrow_y = arrow_y - 0.025/4.
            fig.patches.append(FancyArrow(x=arrow_x, y=arrow_y, dx=0, dy=0.025,
                                          width=0.0025, color='black',
                                          transform=fig.transFigure,
                                          clip_on=False))
        else:
            arrow_y = arrow_y + 0.025
            fig.patches.append(FancyArrow(x=arrow_x, y=arrow_y, dx=0, dy=-0.025,
                                          width=0.0025, color='black',
                                          transform=fig.transFigure,
                                          clip_on=False))
        # fig.patches.append(FancyArrow(x=arrow_x, y=arrow_y, dx=0, dy=-0.10,
        #                               width=0.01, color='black',
        #                               transform=fig.transFigure,
        #                               clip_on=False))

def nice_plots(df):
    # df.groupby("name")
    # print(df)
    if "name" in list(df.columns):
        df.drop("name",axis=1,inplace=True)
        # df.drop("param",axis=1,inplace=True)
        df = df.groupby(["method", "param"]).mean().reset_index()
        print(df)


    # -- fig --
    root = Path("output/run_eval/")
    if not root.exists():
        root.mkdir(parents=True)
    dpi = 300
    ginfo = {'wspace':0.5,"hspace":0.1,
             "top":0.90,"bottom":0.12,"left":.07,"right":0.99}
    fig,axes = plt.subplots(2,4,figsize=(12,4),gridspec_kw=ginfo,dpi=dpi)
    metrics = ["ue2d","sa2d","pooling","ev",
               "ue3d","sa3d","tex","szv"]


    # -- plots --
    i = 0
    for metric in metrics:
        b = i % 4
        a = i // 4
        ax = axes[a][b]

        plot_metric(ax,df,root,metric)
        if a == 0:
            ax.set_xticks([])
            ax.set_xticklabels([])
        else:
            kgrid = [200,700,1200]
            ax.set_xticks(kgrid)
            ax.set_xticklabels(kgrid)
            ax.set_xlabel("Number of Superpixels",fontsize=12)

        # -- arrows --
        # plt_arrows(fig,ax,a,b)

        # -- incriment --
        i+=1

    axes[0][0].legend(ncols=len(metrics),framealpha=0.0,fontsize=10,
                      loc='upper center', bbox_to_anchor=(2.8, 1.28))

    plt.savefig(root/("spix_summary.png"),transparent=True)


    # axes[1][1].legend(ncols=3,framealpha=0.0,fontsize=10)
    # Add an arrow to the right of the plots
    # fig_width = fig.get_figwidth()
    # arrow_x = fig_width * 0.98  # Position at the right edge
    # fig.gca().add_patch(FancyArrow(x=arrow_x, y=0.5, dx=0.1, dy=0,
    #                                width=0.02, color='black',
    #                                transform=fig.transFigure, clip_on=False))

    # plot_arrows(fig,axes[:3])

    # plot_runtime(axes[-1])


    # fig = plt.figure(figsize=(6, 7.25), dpi=dpi)

    # Define the GridSpec layout with two separate grids
    # gs = gridspec.GridSpec(5, 1, figure=fig, height_ratios=[1, 1, 1, 0.15, 1])
    # axes = [fig.add_subplot(gs[i]) for i in range(3)]+[fig.add_subplot(gs[-1])]
    # fig.subplots_adjust(top=0.99, bottom=0.08, left=0.125, right=0.99,hspace=0.15)

    # -- metrics --
    # metrics = ["asa","ue","rec","sse_xy","cd","co","ev"]
    # metrics = ["asa","ev","cd"]
    # metrics = ["asa","ue","cd"]
    # summ.tex,summ.szv

    # # -- fig --
    # dpi = 300
    # ginfo = {'wspace':0.01, 'hspace':0.1,
    #          "top":0.99,"bottom":0.09,"left":.125,"right":0.99}
    # fig,axes = plt.subplots(3,1,figsize=(6,5),gridspec_kw=ginfo,dpi=dpi)


def main():

    print("PID: ",os.getpid())
    root = Path("/home/gauenk/Documents/packages/")
    # base = root/"st_spix/output/run_segtrackerv2_spix/"

    # -- group 0 --
    group = "stspix"
    base = root/"st_spix/output/run_segtrackerv2_spix/"
    methods = ["mbass","st_spix","bist"]
    # methods = ["mbass","st_spix","bist"]
    methods = ["st_spix"]
    df0 = process_group(group,base,methods)

    # -- group 1 --
    # group = "spix-bench"
    # base = root/"superpixel-benchmark/docker/out/segtrackerv2/"
    # # methods = ["ccs","ers","etps","seeds","slic"]
    # methods = ["ers","etps","seeds","slic"]
    # df1 = process_group(group,base,methods)

    # -- group 2 --
    group = "libsvx"
    base = root/"LIBSVXv4.0/Results/SegTrackv2/"
    # methods = ["TSP","streamGBH"]
    methods = ["TSP"]
    df2 = process_group(group,base,methods)

    # -- group 3 --
    group = "gbass"
    base = root/"/home/gauenk/Documents/packages/BASS_check/result/"
    methods = ["bass"]
    df3 = process_group(group,base,methods)

    # -- group 4 --
    group = "bist"
    methods = ["bist"]
    base = root/"st_spix_refactor/result/"
    df4 = process_group(group,base,methods,True)

    # -- plots --
    # df = pd.concat([df0,df1,df2,df3])
    df = pd.concat([df0,df2,df3,df4])
    # df = pd.concat([df0,df3,df4])
    # df = pd.concat([df0,df1,df2])
    # df = df3

    #
    # --
    #

    # -- a --
    # df = df[df['name'] == "frog_2"].reset_index(drop=True)

    # -- b --
    df.drop("name",axis=1,inplace=True)
    df = df.groupby(["method", "param"]).mean().reset_index()

    # df = df[ (df['ave_nsp'] > 200) & (df['ave_nsp']<500)].reset_index(drop=True)
    df = df[ (df['ave_nsp'] > 200) & (df['ave_nsp']<1000)].reset_index(drop=True)
    # print(df.columns)
    cols = ['method','ave_nsp','pooling','ue2d','sa2d','ue3d','sa3d','tex']
    print(df[cols])
    # exit()
    nice_plots(df)

if __name__ == "__main__":
    main()

