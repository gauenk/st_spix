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

import matplotlib.gridspec as gridspec


import matplotlib.pyplot as plt
from matplotlib.patches import FancyArrow

def get_video_names(dname):
    if dname == "davis":
        return get_davis_videos()
    elif dname == "segtrackerv2":
        return get_segtrackerv2_videos()
    else:
        raise ValueError("")

def get_segtrackerv2_videos():
    root = Path("/home/gauenk/Documents/packages/superpixel-benchmark/docker/in/")
    root = root /"SegTrackv2/GroundTruth/"
    vid_names = list([v.name for v in root.iterdir()])
    return vid_names

def get_davis_videos():
    try:
        names = np.loadtxt("/app/in/DAVIS/ImageSets/2017/train-val.txt",dtype=str)
    except:
        fn = "/home/gauenk/Documents/data/davis/DAVIS/ImageSets/2017/train-val.txt"
        names = np.loadtxt(fn,dtype=str)
    return names

def get_method_params(method):
    if method == "etps":
        params_v = odict({"superpixels":[200,300,400,500,600,700,800,900,1000,1100,1200],
                             "regularization-weight":[0.01],"length-weight":[0.1],
                             "size-weight":[1],"iterations":[25]})
    elif method == "seeds":
        params_v = odict({"superpixels":[200,300,400,500,600,700,800,900,1000,1100,1200],
                          "bins":[5],"prior":[0],
                          "confidence":[0.1],"iterations":[25],"color-space":[1],"means":[1]})
    elif method == "ers":
        params_v = odict({"superpixels":[200,300,400,500,600,700,800,900,1000,1100,1200],
                          "lambda":[0.5],"sigma":[5]})
    elif method == "st_spix":
        params_v = odict({"superpixels":[200,250,300,350,400,450,500,550,600,800,1000,1200]})
    elif method == "mbass":
        params_v = odict({"superpixels":[200,250,300,350,400,450,500,550,600,800,1000,1200]})
    elif method == "tsp":
        params_v = odict({"superpixels":[100,150,200,250,300,350,400,450,500,550,600,800,1000,1200]})
    else:
        raise ValueError(f"Uknown method params [{method}]")

    param_grid = []
    N = len(params_v["superpixels"])
    for n in range(N):
        params_n = {}
        for key,val in params_v.items():
            val_n = val[n] if len(val) > 1 else val[0]
            params_n[key] = val_n
        param_grid.append(params_n)
    if method in ["st_spix","mbass","tsp"]:
        names = ["sp%d"%p['superpixels'] for p in param_grid]
    else:
        names = ["%02dsp"%(p['superpixels']/100) for p in param_grid]
    # names = ["%02dsp"%(p['superpixels']/100) for p in param_grid]
    return param_grid,names

def read_results(root,dname,refresh=False):

    # -- read cached --
    cache_fn = root/"cached_results.csv"
    if not refresh and cache_fn.exists():
        return pd.read_csv(cache_fn)

    # -- run script --
    base = Path("/home/gauenk/Documents/packages/superpixel-benchmark/docker/")
    vid_names = get_video_names(dname)
    methods = ["etps","seeds","ers","mbass","st_spix","tsp"]
    # methods = ["tsp"]
    df = []
    for method in tqdm.tqdm(methods,position=0):
        param_grid,pnames = get_method_params(method)
        for params,pname in tqdm.tqdm(zip(param_grid,pnames),position=1,leave=False,total=len(param_grid)):
            for vid_name in tqdm.tqdm(vid_names,position=2,leave=False):
                sp_dir = base/Path("out/%s/%s/%s/%s/" % (dname,method,pname,vid_name))
                fn = sp_dir/"summary.csv"
                if not fn.exists(): break
                _df = pd.read_csv(fn)
                _df = _df[["metric","mean[0]"]]
                _df = _df.rename(columns={"mean[0]":"mean"})
                if method in ["etps","seeds","ers"]:
                    _df['k'] = int(pname.split("sp")[0])*100
                else:
                    _df['k'] = int(pname.split("sp")[1])
                _df['method'] = method
                _df['vid_name'] = vid_name
                # exit()
                df.append(_df)

    # -- combine --
    df = pd.concat(df)

    print(cache_fn)
    # -- save cache --
    df.to_csv(cache_fn)
    return df


def plot_metric(ax,df,root,metric):

    # -- fig --
    dpi = 200
    # ginfo = {'wspace':0.01, 'hspace':0.01,
    #          "top":0.92,"bottom":0.16,"left":.07,"right":0.98}
    # fig,ax = plt.subplots(1,1,figsize=(5,4),gridspec_kw=ginfo,dpi=200)
    ymin = 100000
    ymax = -1
    for method,dfm in df.groupby("method"):
        x = dfm['sp'].to_numpy()
        y = dfm[metric].to_numpy()
        args = np.where(x<1400)
        if method == "mbass":
            method = "M-BASS"
        elif method == "st_spix":
            method = "BIST"
        ax.plot(x[args],y[args],label=method.upper())
        _ymin,_ymax = y[args].min(),y[args].max()
        ymin = _ymin if _ymin < ymin else ymin
        ymax = _ymax if _ymax > ymax else ymax

    # Set three y-ticks
    yticks = np.linspace(ymin,ymax,3)
    ax.set_yticks(yticks)
    ax.set_yticklabels("%1.3f"%y for y in yticks)
    if metric == "asa":
        metric = "SA"
    ax.set_ylabel(metric.upper(),fontsize=12,fontweight='bold',labelpad=6)

    # ax.legend()
    # plt.savefig(root/("%s.png"%metric))
    # plt.close("all")

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



def read_bench():
    base = Path("/home/gauenk/Documents/packages/superpixel-benchmark/docker/")
    dname = "segtrackerv2"
    vid_names = get_video_names(dname)
    methods0 = ["etps","seeds","ers"]
    methods1 = ["mbass","st_spix","tsp"]

    for method in methods0:
        param_grid,pnames = get_method_params(method)
        for params,pname in tqdm.tqdm(zip(param_grid,pnames),position=1,leave=False,total=len(param_grid)):
            if method in ["etps","seeds","ers"]:
                K = int(pname.split("sp")[0])*100
            else:
                K = int(pname.split("sp")[1])


def plot_runtime(ax):
    # names = ["ETPS","SEEDS","ERS","TSP","S-GBH","FGCS","M-BASS","BIST"]
    # times = [1000,1000,1000,2000,3000,20,100,40]
    # ax.bar(names,times)

    # todo: order by publication date

    names0 = ["ETPS","SEEDS","ERS","FGCS","M-BASS","BASS"]
    times0 = [1000,1000,1000,20,100,80]
    ax.bar(names0,times0,color="#D6C0B3",label="Space")
    names1 = ["TSP","S-GBH","CCS","TCS","BIST"]
    times1 = [500,400,300,600,40]
    ax.bar(names1,times1,color="#AB886D",label="Space-Time")

    ymin = min(np.min(times0),np.min(times1))
    ymax = max(np.max(times0),np.max(times1))
    ax.set_ylim([ymin,ymax*2.])
    ax.set_yscale("log")
    ax.set_ylabel("Runtime (ms)",fontsize=12,fontweight="bold")
    ax.set_xticklabels(names0+names1, rotation=45, ha='right')
    for lab in ax.get_xticklabels():
        if lab.get_text() == "BIST":
            lab.set_fontweight('bold')
    ax.legend(fontsize="10",framealpha=0.,ncols=2)

def main():

    # -- execute summary for each sequence --
    root = Path("output/run_spix/results_sb/")
    if not root.exists(): root.mkdir(parents=True)
    dname = "segtrackerv2"

    # -- read --
    df = read_results(root,dname,refresh=False)
    metrics = df['metric'].unique().tolist()
    df = df.pivot(index=["k", "method", "vid_name"],
                  columns="metric", values="mean").reset_index()
    df.drop("vid_name",axis=1,inplace=True)
    df = df.groupby(["k", "method"]).mean().reset_index()
    df = df.sort_values("sp")
    print(df['method'].unique())


    # -- fig --
    dpi = 300
    ginfo = {# 'wspace':0.01,
             'hspace':np.array([0.1,0.1,0.2,0.2]),
             "top":0.99,"bottom":0.09,"left":.125,"right":0.99}
    # fig,axes = plt.subplots(4,1,figsize=(8,5),gridspec_kw=ginfo,dpi=dpi)
    fig = plt.figure(figsize=(6, 7.25), dpi=dpi)


    # Define the GridSpec layout with two separate grids
    gs = gridspec.GridSpec(5, 1, figure=fig, height_ratios=[1, 1, 1, 0.15, 1])
    axes = [fig.add_subplot(gs[i]) for i in range(3)]+[fig.add_subplot(gs[-1])]

    # # Define the GridSpec layout
    # gs = gridspec.GridSpec(4, 1, figure=fig, height_ratios=[1, 1, 1, 1.5])  # Adjust the last value for extra spacing
    # # gs = gridspec.GridSpec(4, 1, figure=fig)
    # # gs.update(top=0.99, bottom=0.09, left=0.125, right=0.99, wspace=0.01)

    # # # Adjust hspace for the first three subplots and add extra space for the fourth
    # # gs[0].update(hspace=0.1)
    # # gs[1].update(hspace=0.1)
    # # gs[2].update(hspace=0.2)
    # # gs[3].update(hspace=0.3)  # Extra space before the fourth subplot
    # axes = [fig.add_subplot(gs[i]) for i in range(4)]

    fig.subplots_adjust(top=0.99, bottom=0.08, left=0.125, right=0.99,hspace=0.15)

    # -- metrics --
    # metrics = ["asa","ue","rec","sse_xy","cd","co","ev"]
    # metrics = ["asa","ev","cd"]
    metrics = ["asa","ue","cd"]

    # -- plots --
    i = 0
    for metric in metrics:
        plot_metric(axes[i],df,root,metric)
        if i < 2:
            axes[i].set_xticks([])
            axes[i].set_xticklabels([])
        i+=1
    axes[0].legend(ncols=3,framealpha=0.0)
    axes[-2].set_xlabel("Number of Superpixels",fontsize=12)

    # Add an arrow to the right of the plots
    # fig_width = fig.get_figwidth()
    # arrow_x = fig_width * 0.98  # Position at the right edge
    # fig.gca().add_patch(FancyArrow(x=arrow_x, y=0.5, dx=0.1, dy=0,
    #                                width=0.02, color='black',
    #                                transform=fig.transFigure, clip_on=False))

    plot_arrows(fig,axes[:3])

    plot_runtime(axes[-1])


    plt.savefig(root/("single_image_summary.png"),transparent=True)

    # -- v2 --
    # for k,df_k in df.groupby("k"):
    #     print("\n\n\n")
    #     print(k)
    #     print("\n\n\n")
    #     for metric,df_metr in df_k.groupby("metric"):
    #         print(" -- %s -- " % metric)
    #         for method,df_method in df_metr.groupby("method"):
    #             # print("(%d,%2.5f)" % (k,df_k['mean'].mean().item()))
    #             print("(%s,%2.5f)" % (method,df_method['mean'].mean().item()))

    # print(df)

if __name__ == "__main__":
    main()
