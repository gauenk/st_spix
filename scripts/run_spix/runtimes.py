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

def plot_runtime(ax):
    # names = ["ETPS","SEEDS","ERS","TSP","S-GBH","FGCS","M-BASS","BIST"]
    # times = [1000,1000,1000,2000,3000,20,100,40]
    # ax.bar(names,times)

    # todo: order by publication date

    # names0 = ["ETPS","SEEDS","ERS","FGCS","M-BASS","BASS"]
    # times0 = [1000,1000,1000,20,100,80]
    names0 = ["SLIC","SEEDS","BASS"]
    times0 = [0.1001,0.1737,0.0399]
    bars = ax.bar(names0,times0,color="#D6C0B3",label="Space")
    # names1 = ["TSP","S-GBH","CCS","TCS","BIST"]
    # times1 = [500,400,300,600,40]
    # names1 = ["TSP","S-GBH","BIST"]
    # names1 = ["TSP","S-GBH","BIST"]
    names1 = ["TSP","BIST"]
    times1 = [0.4312,0.0122]
    bars += ax.bar(names1,times1,color="#AB886D",label="Space-Time")
    names = names0 + names1
    times = times0 + times1

    ymin = min(np.min(times0),np.min(times1))
    ymax = max(np.max(times0),np.max(times1))
    ax.set_ylim([0.,ymax*1.4])
    # ax.set_yscale("log")
    ax.set_ylabel("Runtime (sec)",fontsize=12,fontweight="bold")
    ax.set_xticklabels(names0+names1, rotation=45, ha='right')
    for lab in ax.get_xticklabels():
        if lab.get_text() == "BIST":
            lab.set_fontweight('bold')

    # Add text on top of each bar
    ix = 0
    for bar in bars:
        yval = bar.get_height()  # Get the height of each bar
        txt = "%0.4f" % times[ix]
        ax.text(
            bar.get_x() + bar.get_width() / 2,  # X position
            yval,                               # Y position
            txt,
            # f'{yval}',                          # Text to display
            ha='center',                        # Center align the text horizontally
            va='bottom'                         # Align text at the bottom (above the bar)
        )
        ix += 1
    ax.legend(fontsize="10",framealpha=0.,ncols=2)


dpi = 300
ginfo = {'wspace':0.01, 'hspace':0.1,
         "top":0.96,"bottom":0.20,"left":.165,"right":0.99}
fig,ax = plt.subplots(1,1,figsize=(3.5,2.5),gridspec_kw=ginfo,dpi=dpi)
plot_runtime(ax)
plt.savefig("output/run_eval/runtimes.png",transparent=True)



