"""

   Make some cool viz of the superpixels

"""

from pathlib import Path

import numpy as np
import torch as th
from einops import rearrange,repeat
import torchvision.utils as tv_utils

from st_spix.spix_utils import mark_spix_vid,connected_sp
from st_spix.spix_utils.spix_io import read_video,read_seg,read_spix,get_group_root
from st_spix.spix_utils.spix_io import get_segtrackerv2_videos,get_sp_grid

def get_viz_parameter(method):
    if method == "st_spix":
        return 500
    elif method == "TSP":
        return 500
    elif method == "streamGBH":
        return 10
    elif method == "mbass":
        return 500
    elif method == "bass":
        return 20
    elif method == "ers":
        return 3
    else:
        raise ValueError("")

def get_spix_videos(vname):

    # -- config --
    methods = ["st_spix","TSP","streamGBH",
               # "mbass",
               "bass",
               "ers","etps","seeds","slic"]
    m2groups = {"st_spix":"stspix","mbass":"stspix",
                "TSP":"libsvx","streamGBH":"libsvx",
                "ers":"spix-bench","etps":"spix-bench",
                "seeds":"spix-bench","slic":"spix-bench","bass":"gbass"}
    # methods = ["st_spix","TSP","streamGBH","mbass"]
    methods = ["st_spix","TSP","streamGBH","bass","ers"]

    # -- get video --
    vid = read_video(vname)
    nframes = vid.shape[0]
    vid = rearrange(vid,'t h w f -> t f h w')

    method = "streamGBH"
    group = m2groups[method]
    base = get_group_root(group)

    # -- dev --
    # spgrid = get_sp_grid(group,base,method)
    # for sp in sorted(spgrid):
    #     # spix = read_spix(group,base,method,vname,sp,nframes)
    #     mvname = "old_"+vname
    #     spix = read_spix(group,base,method,mvname,sp,nframes)
    #     print(sp,len(np.unique(spix)))
    # # if method == "streamGBH":
    # #     print(spix)
    # #     print(len(np.unique(spix)))
    # #     print(np.unique(spix,return_counts=True))
    # #     exit()
    # exit()

    marked = []
    for method in methods:
        group = m2groups[method]
        base = get_group_root(group)
        # spgrid = get_sp_grid(group,base,method)
        # print(method,spgrid)
        sp = get_viz_parameter(method)
        spix = read_spix(group,base,method,vname,sp,nframes)
        if method == "streamGBH":
            spix = connected_sp(spix)
        # print(vid.shape,spix.shape)
        marked_m = mark_spix_vid(vid,spix)
        marked.append(marked_m)
    marked = th.stack(marked)
    return marked,methods

def main():

    # -- read spix videos --
    root = Path("output/run_spix/viz_results/")
    if not root.exists(): root.mkdir(parents=True)

    # -- get marked videos --
    vnames = get_segtrackerv2_videos()
    # vname = vnames[0]
    # vname = "worm_3"
    vname = "cheetah"
    # vname = "monkey"
    # vname = "penguin"
    marked,names = get_spix_videos(vname)
    out_dir = root / vname
    if not out_dir.exists(): out_dir.mkdir(parents=True)

    # -- save --
    for ix,name in enumerate(names):
        tv_utils.save_image(marked[ix,-4:],out_dir / ("%s.png"%name))


if __name__ == "__main__":
    main()
