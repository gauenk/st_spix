"""

   Read videos, segmentations, and superpixels related to the SegTrackv2 dataset

"""

import os
import tqdm
import h5py
import numpy as np
import pandas as pd
from pathlib import Path
from PIL import Image
from einops import rearrange

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


def get_group_root(group):
    root = Path("/home/gauenk/Documents/packages/")
    if group == "stspix":
        base = root/"st_spix/output/run_segtrackerv2_spix/"
    elif group == "spix-bench":
        base = root/"superpixel-benchmark/docker/out/segtrackerv2/"
    elif group == "libsvx":
        base = root/"LIBSVXv4.0/Results/SegTrackv2/"
    elif group == "gbass":
        base = root/"BASS_check/result/"
    else:
        raise ValueError(f"unknown group [{group}]")
    return base

def get_sp_grid(group,root,method):
    if group == "spix-bench":
        # path = root/"superpixel-benchmark/docker/out/segtrackerv2/"/method
        path = root / method
        ids = [int(str(f.name).split("sp")[0]) for f in path.iterdir()]
        return ids
    elif group == "libsvx":
        vname = "birdfall"
        path = root / method / "Segments" / vname
        print(path)
        check = lambda f: str(f.name).endswith(".mat")
        proc = lambda f: int(str(f.name).split(".")[0])
        ids = [proc(f) for f in path.iterdir() if check(f)]
        return ids
    elif group == "stspix":
        vname = "birdfall"
        path = root / method
        # path = root /"output/run_segtrackerv2_spix/"/method
        # check = lambda f: str(f.name).endswith("sp")
        check = lambda f: str(f.name).startswith("sp")
        proc = lambda f: int(str(f.name).split("sp")[1])
        ids = [proc(f) for f in path.iterdir() if check(f)]
        return ids
    elif group == "gbass":
        vname = "cheetah"
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
    elif group == "gbass":
        return read_csv(root / method / ("sp%d"%sp) / vname,nframes,1)
    else:
        raise ValueError("")
