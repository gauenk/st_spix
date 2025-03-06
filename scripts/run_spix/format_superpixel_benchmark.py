"""

   Format the results so they can be
   evaulated by the `superpixel_benchmark` repo

"""

import os,math
import subprocess
import numpy as np
import torch as th
from pathlib import Path

def check_davis_complete(dname,root):
    fn = "/home/gauenk/Documents/data/davis/DAVIS/ImageSets/2017/val.txt"
    names = np.loadtxt(fn,dtype=str)
    N = len(names)
    Ncurr = len(list(root.iterdir()))
    return N == Ncurr

def check_segtracker_complete(dname,root):
    root = Path("/home/gauenk/Documents/packages/LIBSVXv4.0/Data/SegTrackv2/GroundTruth/")
    vid_names = list([v.name for v in root.iterdir()])
    N = len(vid_names)
    Ncurr = len(list(root.iterdir()))
    return N == Ncurr

def check_complete(dname,root):
    if dname == "davis":
        return check_davis_complete(dname,root)
    elif dname == "segtrackerv2":
        return check_segtracker_complete(dname,root)
    else:
        raise NotImplemented("")

def shift_segtracker_names():
    # starts @ 0 but should be 1.

    # dnames = ["davis","segtrackerv2"]
    dnames = ["segtrackerv2"]
    methods = ["mbass","st_spix"]
    prefix = "sp"
    # kgrid = [100,150,200,250,300,350,400,450,500,550,600,800,1000,1200]
    kgrid = [200,250,300,350,400,450,500,550,600,800,1000,1200]

    out0 = Path("/home/gauenk/Documents/packages/st_spix/output/")
    out1 = Path("/home/gauenk/Documents/packages/superpixel-benchmark/docker/out")

    for dname in dnames:
        for method in methods:
            for k in kgrid:
                src = out0 / ("run_%s_spix" % dname) / method / ("sp%d"%k)
                if not src.exists(): continue
                dest = out1 / dname / method / ("sp%d"%k)
                if not dest.exists(): continue
                assert(check_complete(dname,src))
                for vid_dir in dest.iterdir():
                    files = {int(f.stem):f.name for f in vid_dir.iterdir() if str(f).endswith(".csv")}
                    files = [x[1] for x in sorted(files.items(), key=lambda x: x[0])]
                    f0_id = int(files[0].split(".")[0])
                    # if f0_id == 0:
                    #     print(vid_dir/files[0])
                    #     cmd = "rm %s" % (vid_dir/files[0])
                    #     log = subprocess.run(cmd,shell=True,capture_output=True,text=True).stdout
                    #     # exit()
                    if f0_id == 1: continue
                    N = len(files)
                    for ix,fname in enumerate(reversed(files)):
                        new_fname = "%05d.csv" % (N - ix)
                        cmd = "mv %s %s" % (vid_dir/fname,vid_dir/new_fname)
                        # print(cmd)
                        log = subprocess.run(cmd,shell=True,capture_output=True,text=True).stdout

def main():
    # dnames = ["davis","segtrackerv2"]
    dnames = ["segtrackerv2"]
    methods = ["mbass","st_spix"]
    prefix = "sp"
    # kgrid = [100,150,200,250,300,350,400,450,500,550,600,800,1000,1200]
    kgrid = [200,250,300,350,400,450,500,550,600,800,1000,1200]

    out0 = Path("/home/gauenk/Documents/packages/st_spix/output/")
    out1 = Path("/home/gauenk/Documents/packages/superpixel-benchmark/docker/out")

    for dname in dnames:
        for method in methods:
            for k in kgrid:
                src = out0 / ("run_%s_spix" % dname) / method / ("sp%d"%k)
                if not src.exists(): continue
                dest = out1 / dname / method / ("sp%d"%k)
                if dest.exists(): continue
                if not dest.parents[0].exists():
                    dest.parents[0].mkdir(parents=True)
                # print(src,dest)
                if not check_complete(dname,src): continue
                print(src,dest)
                cmd = "cp -r %s %s" % (src,dest)
                log = subprocess.run(cmd,shell=True,capture_output=True,text=True).stdout



if __name__ == "__main__":
    # main()
    shift_segtracker_names()
