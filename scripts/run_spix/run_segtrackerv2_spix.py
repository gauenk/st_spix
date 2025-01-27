

import os
import tqdm
import math
import numpy as np
import pandas as pd
from PIL import Image
from pathlib import Path
from einops import rearrange
from easydict import EasyDict as edict

import torch as th
import torchvision.io as tvio

from scipy.io import savemat

from dev_basics.utils.timer import ExpTimer,TimeIt
from dev_basics.utils.gpu_mem import GpuMemer,MemIt

import st_spix
from st_spix.flow_utils import run_raft
from st_spix.prop import stream_bass,indepent_bass

def get_segtrackerv2_videos():
    root = Path("/home/gauenk/Documents/packages/LIBSVXv4.0/Data/SegTrackv2/GroundTruth/")
    vid_names = list([v.name for v in root.iterdir()])
    return vid_names

def get_img_size():
    fn = "/home/gauenk/Documents/packages/LIBSVXv4.0/Data/SegTrackv2/PNGImages/cheetah/00001.png"
    size = Image.open(fn).size
    return size

def get_davis_videos():
    try:
        names = np.loadtxt("/app/in/DAVIS/ImageSets/2017/train-val.txt",dtype=str)
    except:
        fn = "/home/gauenk/Documents/data/davis/DAVIS/ImageSets/2017/train-val.txt"
        names = np.loadtxt(fn,dtype=str)
    return names

def get_spix_function(cfg):

    # -- unpack --
    sp_size = cfg.sp_size
    sigma2_app = cfg.sigma2_app
    sigma2_size = cfg.sigma2_size
    alpha_hastings = cfg.alpha_hastings
    potts = cfg.potts
    sm_start = 0
    niters_seg = 4
    niters = sp_size

    # -- get function --
    # print(cfg.method)
    if cfg.method == "bass":
        print("PLEASE CHANGE ME TO USE THE ORIGINAL BASS.")
        def spix_fxn(vid_lab,fflow):
            spix = indepent_bass(vid_lab,niters=niters,niters_seg=niters_seg,
                                 sp_size=sp_size,sigma2_app=sigma2_app,
                                 alpha_hastings=alpha_hastings,
                                 potts=potts,sm_start=sm_start,rgb2lab=False)
            return spix
    elif cfg.method == "mbass":
        def spix_fxn(vid_lab,fflow):
            spix = indepent_bass(vid_lab,niters=niters,niters_seg=niters_seg,
                                 sp_size=sp_size,sigma2_app=sigma2_app,
                                 alpha_hastings=alpha_hastings,
                                 potts=potts,sm_start=sm_start,rgb2lab=False)
            return spix
    elif cfg.method == "bist":
        def spix_fxn(vid_lab,fflow):
            outs = stream_bass(vid_lab,flow=fflow,
                               niters=niters,niters_seg=niters_seg,
                               sp_size=sp_size,sigma2_app=sigma2_app,
                               sigma2_size=sigma2_size,
                               alpha_hastings=alpha_hastings,
                               potts=potts,sm_start=sm_start,rgb2lab=False)
            return outs[0]
    return spix_fxn

def save_spix_to_csv(save_root,spix):
    # save_root = save_root / ("%s/%s" % (method_name,vid_name))
    if not save_root.exists(): save_root.mkdir(parents=True)
    spix = spix.detach().cpu().numpy()
    for frame_index in range(len(spix)):
        df = pd.DataFrame(spix[frame_index]) # ?.T... no...
        save_fn = save_root / ("%05d.csv" % frame_index)
        df.to_csv(save_fn,header=False,index=False)

def save_compute_stats(save_root,timer,memer):
    save_fn = save_root / "computation.txt"
    stats_txt = str(timer) + "\n" + str(memer)
    with open(save_fn,"w") as f:
        f.write(stats_txt)

def read_video(root,vid_name):

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
        img = tvio.read_image(str(root/fn))/255.
        vid.append(img)
    vid = th.stack(vid).cuda()
    return vid

def check_complete(cfg,vid_name):
    save_root = Path(cfg.save_root) / cfg.method / vid_name
    # print(save_root)
    if not save_root.exists(): save_root.mkdir(parents=True)
    mat_fn = save_root / ("%d.mat"%cfg.k)
    return mat_fn.exists()

def compute_spix(cfg,vid_name):

    # -- check if run --
    if check_complete(cfg,vid_name): return
    # print(cfg.method,cfg.k,vid_name)
    # return
    # exit()

    # -- get video --
    # vid = st_spix.data.davis_example(isize=None,nframes=-1,
    #                                  vid_names=[vid_name],data_set="all")[0]
    vid = read_video(cfg.data_root,vid_name)
    vid_lab = st_spix.utils.vid_rgb2lab_th(vid.clone(),normz=False)

    # -- get function --
    spix_fxn = get_spix_function(cfg)

    # -- compute flow --
    if cfg.method == "bist":
        fflow,bflow = run_raft(th.clip(255.*vid,0.,255.).type(th.uint8))
    else:
        T,F,H,W = vid.shape
        fflow = th.zeros((T,2,H,W),device=vid.device)

    # -- format info --
    vid_lab = rearrange(vid_lab,'b f h w -> b h w f').contiguous()
    fflow = rearrange(fflow,'b f h w -> b h w f').contiguous()

    # -- run --
    timer = ExpTimer()
    memer = GpuMemer()
    with MemIt(memer,"main"):
        with TimeIt(timer,"main"):
            spix = spix_fxn(vid_lab,fflow)

    # print(spix.shape,vid.shape)
    # exit()
    # -- save --
    method_name = cfg.method
    save_root = Path(cfg.save_root) / cfg.method / ("sp%d"%cfg.k) / vid_name
    # print(save_root)
    save_spix_to_csv(save_root,spix)
    save_mat_file(cfg,vid_name,spix.cpu().numpy().astype(np.uint32))
    save_compute_stats(save_root,timer,memer)
    # exit()

def save_mat_file(cfg,vid_name,spix):
    # -- Save frames to the specified .mat file --
    save_root = Path(cfg.save_root) / cfg.method / vid_name
    if not save_root.exists(): save_root.mkdir(parents=True)
    save_name = save_root / ("%d.mat"%cfg.k)
    savemat(save_name, {'svMap': spix})

def main():

    # -- pid --
    print("PID: ",os.getpid())

    # -- main --
    data_root = "/home/gauenk/Documents/packages/LIBSVXv4.0/Data/SegTrackv2/PNGImages/"
    # kgrid = [100,150,200,250,300,350,400,450,500,550,600,800,1000,1200]
    kgrid = [200,250,300,350,400,450,500,550,600,800,1000,1200]
    # cfg = edict({"alpha_hastings":0.,"potts":1.,"sigma2_app":0.01,
    #              "sigma2_size":1.,"save_root":"./output/run_segtrackerv2_spix",
    #              "data_root":data_root})
    cfg = edict({"alpha_hastings":0.,"potts":.5,"sigma2_app":0.01,
                 "sigma2_size":1e5,"save_root":"./output/run_segtrackerv2_spix",
                 "data_root":data_root})
    # methods = ["mbass","st_spix"]
    methods = ["mbass"]
    vid_names = get_segtrackerv2_videos()
    # vid_names = ["tennis"]
    # vid_names = ["monkey"]
    for k in tqdm.tqdm(kgrid,position=0):

        # -- set sp size --
        size = get_img_size()
        npix = size[0]*size[1]
        sp_size = int(round(math.sqrt(npix / (1.*k))))
        if sp_size > 50: continue
        cfg.sp_size = sp_size
        cfg.k = k

        # -- run for davis --
        for vid_name in tqdm.tqdm(vid_names,position=1,leave=False):
            for method in tqdm.tqdm(methods,position=2,leave=False):
                cfg.method = method

                # -- compute spix --
                compute_spix(cfg,vid_name)


if __name__ == "__main__":
    main()
