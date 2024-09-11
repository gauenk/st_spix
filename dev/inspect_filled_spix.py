# import torch as th
# import torchvision.utils as tv_utils
# from pathlib import Path

import os
import torch as th
import numpy as np
import pandas as pd
from pathlib import Path
import st_spix
from st_spix.spix_utils import img4bass,mark_spix
from st_spix import flow_utils
import st_spix_cuda
import st_spix_prop_cuda
from st_spix import flow_utils as futils
import torchvision.io as iio
from einops import rearrange,repeat
from skimage.segmentation import mark_boundaries
import torchvision.utils as tv_utils
import torch.nn.functional as th_f

import seaborn as sns
import matplotlib.pyplot as plt
from dev_basics.utils.metrics import compute_psnrs

import stnls

from dev_basics import flow as flow_pkg
from dev_basics.utils.timer import ExpTimer,TimeIt

from easydict import EasyDict as edict


# -- colorwheel --
import matplotlib.pyplot as plt
import matplotlib as mpl
from matplotlib.colors import LinearSegmentedColormap, ListedColormap


def save_pair(spix0,spix1,spid,root):
    _spix0 = 1.*(spix0 == spid)
    _spix1 = 1.*(spix1 == spid)
    grid = tv_utils.make_grid([_spix0,_spix1])[:,None]
    print(grid.shape)
    tv_utils.save_image(grid,root / ("spix_%d.png"%spid))

def shift_labels(spix,means,flow):

    # -- unpack --
    B,H,W = spix.shape
    flow = flow.clone()

    # -- scatter --
    grid = futils.index_grid(H,W,dtype=spix.dtype,
                             device=spix.device,normalize=True)
    grid = st_spix.sp_pool_from_spix(grid,spix)
    gscatter,gcnts = st_spix.scatter.run(grid,flow,swap_c=True)

    # -- invalidate --
    eps = 1e-13
    invalid = th.where(th.logical_or(gcnts<1-eps,1+eps<gcnts))
    for i in range(2):
        gscatter[:,i][invalid] = -100.

    # -- all pairwise differences --
    locs = th.stack([means[:,-2]/(W-1),means[:,-1]/(H-1)],-1)
    gscatter = rearrange(gscatter,'b f h w -> (b h w) f')
    dists = th.cdist(gscatter,locs)

    # -- gather --
    spix_grid = th.arange(means.shape[0]).to(spix.device) #  I think "0" in invalid?
    shifted_spix = spix_grid[dists.argmin(1)].int()
    # print(shifted_spix.min() >= -1)
    # print("shaped: ",means.shape,flow.shape,shifted_spix.shape)
    shifted_spix = rearrange(shifted_spix,'(b h w) -> b h w',h=H,w=W)
    shifted_spix[invalid] = -1

    return shifted_spix,gcnts

def run_scatter(img,spix,flow,means,cov,counts,ids,
                niters,inner_niters,npix_in_side,i_std,
                alpha,beta):

    flow_sp,_means = st_spix.pool_flow_and_shift_mean(flow,means.clone(),spix,ids)
    spix_s,cnts = shift_labels(spix.clone(),means[0],flow_sp) # propogate labels
    return spix_s,cnts

def run_inspect_scatter():

    # -- config --
    npix_in_side = 80
    niters,inner_niters = 1,40
    # i_std,alpha,beta = 0.018,20.,100.
    i_std,alpha,beta = 0.1,0.001,100.

    # -- load images --
    vid = st_spix.data.davis_example(isize=None,nframes=10)[:1,:10,:,:480,:480]
    # vid = vid + (25./255.)*th.randn_like(vid)
    vid = th.clip(255.*vid,0.,255.).type(th.uint8)
    B,T,F,H,W = vid.shape

    # -- bass --
    img0 = img4bass(vid[:,0])
    bass_fwd = st_spix_cuda.bass_forward
    # spix0,means,cov,counts,ids = bass_fwd(img0,npix_in_side,i_std,alpha,beta)
    spix0,means,cov,counts,ids = bass_fwd(img0,npix_in_side,i_std,alpha,beta)

    # -- flow and ids --
    flows = flow_pkg.run(vid/255.,sigma=0.0,ftype="cv2")
    flow = flows.fflow[0,0][None,:]
    ids = ids.unsqueeze(1).expand(-1, means.size(-1)).long()[None,:]

    # -- run scatter --
    img_curr = img4bass(vid[:,1])
    flow_curr = flows.fflow[0,0][None,:]
    spix_s,cnts = run_scatter(img_curr,spix0,flow_curr,
                              means,cov,counts,ids,niters,
                              inner_niters,npix_in_side,i_std,alpha,beta)

    return img0[0],img_curr[0],spix0[0],spix_s[0],cnts[0]

def inspect_prop_seg_run(root):
    spix0 = th.load("output/prop_seg/spix0_nr.pth")[0]
    spix1 = th.load("output/prop_seg/spix1_nr.pth")[0]
    print("-- viz --")

    # print(spix0[150:168,194:204+4])
    # print(spix1[165:184,212:220+4])
    print(spix0[167:184,65:75])
    print(spix1[167:184,65:75])
    # exit()
    # print(spix0[115:125,155:165])
    # print(spix1[125:135,175:185])
    # print(spix0[190:200,275:285])


    spid0 = th.mode(spix0[150:168,194:204+4]).values[0].item()
    spid1 = th.mode(spix0[115:125,155:165]).values[0].item()
    spid2 = th.mode(spix0[190:200,275:285]).values[0].item()
    spid3 = th.mode(spix0[167:184,65:75]).values[0].item()
    save_pair(spix0,spix1,spid0,root)
    save_pair(spix0,spix1,spid1,root)
    save_pair(spix0,spix1,spid2,root)
    save_pair(spix0,spix1,200,root)
    save_pair(spix0,spix1,spid3,root)
    exit()

def to_np(tensor):
    return tensor.cpu().numpy()
def to_th(tensor):
    return th.from_numpy(tensor)
def swap_c0(img):
    return rearrange(img,'... h w f -> ... f h w')
def swap_c3(img):
    return rearrange(img,'... f h w -> ... h w f')
# def mark_spix(img,spix):
#     marked = to_th(swap_c0(mark_boundaries(to_np(img),to_np(spix))))
#     args = th.where(spix==-1)
#     marked[0][args] = 0
#     marked[1][args] = 0
#     marked[2][args] = 1.
#     return marked

def main():
    root = Path("output/inspect_filled_spix")
    if not root.exists():
        root.mkdir(parents=True)

    # -- inspect output from dev/prop_seg/ --
    inspect_prop_seg_run(root)

    # -- inspect --
    img0,img1,spix0,spix1,cnts = run_inspect_scatter()
    print(img0.shape)
    print(img1.shape)
    print(spix0.shape)
    print(spix1.shape)
    print(cnts.shape)
    print(spix0[150:168,194:204+4])
    print(spix1[165:184,212:220+4])
    # mark0 = swap_c0(mark_boundaries(swap_c3(to_np(img0)),to_np(spix0)))
    mark0 = mark_spix(img0,spix0)
    mark1 = mark_spix(img1,spix1)
    # mark1 = mark_boundaries(img1.cpu().numpy(),spix1.cpu().numpy())
    tv_utils.save_image(mark0,root / "mark0.png")
    tv_utils.save_image(mark1,root / "mark1.png")

    # _spix0 = 1.*(spix0 == 236)
    # _spix1 = 1.*(spix1 == 236)
    # tv_utils.save_image(_spix0/_spix0.max(),root / "spix0_236.png")
    # tv_utils.save_image(_spix1/_spix1.max(),root / "spix1_236.png")


if __name__ == "__main__":
    main()
