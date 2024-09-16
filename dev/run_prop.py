"""

      Execute the Algorithm's Pipeline

"""

import torch as th
import numpy as np
from einops import rearrange,repeat
from pathlib import Path
from functools import reduce


# -- masked tensors --
from torch.masked import masked_tensor, as_masked_tensor
import warnings
warnings.filterwarnings(action='ignore', category=UserWarning)


from st_spix.spix_utils import mark_spix_vid,img4bass
import torchvision.utils as tv_utils

import st_spix
from st_spix import flow_utils as futils

from torchvision.transforms.functional import resize

from st_spix import scatter
from st_spix import deform
from st_spix.sp_pooling import pooling,SuperpixelPooling

import stnls
from dev_basics import flow as flow_pkg

import matplotlib as mpl
import matplotlib.cm as cm
from matplotlib import colormaps
from matplotlib import patches, pyplot as plt
# import matplotlib.pyplot as plt

from st_spix.prop import stream_bass

def draw_spix_vid(vid,spix):
    viz_seg = []
    nspix = spix.max().item()+1
    for t in range(vid.shape[0]):
        viz_seg.append(draw_spix(vid[t],spix[t],nspix))
    viz_seg = th.stack(viz_seg)
    return viz_seg/255.

def draw_spix(img,spix,nspix):
    masks = th.nn.functional.one_hot(spix.long(),num_classes=nspix).bool()
    masks = masks.permute(2,0,1)
    # nspix = spix.max().item()+1
    viridis = mpl.colormaps['tab20'].resampled(nspix)
    scolors = [list(255*a for a in viridis(ix/(1.*nspix))[:3]) for ix in range(nspix)]
    print(img.min(),img.max())
    img = th.clip(255*img,0.,255.).type(th.uint8)
    # print(img.shape,masks.shape)
    # print(masks[0])
    marked = tv_utils.draw_segmentation_masks(img,masks,colors=scolors)
    return marked

def main():

    # -- get root --
    root = Path("./output/run_prop/")
    if not root.exists(): root.mkdir()

    # -- config --
    niters = 40
    niters_seg = 4
    sp_size = 12
    alpha,potts = 10.,10.
    pix_cov = 0.1

    # -- read img/flow --
    vid = st_spix.data.davis_example(isize=None,nframes=10,vid_names=['tennis'])
    # vid = st_spix.data.davis_example(isize=None,nframes=10,vid_names=['baseball'])
    size = 256
    # vid = vid[0,5:7,:,50:50+size,300:300+size]
    vid = vid[0,2:4,:,50:50+size,200:200+size]
    vid = resize(vid,(128,128))

    # -- run flow [raft] --
    from st_spix.flow_utils import run_raft
    fflow,bflow = run_raft(th.clip(255.*vid,0.,255.).type(th.uint8))
    # print(vid.shape,fflow.shape)
    # fflow,bflow = run_raft(vid)
    if fflow.shape[-1] != vid.shape[-1]:
        print("RAFT wants image size to be a multiple of 8.")
        exit()

    # -- resize again --
    # vid = resize(vid,(64,64))
    # fflow = resize(fflow,(64,64))/2. # reduced scale by 2
    size = 128
    vid = resize(vid,(size,size))
    fflow = resize(fflow,(size,size))/(128./size) # reduced scale by X


    # -- save --
    B,F,H,W = vid.shape
    tv_utils.save_image(vid,root / "vid.png")

    # -- propogate --
    spix,children = stream_bass(vid,flow=fflow,niters=niters,niters_seg=niters_seg,
                                sp_size=sp_size,pix_cov=pix_cov,alpha=alpha,potts=potts)
    # -- view --
    marked = mark_spix_vid(vid,spix)
    marked[1,0][th.where(spix[1]<0)] = 0.
    marked[1,1][th.where(spix[1]<0)] = 1.
    marked[1,2][th.where(spix[1]<0)] = 0.
    marked[1,0][th.where(spix[1]==60)] = 1.
    marked[1,1][th.where(spix[1]==60)] = 0.
    marked[1,2][th.where(spix[1]==60)] = 0.

    # -- save --
    print("saving images.")
    viz_seg = draw_spix_vid(vid,spix)
    futils.viz_flow_quiver(root/"flow.png",fflow[[0]],step=2)
    tv_utils.save_image(marked,root / "marked_fill.png")
    tv_utils.save_image(viz_seg,root / "viz_seg.png")


if __name__ == "__main__":
    main()
