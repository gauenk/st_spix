

import os,glob
import torch as th
import numpy as np
import pandas as pd
from pathlib import Path
import st_spix
from st_spix import flow_utils
from st_spix.spix_utils import img4bass,mark_spix
import st_spix_cuda
import st_spix_prop_cuda
from st_spix import flow_utils as futils
import torchvision.io as iio
from einops import rearrange,repeat
from skimage.segmentation import mark_boundaries
import torchvision.utils as tv_utils
import torch.nn.functional as th_f
from torchvision.utils import draw_segmentation_masks

import seaborn as sns
import matplotlib as mpl
import matplotlib.pyplot as plt
from dev_basics.utils.metrics import compute_psnrs

try:
    import stnls
except:
    pass

from dev_basics import flow as flow_pkg
from dev_basics.utils.timer import ExpTimer,TimeIt
from torchvision.transforms.functional import resize

from st_spix.prop_seg import *

from easydict import EasyDict as edict

def main():

    # -- get root --
    print("PID: ",os.getpid())
    # root = Path("./output/check_bass_connected/default")
    # root = Path("./output/check_bass_connected/update")
    root = Path("./output/check_bass_connected/nomerge")
    if not root.exists(): root.mkdir()
    device = "cuda"

    # -- read images run on BASS --
    # names = list(glob.glob("../BASS_check2/images/*"))
    names = list(glob.glob("../BASS_check2/result/*csv"))
    names = [name.split("/")[-1].split(".")[0] for name in names]
    print(names)
    for name in names:

        # --------------------------------------
        #     Read files run using BASS
        # --------------------------------------

        print(name)
        # img0 = iio.read_image("../BASS_check/images/%s.jpg"%name)/255.
        spix = pd.read_csv("../BASS_check2/result/%s.csv"%name,header=None)
        spix = th.tensor(spix.values)[None,:].int().to(device)
        # img0 = img0.contiguous()
        spix = spix.contiguous()
        nspix = int(spix.max())+1

        # --------------------------------------
        #     Split Disconnected Superpixels
        # --------------------------------------

        spix_split = spix.clone()
        fxn = st_spix_prop_cuda.split_disconnected
        spix_split,children,split_starts = fxn(spix_split,nspix)

        # ----------------------------------------------------------
        #        Spotcheck
        # ----------------------------------------------------------

        # print(spix[0,:10,:10])
        # print(spix[0,10:20,:10])
        # print(spix[0,:10,10:20])
        # print(spix_split[0,:10,10:20])
        # print(children.shape)
        # print(th.where(spix[0] != spix_split[0]))
        split_df = pd.DataFrame(spix_split[0].cpu().numpy())
        split_df.to_csv(root/("%s.csv"%name))
        # spix = pd.read_csv("../BASS_check/result/%s.csv"%name,header=None)

        # print(spix_split.shape)
        # print(children)
        # print(split_starts)
        # print(spix_split.max().item(),spix.max().item())

        # ----------------------------------------------------------
        #     Vizualize Split Superpixels with Difference Colors
        # ----------------------------------------------------------

        if children.shape[1] == 0: spix_ids = [0]
        else: spix_ids = th.where(children[:,0]>0)[0]
        nsplit = len(spix_ids)
        print("nsplit: ",nsplit,spix_ids[:5])
        # print(spix_ids)
        # print(children[spix_ids[0]])
        # print(th.where(spix_split==children[spix_ids[0]][0]))


        # viridis = mpl.colormaps['coolwarm'].resampled(nsplit)
        # seg0 = img0.clone()
        # show_childs = img0.clone()
        # draw_fxn = draw_segmentation_masks
        # for ix,spix_id in enumerate(spix_ids):

        #     # -- view split spix --
        #     colors = [list(255*a for a in viridis(ix/(1.*nsplit))[:3])]
        #     mask0 = spix == spix_id
        #     alpha = 0.8
        #     seg0 = draw_fxn(seg0,mask0[0],alpha=alpha,colors=colors)

        #     # -- view all the children indices --
        #     # mask = spix_split==children[spix_id][0]
        #     # for i in range(1,children.shape[1]):
        #     #     if children[spix_id][0]==-1: break
        #     #     mask = th.logical_or(mask,spix_split==children[spix_id][i])
        #     # show_childs = draw_fxn(show_childs,mask,alpha=alpha,colors=colors)
        #     # if ix > 10:
        #     #     break

        # # -- view differences --
        # colors = ["red"]
        # mask = (spix_split == 104) != (spix == 104)
        # deltas = draw_fxn(img0.clone(),mask,alpha=0.9,colors=colors)
        # mask = th.logical_and((spix_split == 104),(spix == 104))
        # deltas = draw_fxn(deltas,mask,alpha=0.9,colors=["blue"])

        # # -- save --
        # print(f"Saving at {root}")
        # tv_utils.save_image(seg0[None,],root / ("seg_%s.png"%name))
        # tv_utils.save_image(deltas[None,],root / ("delta_%s.png"%name))
        # # tv_utils.save_image(show_childs[None,],root / ("childs_%s.png"%name))

if __name__ == "__main__":
    main()
