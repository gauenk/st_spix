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
from st_spix.prop_seg import stream_bass

from torchvision.transforms.functional import resize

import st_spix_cuda
from st_spix import scatter
from st_spix import deform
from st_spix.sp_pooling import pooling,SuperpixelPooling

import stnls
from dev_basics import flow as flow_pkg

import matplotlib.cm as cm
from matplotlib import colormaps
from matplotlib import patches, pyplot as plt
# import matplotlib.pyplot as plt

def main():


    # -- get root --
    root = Path("./output/run_deform/")
    if not root.exists(): root.mkdir()

    # -- config --
    sp_size = 15
    nrefine = 20
    niters,inner_niters = 1,1
    i_std,alpha,beta = 0.1,1.,10.

    # -- read img/flow --
    vid = st_spix.data.davis_example(isize=None,nframes=10,vid_names=['tennis'])
    vid = vid[0,3:6,:,:128,290-128:290]

    # -- view --
    spix,fflow = stream_bass(vid,sp_size=sp_size,alpha=alpha,
                             beta=beta,nrefine=nrefine,fflow=fflow)
    B = spix.shape[0]
    th.cuda.empty_cache()

    # -- get deform --
    deform_vals,deform_inds = deform.get_deformation(vid,spix,centers,covs,R)

    # -- deform video --
    warped = deform.deform_vid(vid,vals,inds)

    # -- save --
    tv_utils.save_image(marked,root / "warped.png")


if __name__ == "__main__":
    main()
