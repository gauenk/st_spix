
import torch as th
import numpy as np
from einops import rearrange

from pathlib import Path



from st_spix.spix_utils import mark_spix_vid
import torchvision.utils as tv_utils

import st_spix
from st_spix.prop_seg import stream_bass
from st_spix.sp_pooling import pooling

from torch.autograd import gradcheck

from torchvision.transforms.functional import resize

def main():

    # -- get data --
    vid = st_spix.data.davis_example(isize=None,nframes=10)[0,:1,:,:480,:480]
    vid = resize(vid,(64,64))
    T,F,H,W = vid.shape
    spix,fflow = stream_bass(vid,sp_size=15,beta=5.)
    vid = vid.double().requires_grad_(True)

    # -- run grad check @ 1 --
    # vid = rearrange(vid,'b f h w -> b h w f').contiguous()
    pooling_partial = lambda tensor: pooling(tensor,spix)[0]
    test = gradcheck(pooling_partial, vid, eps=1e-6, atol=1e-4)
    print(test)

    # -- run grad check @ 0 --
    pooling_partial = lambda tensor: pooling(tensor,spix)[1]
    test = gradcheck(pooling_partial, vid, eps=1e-6, atol=1e-4)
    print(test)



if __name__ == "__main__":
    main()
