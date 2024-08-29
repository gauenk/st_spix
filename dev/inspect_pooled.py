
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

    # -- init --
    root = Path("./output/inspect_pooled")
    if not root.exists(): root.mkdir()

    # -- get data --
    vid = st_spix.data.davis_example(isize=None,nframes=10)[0,:10,:,:480,:480]
    # vid = resize(vid,(64,64))
    T,F,H,W = vid.shape
    spix,fflow = stream_bass(vid,sp_size=15,beta=5.)
    vid = vid.double().requires_grad_(True)

    # -- run pooling --
    pooled,down = pooling(vid,spix)
    print("vid.shape: ",vid.shape)
    print("pooled.shape: ",pooled.shape)
    tv_utils.save_image(pooled,root / "pooled.png")



if __name__ == "__main__":
    main()
