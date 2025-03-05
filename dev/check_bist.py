
import torch as th
from einops import rearrange
import bist_cuda
import torchvision.utils as tv_utils

import math
from pathlib import Path

import st_spix
from st_spix import flow_utils
from st_spix.spix_utils import mark_spix_vid
from st_spix.flow_utils import run_raft,run_spynet

def main():

    # -- setup --
    device = "cuda:0"
    sp_size = 25
    # potts = 10.
    potts = 20.
    # sigma2_app = math.pow(0.009,2)
    sigma2_app = math.pow(0.021,2)
    # alpha = math.log(0.5)
    alpha = math.log(0.1)
    niters = sp_size
    video_mode = False

    # -- vid,fflow --
    vid = st_spix.data.davis_example(isize=None,nframes=10)[0,:8]
    vid = vid.to(device)
    fflow,bflow = run_raft(th.clip(255.*vid,0.,255.).type(th.uint8))
    # fflow,bflow = run_spynet(th.clip(1.*vid,0.,1.))
    # fflow = th.stack([fflow[:,1],fflow[:,0]],1)

    # -- get spix --
    _vid = rearrange(vid,'t c h w -> t h w c').contiguous()
    _fflow = rearrange(fflow,'t c h w -> t h w c').contiguous()
    spix = bist_cuda.bist_forward(_vid,fflow,sp_size,niters,potts,
                                  sigma2_app,alpha,video_mode)
    print(spix.shape)
    # marked = mark_spix_vid(vid,spix)
    marked = bist_cuda.get_marked_video(_vid,spix)
    print(marked.shape)
    marked = rearrange(marked,'t h w c -> t c h w').contiguous()


    # -- .. --
    substr = "bist" if video_mode else "bass"
    fn = Path("output/check_bist/marked_%s.png"%substr)
    if not fn.parents[0].exists(): fn.parents[0].mkdir()
    tv_utils.save_image(marked,fn)

if __name__ == "__main__":
    main()
