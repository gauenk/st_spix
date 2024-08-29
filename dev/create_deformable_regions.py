
import torch as th
import numpy as np
from einops import rearrange
from pathlib import Path


from st_spix.spix_utils import mark_spix_vid,img4bass
import torchvision.utils as tv_utils

import st_spix
from st_spix.prop_seg import stream_bass

from torchvision.transforms.functional import resize

import st_spix_cuda
from st_spix import scatter
from st_spix.sp_pooling import pooling,SuperpixelPooling

import stnls
from dev_basics import flow as flow_pkg

def run_stnls(nvid,acc_flows,ws):
    wt,ps,s0,s1,full_ws=1,1,1,1,False
    k = 1
    search_p = stnls.search.PairedSearch(ws,ps,k,
                                         nheads=1,dist_type="l2",
                                         stride0=s0,stride1=s1,
                                         self_action=None,use_adj=False,
                                         full_ws=full_ws,itype="float")
    _,flows = search_p.paired_vids(nvid,nvid,acc_flows,wt,skip_self=True)
    return flows

def get_scattering_field(spix,R):

    # -- unpack --
    shape = spix.shape
    B = spix.shape[0]
    spix = spix.reshape(B,1,-1)
    npix = spix.shape[-1]

    # -- find matches --
    ids = th.arange(spix.max()+1)[None,:,None].to(spix.device)
    match = spix == ids

    # -- increment along pixels to get index within superpixel --
    csum = th.cumsum(match,-1)-1

    # -- fill index with superpixel coordinate within superpixel --
    inds = th.where(match)
    sinds = -th.ones((B,npix),device=spix.device,dtype=th.long)
    sinds[inds[0],inds[2]] = csum[inds[0],inds[1],inds[2]]
    assert (th.max(sinds) == (R-1))

    # -- offset sinds with superpixel label offset --
    sinds = sinds + spix.reshape(B,-1)*R
    return sinds

def main():

    # -- get root --
    root = Path("./output/create_deformable_regions/")
    if not root.exists(): root.mkdir()

    # -- config --
    npix_in_side = 40
    niters,inner_niters = 1,2
    i_std,alpha,beta = 0.1,0.1,10.

    # -- read img/flow --
    vid = st_spix.data.davis_example(isize=None,nframes=10,vid_names=['tennis'])
    vid = vid[0,3:6,:,:480,:480]
    vid = resize(vid,(256,256))
    flows = flow_pkg.run(vid[None,:],sigma=0.0,ftype="cv2")
    B,F,H,W = vid.shape

    # -- view --
    spix,fflow = stream_bass(vid,sp_size=20,beta=5.)
    B = spix.shape[0]

    # -- save --
    marked = mark_spix_vid(vid,spix)
    tv_utils.save_image(marked,root / "marked.png")

    # -- get largest superpixel size --
    mode = th.mode(spix.reshape(B,-1)).values[:,None]
    largest_spix = th.max(th.sum(spix.reshape(B,-1) == mode,-1)).item()
    R = largest_spix # "R" for "Radius"
    nspix = int(spix.max()+1)

    # -=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-
    #
    #      From Video to Superpixels
    #
    # -=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-

    # -- fill into contiguous tensor [very very silly] --
    inds = get_scattering_field(spix,R)
    regions = th.zeros((B,nspix*R,F),device=spix.device)
    print(inds.shape,regions.shape,vid.shape)
    vid_r = rearrange(vid,'b f h w -> b (h w) f')
    regions = regions.scatter_(1,inds[...,None],vid_r)
    regions = regions.reshape(B,nspix,R,F)
    print(regions.shape)
    th.save(regions,"regions.pth")

    # -=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-
    #
    #      From Superpixels to Video
    #
    # -=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-

    print(inds)
    print(th.min(inds),th.max(inds))


    # exit()
    # # inds.shape = (B,HW,F)
    # # max(inds) = nspix * R; min(inds) = 0
    # # regions[batch_ix,inds[batch_ix,hw_ix,ftr_ix],ftr_ix]
    # #        = vid[batch_ix,hw_ix,ftr_ix]
    # regions = regions.reshape(B,nspix,R,F)

if __name__ == "__main__":
    main()
