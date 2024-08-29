
import torch as th
import numpy as np
from pathlib import Path

from st_spix.spix_utils import mark_spix_vid,img4bass
import torchvision.utils as tv_utils

import st_spix
from st_spix.prop_seg import stream_bass

from torchvision.transforms.functional import resize

import st_spix_cuda
from st_spix import scatter
from dev_basics import flow as flow_pkg
from st_spix.sp_pooling import pooling,SuperpixelPooling


def main():

    # -- get root --
    root = Path("./output/inspect_counts")
    if not root.exists(): root.mkdir()

    # -- config --
    npix_in_side = 15
    niters,inner_niters = 1,30
    i_std,alpha,beta = 0.1,0.1,10.

    # -- counts --
    # cnts = th.load("cnts.pth") # load from anim_shift.py
    # print(cnts.shape)
    # block = cnts[0,125:135,130:140]
    # print(cnts[0,125:135,130:140])
    # print(block[0,0].item())
    # print(cnts[0,125:135,130:140]>1)
    # print(cnts[0,125:135,130:140]<1)

    # -- read img/flow --
    vid = st_spix.data.davis_example(isize=None,nframes=10,vid_names=['kid-football'])
    vid = vid[0,:2,:,:480,:480]
    vid = vid[...,25:130,220:220+105]
    # vid[:,0] = 0.
    flows = flow_pkg.run(vid[None,:],sigma=0.0,ftype="cv2")
    fflow = 2*flows.fflow[0,0][None,:]
    bflow = flows.bflow[0,1][None,:]
    print(fflow.abs().mean(),bflow.abs().mean())

    # -- bass --
    img0 = img4bass(vid[None,0])
    bass_fwd = st_spix_cuda.bass_forward
    spix,means,cov,counts,ids = bass_fwd(img0,npix_in_side,i_std,alpha,beta)
    ids = ids.unsqueeze(1).expand(-1, means.size(-1)).long()[None,:]

    # -- save --
    marked = mark_spix_vid(vid,spix)
    tv_utils.save_image(marked,root / "marked.png")

    # -- get flow --
    fflow_sp_v0,_ = pooling(fflow,spix)
    bflow_sp_v0,_ = pooling(bflow,spix)
    _pooling = SuperpixelPooling.apply
    fflow_sp,fflow_ds = _pooling(fflow,spix)
    bflow_sp,bflow_ds = _pooling(bflow,spix)

    # -- view --
    # print("fflow_sp delta: ",th.mean((fflow_sp - fflow_sp_v0)**2))
    # print("bflow_sp delta: ",th.mean((bflow_sp - bflow_sp_v0)**2))
    # print(fflow_sp.abs().mean(),bflow_sp.abs().mean())
    # T,F,H,W = vid.shape
    # spix,fflow = stream_bass(vid,sp_size=20,beta=5.)
    # print(spix.shape)

    # -- get warped --
    warped,_ = scatter.run(vid[None,0],fflow_sp[None,0])
    # bflow = bflow_sp[None,0]
    # bflow[:,0] = -bflow[:,0]
    warped_v1,_ = scatter.run_v1(vid[None,0],bflow_sp[None,0])

    # -- save --
    print("delta: ",th.mean((warped - warped_v1)**2).item())
    print("delta [0]: ",th.mean((warped - vid[None,1])**2).item())
    print("delta [1]: ",th.mean((warped_v1 - vid[None,1])**2).item())

    tv_utils.save_image(warped,root / "warped.png")
    tv_utils.save_image(warped_v1,root / "warped_v1.png")
    tv_utils.save_image(vid,root / "vid.png")

    print("vid.shape: ",vid.shape)
    delta_0 = th.abs(warped - vid[None,1])
    delta_0 = delta_0 / delta_0.max()
    delta_1 = th.abs(warped_v1 - vid[None,1])
    delta_1 = delta_1 / delta_1.max()
    deltas = th.cat([delta_0,delta_1])
    print(deltas.shape)
    tv_utils.save_image(deltas,root / "deltas.png")

if __name__ == "__main__":
    main()
