
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

def main():

    # -- get root --
    root = Path("./output/inspect_warped")
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
    # vid = st_spix.data.davis_example(isize=None,nframes=10,vid_names=['kid-football'])
    vid = st_spix.data.davis_example(isize=None,nframes=10,vid_names=['tennis'])
    vid = vid[0,3:6,:,:480,:480]
    vid = resize(vid,(256,256))
    # vid = vid[...,25:130,220:220+105]
    # vid[:,0] = 0.
    flows = flow_pkg.run(vid[None,:],sigma=0.0,ftype="cv2")

    acc_flows = stnls.nn.search_flow(flows.fflow.clone(),flows.bflow.clone(),1,1)
    print(acc_flows.shape)
    flows_k = run_stnls(vid[None,:],acc_flows,9)
    print(flows_k.shape)
    ones = th.ones_like(flows_k[...,0])
    stacking = stnls.agg.NonLocalGather(1,1,itype="float")
    stack = stacking(vid[None,:],ones,flows_k)[:,0]
    # stack = rearrange(stack,'b k t c h w -> b k t c h w')
    print("stack.shape: ",stack.shape)
    tv_utils.save_image(stack[0,0],root / "stack_0.png")
    tv_utils.save_image(stack[0,1],root / "stack_1.png")
    # exit()

    fflow = flows.fflow[0,0][None,:]
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

    # -- get silly warped --
    warped,_ = scatter.run(vid[None,0],fflow_sp[None,0])
    warped_v1,_ = scatter.run_v1(vid[None,0],bflow_sp[None,0])

    # -- save --
    print("delta: ",th.mean((warped - warped_v1)**2).item())
    print("delta [0]: ",th.mean((warped - vid[None,1])**2).item())
    print("delta [1]: ",th.mean((warped_v1 - vid[None,1])**2).item())

    tv_utils.save_image(warped,root / "warped.png")
    tv_utils.save_image(warped_v1,root / "warped_v1.png")
    tv_utils.save_image(vid,root / "vid.png")

    # -- compute deltas --
    print("vid.shape: ",vid.shape)
    thresh = 5e-2
    delta_0 = th.abs(warped - vid[None,1])
    delta_0[th.where(delta_0>thresh)] = -1
    dmax = delta_0.max()
    delta_0[th.where(delta_0<0)] = dmax
    print("delta_0: ",delta_0.mean(),delta_0[th.where(delta_0<dmax)].mean())
    delta_0 = delta_0 / delta_0.max()

    delta_1 = th.abs(warped_v1 - vid[None,1])
    delta_1[th.where(delta_1>thresh)] = -1
    dmax = delta_1.max()
    delta_1[th.where(delta_1<0)] = dmax
    print("delta_1: ",delta_1.mean(),delta_1[th.where(delta_1<dmax)].mean())
    delta_1 = delta_1 / delta_1.max()
    deltas = th.cat([delta_0,delta_1])

    print(deltas.shape)
    tv_utils.save_image(deltas,root / "deltas.png")

if __name__ == "__main__":
    main()
