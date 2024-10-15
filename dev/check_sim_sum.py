

import numpy as np
import torch as th
from pathlib import Path

import stnls
import st_spix
from st_spix.prop import stream_bass,run_fwd_bwd
from torchvision.transforms.functional import resize

import st_attn_cuda


def main():

    # -- get root --
    root = Path("./output/check_sim_sum/")
    if not root.exists(): root.mkdir()

    # -- config --
    niters = 80
    niters_seg = 4
    sm_start = 10
    sp_size = 15
    alpha_hastings,potts = 1.,8.
    pix_var = 0.09

    # -- read img/flow --
    vid = st_spix.data.davis_example(isize=None,nframes=10,vid_names=['tennis'])
    # vid = st_spix.data.davis_example(isize=None,nframes=10,vid_names=['baseball'])
    size = 256
    # vid = vid[0,5:7,:,50:50+size,300:300+size]
    vid = vid[0,2:5,:,50:50+size,200:200+size]
    vid = resize(vid,(128,128))
    vid_og = vid.clone()

    # -- get flows --
    from st_spix.flow_utils import run_raft
    fflow,bflow = run_raft(th.clip(255.*vid,0.,255.).type(th.uint8))

    # -- get superpixels --
    outs = stream_bass(vid,flow=fflow,
                       niters=niters,niters_seg=niters_seg,
                       sp_size=sp_size,pix_var=pix_var,
                       alpha_hastings=alpha_hastings,
                       potts=potts,sm_start=sm_start)
    spix = outs[0][None,:]
    nspix = spix.max().item()+1

    # -- compute sims --
    T,F,H,W = vid.shape
    sims = th.ones((1,T,nspix,H,W)).to(vid.device)

    # -- run sim sums --
    ws,wt = 5,1
    fwd = st_attn_cuda.sim_sum_fwd
    flows = stnls.nn.search_flow(fflow[None,:],bflow[None,:],wt).round().int()
    flows = th.zeros_like(flows)
    print("sims.shape: ",sims.shape)
    print("spix.shape: ",spix.shape)
    print("flows.shape: ",flows.shape)
    # simsum = fwd(sims,spix,flows,ws,wt)
    search = stnls.search.NonLocalSearch(ws,wt,dist_type="prod",itype="int")
    dists,inds = search(sims,sims,flows)
    print(dists)
    print(dists.shape)
    print(inds.shape)
    print("nspix: ",nspix)

    # print(simsum)

    # // sims.shape = (nbatch,nframes,height,width,nspix)
    # // seg.shape = (nbatch,nframes,height,width)
    # // flows.shape = (nbatch.nframes,offsets,height,width,2)


if __name__ == "__main__":
    main()
