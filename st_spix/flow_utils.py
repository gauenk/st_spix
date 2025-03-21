
import torch as th
import torch.nn.functional as th_f
import numpy as np
import matplotlib.pyplot as plt
from einops import rearrange

def index_grid(H,W,dtype=th.float,device="cuda",normalize=False,stack_dim=0):
    # -- create mesh grid --
    grid_y, grid_x = th.meshgrid(th.arange(0, H, dtype=dtype, device=device),
                                 th.arange(0, W, dtype=dtype, device=device))
    if normalize:
        grid_x = grid_x / (W-1)
        grid_y = grid_y / (H-1)
    if stack_dim == 0:
        grid = th.stack((grid_x, grid_y), 0).float()[None,:]  # 1, 2, W(x), H(y)
    elif stack_dim == -1:
        grid = th.stack((grid_x, grid_y), -1).float()[None,:]  # 1, W(x), H(y), 2
    else:
        raise ValueError("Dont do it.")
    grid.requires_grad = False
    return grid

def flow_warp(x, flow, interp_mode='bilinear',
              padding_mode='reflection', align_corners=True):
    """Warp an image or feature map with optical flow.

    Args:
        x (Tensor): Tensor with size (n, c, h, w).
        flow (Tensor): Tensor with size (n, h, w, 2), normal value.
        interp_mode (str): 'nearest' or 'bilinear' or 'nearest4'. Default: 'bilinear'.
        padding_mode (str): 'zeros' or 'border' or 'reflection'.
            Default: 'zeros'.
        align_corners (bool): Before pytorch 1.3, the default value is
            align_corners=True. After pytorch 1.3, the default value is
            align_corners=False. Here, we use the True as default.


    Returns:
        Tensor: Warped image or feature map.
    """
    # -- unpack --
    H,W = x.shape[-2:]
    n, _, h, w = x.size()
    assert flow.ndim == 4,"Flow must be 4-dim"

    # -- create mesh grid --
    grid = index_grid(h,w,dtype=x.dtype,device=x.device)
    vgrid = grid + flow

    # -- scale grid to [-1,1] --
    hp,wp = x.shape[-2:]
    vgrid_x = 2.0 * vgrid[:, 0, :, :] / max(wp - 1, 1) - 1.0
    vgrid_y = 2.0 * vgrid[:, 1, :, :] / max(hp - 1, 1) - 1.0
    vgrid_scaled = th.stack((vgrid_x, vgrid_y), dim=-1)

    # -- resample --
    output = th_f.grid_sample(x, vgrid_scaled, mode=interp_mode,
                              padding_mode=padding_mode, align_corners=align_corners)

    return output


def viz_flow_quiver(name,flow,step=8):
    if th.is_tensor(flow):
        flow = flow.detach().cpu().numpy()
    B,_,H,W = flow.shape
    assert B == 1
    flow = rearrange(flow,'1 f h w -> h w f')
    # print(flow.shape)
    # print(len(np.arange(0, flow.shape[1], step)),
    #       len(np.arange(flow.shape[0], 0, -step)))
    plt.quiver(np.arange(0, flow.shape[1], step),
               np.arange(flow.shape[0], 0, -step),
               flow[::step, ::step, 0], flow[::step, ::step, 1])
    plt.savefig(name)
    plt.close("all")

def run_raft(vid):

    # -- raft imports --
    import torch
    import torch as th
    from raft.raft import RAFT
    from raft.utils.utils import InputPadder
    from easydict import EasyDict as edict

    # -- load model --
    model_fn = "/home/gauenk/Documents/packages/RAFT/models/raft-kitti.pth"
    args = {"model":model_fn, "small":False,
            "mixed_precision":False, "alternate_corr":False}
    args = edict(args)
    model = torch.nn.DataParallel(RAFT(args))
    model.load_state_dict(torch.load(args.model,weights_only=False))
    model = model.module
    model.to(vid.device)
    # print(vid.min(),vid.max())

    model.eval()
    fflow,bflow = run_model_on_video(vid,model)

    if fflow.shape[-1] != vid.shape[-1]:
        print("RAFT wants image size to be a multiple of 8.")
        exit()

    return fflow,bflow

def run_spynet(vid):
    from .spynet import SpyNet
    model = SpyNet().to(vid.device).eval()
    fflow,bflow = run_model_on_video(vid,model)
    return fflow,bflow

def load_raft():
    # -- raft imports --
    import torch
    import torch as th
    from raft.raft import RAFT
    from raft.utils.utils import InputPadder
    from easydict import EasyDict as edict

    # -- load model --
    model_fn = "/home/gauenk/Documents/packages/RAFT/models/raft-kitti.pth"
    args = {"model":model_fn, "small":False,
            "mixed_precision":False, "alternate_corr":False}
    args = edict(args)
    model = torch.nn.DataParallel(RAFT(args))
    model.load_state_dict(torch.load(args.model,weights_only=False))
    model = model.module
    # model.to(vid.device)
    model.eval()
    return model

def run_raft_on_video(vid,model):
    return run_model_on_video(vid,model)

def run_model_on_video(vid,model):

    # -- raft imports --
    import torch
    import torch as th
    # from raft.raft import RAFT
    from raft.utils.utils import InputPadder
    from easydict import EasyDict as edict

    # -- run/collect flow --
    H,W = vid.shape[-2:]
    fflow,bflow = [],[]
    for ti in range(vid.shape[0]-1):

        # -- unpack --
        img1,img2 = vid[[ti]],vid[[ti+1]]

        # -- normalize? --
        # img1 = (img1*255.)
        # img2 = (img2*255.)

        # -- padding --
        padder = InputPadder(img1.shape)
        img1, img2 = padder.pad(img1, img2)

        if "RAFT" in model.__class__.__name__:

            # print(img1.shape,img2.shape)

            # -- compute padding --
            with th.no_grad():
                # _, fflow_ti = model(img1, img2, iters=20, test_mode=True)
                # _, bflow_ti = model(img2, img1, iters=20, test_mode=True)
                # print("flow_low.shape,flow_up.shape: ",flow_low.shape,flow_up.shape)
                _, fflow_ti = model(img1, img2, iters=20, test_mode=True)
                _, bflow_ti = model(img2, img1, iters=20, test_mode=True)

        else:

            # -- compute padding --
            with th.no_grad():
                fflow_ti = model(img1, img2)
                bflow_ti = model(img2, img1)
            # print("fflow_ti.shape: ",fflow_ti.shape)
            # print("bflow_ti.shape: ",bflow_ti.shape)
            # exit()

        fflow.append(fflow_ti[...,:H,:W])
        bflow.append(bflow_ti[...,:H,:W])

    # -- pad with zeros [for compatibility; dev only] --
    zflow = th.zeros_like(fflow[0])
    fflow = fflow   + [zflow]
    bflow = [zflow] + bflow

    # -- stacking --
    fflow = th.cat(fflow)
    bflow = th.cat(bflow)

    return fflow,bflow
