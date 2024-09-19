import torch as th
# import st_spix_cuda
import bass_cuda
from einops import rearrange,repeat

def run(img,flow,swap_c=True):
    if swap_c:
        img = rearrange(img,'b c h w -> b h w c')
        flow = rearrange(flow,'b c h w -> b h w c')
    eps = 1e-8
    # flow[...,0] = -2.*flow[...,0]
    # flow[...,1] = -flow[...,1]
    scatter,cnts = bass_cuda.scatter_img_forward(img.contiguous(),
                                                    flow.contiguous(),eps)
    if swap_c:
        scatter = rearrange(scatter,'b h w c -> b c h w')
    return scatter,cnts

def run_v1(img,flow):

    from st_spix.flow_utils import index_grid,flow_warp

    B,F,H,W = img.shape
    scatter = flow_warp(img, flow, "nearest","zeros")
    ones = th.ones_like(img[:,:1])
    cnts = flow_warp(ones, flow, "bilinear","zeros")

    # print("scatter.shape: ",scatter.shape)
    # print("cnt.shape: ",cnts.shape)
    # exit()

    return scatter,cnts[:,0]
