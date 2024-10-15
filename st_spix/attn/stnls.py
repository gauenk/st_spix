"""
Neighborhood Attention 2D PyTorch Module

This source code is licensed under the license found in the
LICENSE file in the root directory of this source tree.

"""

import torch
import torch as th
from torch import nn
from torch.nn.functional import pad
from torch.nn.init import trunc_normal_
from einops import rearrange

from stnls.search import NonLocalSearch
from stnls.agg import NonLocalGather
# from natten.functional import natten2dav, natten2dqkrpb
try:
    from ..models.attn_scale_net import AttnScaleNet
except:
    pass


def run_search_fxn(q, k, flows, kernel_size, dilation, dist_type):
    attn,flows_k = run_stnls_search(q,k,flows,kernel_size,dilation,dist_type)
    return attn,flows_k
    # if dist_type == "prod":
    #     attn = na2d_qk_with_bias(q, k, None, kernel_size, dilation)
    #     # attn = run_stnls_search(q,k,kernel_size,dilation,dist_type)[0]
    #     return attn
    # elif dist_type == "l2":
    #     attn = run_stnls_search(q,k,kernel_size,dilation,dist_type)[0]
    #     return attn
    # else:
    #     raise ValueError(f"Uknown dist_type [{dist_type}]")

def run_stnls_search(q,k,flows,kernel_size,dilation,dist_type):
    num_heads = q.shape[1]
    ws = kernel_size
    ps,pt,_k = 1,1,-1
    stride0,stride1 = 1,1
    topk_mode = "none"
    wt = flows.shape[-4]//2
    search = NonLocalSearch(ws, wt, ps, _k, nheads=num_heads,
                            stride0=stride0, stride1=stride1,
                            dist_type=dist_type, dilation=dilation,
                            pt=pt, self_action=None, topk_mode=topk_mode,
                            reflect_bounds=True, full_ws=True,
                            use_adj=False, itype="int")
    q = rearrange(q,'t hd h w f -> 1 hd t f h w').contiguous()
    k = rearrange(k,'t hd h w f -> 1 hd t f h w').contiguous()
    # print("flows.shape: ",flows.shape)
    # print("q.shape: ",q.shape)
    attn,flows = search(q,k,flows)
    # print("attn.shape: ",attn.shape)
    # exit()
    # print(attn)
    # print(dist_type)
    # exit()
    return attn,flows

def run_stnls_agg(v,attn,flows):
    # weights = th.nn.functional.softmax(10*dists,-1)
    ps,stride0 = 1,1
    agg = NonLocalGather(ps,stride0)
    attn = attn[:,:,None] # b hd t h w kernel
    flows = flows[:,:,None]
    v = rearrange(v,'b hd h w f -> b hd 1 f h w').contiguous()
    v = th.sum(agg(v,attn,flows),2) # b hd k t f h w
    v = rearrange(v,'b 1 1 f h w -> b f h w').contiguous()
    return v

def run_nat_agg(v,attn,ksize,dilation):
    x = na2d_av(attn, v, ksize, dilation)
    x = rearrange(x,'b 1 h w f -> b f h w')
    return x

class StnlsNeighAttnMat(nn.Module):
    """
    Neighborhood Attention 2D Module
    """

    def __init__(
            self,
            dim,
            num_heads,
            kernel_size,
            dilation=1,
            bias=False,
            qk_bias=False,
            qk_scale=None,
            learn_attn_scale=False,
            # detach_learn_attn=False,
            dist_type="prod",
            sp_nftrs=3):
        super().__init__()
        self.num_heads = num_heads
        self.head_dim = dim // self.num_heads
        self.scale = qk_scale or self.head_dim**(-0.5)
        # self.detach_learn_attn = detach_learn_attn
        self.dist_type = dist_type
        # assert dist_type == "prod","Only dist_type = 'prod' supported with NA"
        # print(self.scale)
        # exit()
        assert (
            kernel_size > 1 and kernel_size % 2 == 1
        ), f"Kernel size must be an odd number greater than 1, got {kernel_size}."
        self.kernel_size = kernel_size
        assert (
            dilation is None or dilation >= 1
        ), f"Dilation must be greater than or equal to 1, got {dilation}."
        self.dilation = dilation or 1
        self.window_size = self.kernel_size * self.dilation

        self.qk = nn.Linear(dim, dim * 2, bias=qk_bias)
        if bias:
            self.rpb = nn.Parameter(
                torch.zeros(num_heads, (2 * kernel_size - 1), (2 * kernel_size - 1))
            )
            trunc_normal_(self.rpb, std=0.02, mean=0.0, a=-2.0, b=2.0)
        else:
            self.register_parameter("rpb", None)
        assert self.rpb is None

        self.learn_attn_scale = learn_attn_scale
        if not self.learn_attn_scale:
            self.attn_scale_net = nn.Identity()
        else:
            self.attn_scale_net = AttnScaleNet(dim, 1, sp_nftrs)

    def forward(self, x, flows):
        B, Hp, Wp, C = x.shape
        H, W = int(Hp), int(Wp)
        pad_l = pad_t = pad_r = pad_b = 0
        if H < self.window_size or W < self.window_size:
            pad_l = pad_t = 0
            pad_r = max(0, self.window_size - W)
            pad_b = max(0, self.window_size - H)
            x = pad(x, (0, 0, pad_l, pad_r, pad_t, pad_b))
            _, H, W, _ = x.shape
        # print("x.shape: ",x.shape)
        qk = (
            self.qk(x)
            .reshape(B, H, W, 2, self.num_heads, self.head_dim)
            .permute(3, 0, 4, 1, 2, 5)
        )
        q, k = qk[0], qk[1]
        # print("q.shape: ",q.shape)
        # print(B, H, W, 2, self.num_heads, self.head_dim)
        # exit()

        # # -- compare --
        # diff0 = th.mean((q-x[:,None])**2).item()
        # diff1 = th.mean((k-x[:,None])**2).item()
        # print("differences: ",diff0,diff1)

        # -- rescaling --
        # print(self.scale)
        # exit()
        scale = self.scale
        if self.learn_attn_scale:
            scale = self.attn_scale_net(rearrange(x,'b h w c -> b c h w'))
            scale = rearrange(scale,'t 1 h w -> 1 1 t h w 1') # B,HD,T,H,W,F
        # if self.dist_type == "prod": q = scale * q # before if "prod"
        # print(q.shape)
        attn,flows_k = run_search_fxn(q, k, flows, self.kernel_size,
                                      self.dilation, self.dist_type)
        # print(attn.shape)
        # exit()
        attn = scale * attn
        # print("attn.shape: ",attn.shape)
        # if self.dist_type == "l2": attn = scale * attn # after if "l2"
        # attn = na2d_qk_with_bias(q, k, self.rpb, self.kernel_size, self.dilation)
        return attn,flows_k

    def extra_repr(self) -> str:
        return (
            f"head_dim={self.head_dim}, num_heads={self.num_heads}, "
            + f"kernel_size={self.kernel_size}, dilation={self.dilation}, "
            + f"rel_pos_bias={self.rpb is not None}"
        )


class StnlsNeighAttnAgg(nn.Module):

    def __init__(
            self,
            dim,
            num_heads,
            kernel_size,
            dilation=1,
            v_bias=False,
            v_layer=True,
            proj_layer=True,
            attn_drop=0.0,
            proj_drop=0.0):
        super().__init__()

        # -- for padding --
        assert (
            kernel_size > 1 and kernel_size % 2 == 1
        ), f"Kernel size must be an odd number greater than 1, got {kernel_size}."
        self.kernel_size = kernel_size
        assert (
            dilation is None or dilation >= 1
        ), f"Dilation must be greater than or equal to 1, got {dilation}."
        self.dilation = dilation or 1
        self.window_size = self.kernel_size * self.dilation

        self.num_heads = num_heads
        self.head_dim = dim // self.num_heads
        if v_layer:
            self.v = nn.Linear(dim, dim * 1, bias=v_bias)
        else:
            self.v = nn.Identity()
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(dim, dim) if proj_layer else nn.Identity()
        self.proj_drop = nn.Dropout(proj_drop)

    def forward(self, x, attn, flows):
        B, Hp, Wp, C = x.shape
        H, W = int(Hp), int(Wp)
        pad_l = pad_t = pad_r = pad_b = 0
        if H < self.window_size or W < self.window_size:
            pad_l = pad_t = 0
            pad_r = max(0, self.window_size - W)
            pad_b = max(0, self.window_size - H)
            x = pad(x, (0, 0, pad_l, pad_r, pad_t, pad_b))
            _, H, W, _ = x.shape

        # -- get values --
        v = rearrange(self.v(x),'t h w (hd f) -> 1 hd t f h w',hd=self.num_heads)
        # B,HD_v,T,F,H,W = v.shape

        # -- drop attn [trainig] --
        attn = self.attn_drop(attn)

        # -- aggregate --
        import stnls
        stack = stnls.agg.NonLocalGather(1,1)
        # print("attn.shape: ",attn.shape)
        # print("flows.shape: ",flows.shape)
        # exit()
        x = stack(v,attn,flows) # re-weight each "k" by "attn[@k]"
        x = rearrange(x,'b hd k t f h w -> (b t) h w (hd f) k').sum(-1)

        # -- restory back to original shape --
        if pad_r or pad_b:
            x = x[:, :Hp, :Wp, :]
        return self.proj_drop(self.proj(x))

    def extra_repr(self) -> str:
        return (
            f"head_dim={self.head_dim}, num_heads={self.num_heads}, "
        )

def natten_padding(x,kernel_size):
    window_size = kernel_size*kernel_size
    B, Hp, Wp, C = x.shape
    H, W = int(Hp), int(Wp)
    pad_l = pad_t = pad_r = pad_b = 0
    if H < window_size or W < window_size:
        pad_l = pad_t = 0
        pad_r = max(0, window_size - W)
        pad_b = max(0, window_size - H)
        x = pad(x, (0, 0, pad_l, pad_r, pad_t, pad_b))
        _, H, W, _ = x.shape
    pad_info = {"Hp":Hp,"Wp":Wp,"pad_r":pad_r,"pad_b":pad_b}
    return x,pad_info

def natten_remove_padding(x,pad_info):
    Hp,Wp = pad_info["Hp"],pad_info["Wp"]
    pad_r,pad_b = pad_info["pad_r"],pad_info["pad_b"]
    if pad_r or pad_b:
        x = x[:, :Hp, :Wp, :]
    return x
