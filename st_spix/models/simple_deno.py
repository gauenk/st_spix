"""

   Simple Denoising Network

"""

# -- import torch --
import torch
import torch as th
import torch.nn as nn
import torch.nn.functional as F

# -- basic --
import math
import numpy as np
from einops import rearrange
from easydict import EasyDict as edict

# -- submodules --
from ..utils import extract
from .sp_net import SuperpixelNetwork
from ..attn import SuperpixelAttention

class SimpleDenoiser(nn.Module):

    defs = dict(SuperpixelNetwork.defs)
    defs.update(SuperpixelAttention.defs)
    defs.update({"lname":"deno","net_depth":1,"simple_linear_bias":False,"output_dict":True})

    def __init__(self, in_dim, dim, **kwargs):
        super().__init__()

        # -- init --
        self.output_dict = kwargs['output_dict']

        # -- linear --
        bias = kwargs['simple_linear_bias']
        self.lin0 = nn.Linear(in_dim,dim,bias=bias)
        self.lin1 = nn.Linear(dim,in_dim,bias=bias)


        # -- learn attn scale --
        attn_kwargs = extract(kwargs,SuperpixelNetwork.defs)
        self.attn = SuperpixelAttention(dim,**attn_kwargs)

        # -- superpixel network --
        self.use_sp_net = not(kwargs['attn_type'] == "na" and kwargs['lname'] == "deno")
        spix_kwargs = extract(kwargs,SuperpixelNetwork.defs)
        self.spix_net = SuperpixelNetwork(dim,**spix_kwargs) if self.use_sp_net else None

    def forward(self, x, noise_info=None):
        """

        Forward function.

        """

        # -- unpack --
        H,W = x.shape[-2:]
        shape0 = lambda x: rearrange(x,'b c h w -> b (h w) c')
        shape1 = lambda x: rearrange(x,'b (h w) c -> b c h w',h=H,w=W)
        apply_lin = lambda x,lin: shape1(lin(shape0(x)))

        # -- forward --
        ftrs = apply_lin(x,self.lin0)
        if self.use_sp_net: sims = self.spix_net(ftrs)[0]
        else: sims = None
        ftrs = ftrs+self.attn(ftrs,sims)
        deno = x + apply_lin(ftrs,self.lin1)

        if self.output_dict:
            return {"deno":deno,"sims":sims}
        else:
            return deno # remove me
