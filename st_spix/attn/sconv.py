
# -- basic imports --
import torch
import torch as th
from einops.layers.torch import Rearrange

import torch
import torch as th
from torch import nn
from torch.nn.functional import pad,one_hot
from torch.nn.init import trunc_normal_
from einops import rearrange,repeat

# -- external package --
import stnls

# -- basic utils --
from spix_paper.utils import extract_self

# -- attn modules --
from .nat import NeighAttnMat,NeighAttnAgg
from .stnls import StnlsNeighAttnMat,StnlsNeighAttnAgg
from .attn_reweight import AttnReweight
from ..spix_utils import compute_slic_params
from ..utils import get_fxn_kwargs

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn import Unfold

# # Usage example
# conv_layer = ConvUsingLinear(in_channels=3, out_channels=16, kernel_size=3, stride=1, padding=1)
# input_tensor = torch.randn(1, 3, 32, 32)  # Example input
# output_tensor = conv_layer(input_tensor)
# print(output_tensor.shape)  # Should output (1, 16, 32, 32)


class SuperpixelConv(nn.Module):

    # defs = {"qk_scale":None,"nheads":1,"kernel_size":5,"dilation":1,
    #         "use_proj":False,"use_weights":True,"learn_attn_scale":False,
    #         "attn_drop":0.0,"proj_drop":0.0,"detach_sims":False,
    #         "qk_layer":False,"v_layer":False,"proj_layer":False,
    #         "sp_nftrs":None,"proj_attn_layer":True,
    #         "proj_attn_bias":True,"run_attn_search":True,
    #         "na_grid":"stnls","sp_type":"none",
    #         "use_kernel_reweight":True,
    #         "use_kernel_renormalize":True}
            # "use_proj":False,"use_weights":True,"learn_attn_scale":False,
            # "attn_drop":0.0,"proj_drop":0.0,"detach_sims":False,
            # "qk_layer":False,"v_layer":False,"proj_layer":False,
            # "sp_nftrs":None,"proj_attn_layer":True,
            # "proj_attn_bias":True,"run_attn_search":True,
            # "na_grid":"stnls","sp_type":"none",

    defs = {"kernel_size":5,
            "use_kernel_reweight":True,
            "use_kernel_renormalize":True,
            "normalize_sims_type":"sum"}

    def __init__(self, dim, **kwargs):
        super().__init__()

        # -- init --
        stride = 1
        extract_self(self,kwargs,self.defs)
        in_chnls = dim
        out_chnls = dim
        kernel_size = self.kernel_size
        self.in_dim = dim
        self.out_dim = dim
        self.stride = stride
        self.padding = (kernel_size-1)//2
        self.unfold = Unfold(kernel_size=kernel_size,
                             stride=self.stride,padding=self.padding)
        self.linear = nn.Linear(in_chnls * kernel_size * kernel_size, out_chnls)

    def forward(self, x, sims, _api_compat1=None):
        # Extract patches and flatten them
        patches = self.unfold(x)  # Shape: (batch_size, in_channels * kernel_size * kernel_size, L)
        patches = patches.transpose(1, 2)  # Shape: (batch_size, L, in_channels * kernel_size * kernel_size)

        # -- unpack weights --
        batchsize = len(x)
        kernel = self.linear.weight
        out_dim,total_ksize = kernel.shape
        kernel = kernel.unsqueeze(0).expand(batchsize,out_dim,total_ksize)
        bias = self.linear.bias

        # -- dev --
        # out0 = th.bmm(patches,kernel.transpose(1,2)) + bias
        # print("out0.shape: ",out0.shape)
        # # exit()

        # -- reweight with sims --
        if self.use_kernel_reweight:

            # -- compute the reweighting term --
            rweight = self.get_reweight_map(sims)
            rweight = rearrange(rweight,'b h w k -> b 1 1 h w k')
            # rweight[...] = 1.

            # -- apply the reweighting term --
            in_dim,out_dim = self.in_dim,self.out_dim
            ksize2 = self.kernel_size*self.kernel_size
            kernel = kernel.reshape(batchsize,out_dim,in_dim,ksize2)
            kernel = rearrange(kernel,'b od id k -> b od id 1 1 k')
            kernel = kernel * rweight
            kernel = rearrange(kernel,'b od id h w k -> b od (h w) (id k)')

            # -- optionally renormalize --
            if self.use_kernel_renormalize:
                kernel = kernel / (1e-10+kernel.abs().sum(-1,keepdim=True))

            # -- apply kernel --
            out = th.sum(patches.unsqueeze(1) * kernel,-1).transpose(1,2) + bias
            # print("hi.")

        else:

            # -- apply kernel --
            out = th.bmm(patches,kernel.transpose(1,2)) + bias


        # print("delta: ",th.mean((out - out0)**2).item())
        # exit()

        # Reshape back to (batch_size, out_channels, H_out, W_out)
        batch_size, num_patches, _ = out.shape
        H_out = int((x.size(2) + 2 * self.padding - self.kernel_size) / self.stride + 1)
        W_out = int((x.size(3) + 2 * self.padding - self.kernel_size) / self.stride + 1)

        out = out.transpose(1, 2).reshape(batch_size, -1, H_out, W_out)
        return out

    def get_reweight_map(self,sims):
        # -- compute \sum_s p(s_i=s)p(s_j=s) --
        ws = self.kernel_size
        # print(sims.shape)
        sims = sims[None,:].contiguous()
        # print(sims.shape)
        # sims = rearrange(sims,'t b h w sh sw -> t b (sh sw) h w')
        # print(sims.shape)
        # exit()
        search = stnls.search.NonLocalSearch(ws,0,dist_type="prod",itype="int")
        # print(sims.shape)
        # exit()
        T,B,F,H,W = sims.shape
        # B,HD,T,W_t,2,H,W = flows.shape
        flows = th.zeros((B,1,T,1,2,H,W),device=sims.device)
        assert not(th.any(th.isnan(sims)).item()),"[0] Must be no nan."
        # print(sims.shape,sims.min().item(),sims.max().item())
        sim_attn = search(sims,sims,flows)[0]
        # print(sim_attn.shape,sim_attn.min().item(),sim_attn.max().item())
        # assert not(th.any(th.isnan(sim_attn)).item()),"[1] Must be no nan."
        if self.normalize_sims_type == "sum":
            sim_attn = sim_attn/(1e-5+sim_attn.sum(-1,keepdim=True))
        elif self.normalize_sims_type == "max":
            sim_attn = sim_attn/(1e-5+sim_attn.max(-1,keepdim=True).values)
        else:
            normz = self.normalize_sims_type
            raise ValueError(f"Uknown normalization method [{normz}]")
        # print("[a] sim_attn.shape: ",sim_attn.shape)
        sim_attn = rearrange(sim_attn,'1 1 t h w k -> t h w k')
        # print("sim_attn.shape: ",sim_attn.shape)
        # sim_attn = rearrange(sim_attn,'1 1 t h w k -> t 1 1 h w k')
        # assert not(th.any(th.isnan(sim_attn)).item()),"Must be no nan."
        return sim_attn


class SuperpixelConv_old(nn.Module):
    """

    Superpixel Convolution Module

    """

    defs = {"attn_type":"soft","dist_type":"prod","normz_patch":False,
            "qk_scale":None,"nheads":1,"kernel_size":5,"dilation":1,
            "use_proj":False,"use_weights":True,"learn_attn_scale":False,
            "attn_drop":0.0,"proj_drop":0.0,"detach_sims":False,
            "qk_layer":False,"v_layer":False,"proj_layer":False,
            "sp_nftrs":None,"proj_attn_layer":True,
            "proj_attn_bias":True,"run_attn_search":True,
            "na_grid":"stnls","sp_type":"none",
            "use_kernel_reweight":True,
            "use_kernel_renormalize":True}

    def __init__(self,dim,**kwargs):
        super().__init__()

        # -- init --
        extract_self(self,kwargs,self.defs)

        # -- check superpixels --
        assert self.attn_type in ["soft","hard","hard+grad","na"]

        # -- attention modules --
        nheads = self.nheads
        kernel_size = self.kernel_size
        kwargs_attn = get_fxn_kwargs(kwargs,NeighAttnMat.__init__)
        self.na_search = NeighAttnMat(dim,nheads,kernel_size,**kwargs_attn)
        kwargs_agg = get_fxn_kwargs(kwargs,NeighAttnAgg.__init__)
        self.na_agg = NeighAttnAgg(dim,nheads,kernel_size,**kwargs_agg)

        # -- [optional] project attention to create a weighted conv --
        assert self.run_attn_search or self.proj_attn_layer,"At least one must be true."
        if self.proj_attn_layer:
            bias = self.proj_attn_bias
            dim = kernel_size * kernel_size
            self.proj_attn = nn.Linear(dim,dim,bias=bias)
        else:
            self.proj_attn = nn.Identity()


    def xform_attn2conv(self,attn,wt):
        # yes; this is method to implement spixconv is really [really] silly
        if not (self.run_attn_search is True):
            attn = 0*attn
        attn = rearrange(attn,'... (h f) -> ... h f',h=2*wt+1)
        attn = self.proj_attn(attn)
        attn = rearrange(attn,'... h f -> ... (h f)',h=2*wt+1)
        return attn

    def forward(self, x, sims, flows, state=None):

        # -- unpack superpixel info --
        if self.detach_sims:
            sims = sims.detach()
        # flows do nothing here.

        # -- compute attn differences  --
        ws = self.kernel_size
        x = x.permute(0,2,3,1) # t f h w -> t h w f
        kernel,_ = self.na_search(x)
        kernel = rearrange(kernel,'b hd 1 h w k -> b hd h w k')
        rweight = self.get_reweight_map(sims)
        if self.use_kernel_reweight:
            kernel = kernel * rweight
        if self.use_kernel_renormalize:
            kernel = kernel / (1e-10+kernel.abs().sum(-1,keepdim=True))
            # kernel = kernel / (1.+kernel.abs().sum(-1,keepdim=True)) # idk; this worked.

        # assert not(th.any(th.isnan(kernel)).item()),"Must be no nan."
        # print(kernel.mean().item())
        # print("kernel.shape: ",kernel.shape)

        x = self.na_agg(x,kernel,None)
        # print("x.shape: ",x.shape)
        # exit()

        # -- spoof --
        # T,F,H,W = x.shape
        # x = rearrange(x,'t f h w -> (t h w) f')

        # -- prepare --
        x = x.permute(0,3,1,2)#.clone() # b h w f -> b f h w

        assert not(th.any(th.isnan(x)).item()),"Must be no nan."
        return x


    def get_reweight_map(self,sims):
        # -- compute \sum_s p(s_i=s)p(s_j=s) --
        ws = self.kernel_size
        # print(sims.shape)
        sims = sims[None,:].contiguous()
        # print(sims.shape)
        # sims = rearrange(sims,'t b h w sh sw -> t b (sh sw) h w')
        # print(sims.shape)
        # exit()
        search = stnls.search.NonLocalSearch(ws,0,dist_type="prod",itype="int")
        # print(sims.shape)
        # exit()
        T,B,F,H,W = sims.shape
        # B,HD,T,W_t,2,H,W = flows.shape
        flows = th.zeros((B,1,T,1,2,H,W),device=sims.device)
        assert not(th.any(th.isnan(sims)).item()),"[0] Must be no nan."
        # print(sims.shape,sims.min().item(),sims.max().item())
        sim_attn = search(sims,sims,flows)[0]
        # print(sim_attn.shape,sim_attn.min().item(),sim_attn.max().item())
        # assert not(th.any(th.isnan(sim_attn)).item()),"[1] Must be no nan."
        sim_attn = sim_attn/(1e-5+sim_attn.sum(-1,keepdim=True))
        # print("[a] sim_attn.shape: ",sim_attn.shape)
        sim_attn = rearrange(sim_attn,'1 1 t h w k -> t h w k')
        # print("sim_attn.shape: ",sim_attn.shape)
        # sim_attn = rearrange(sim_attn,'1 1 t h w k -> t 1 1 h w k')
        # assert not(th.any(th.isnan(sim_attn)).item()),"Must be no nan."
        return sim_attn
