
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

class SuperpixelAttention(nn.Module):
    """

    Superpixel Attention Module

    """

    defs = {"attn_type":"soft","dist_type":"prod","normz_patch":False,
            "qk_scale":None,"nheads":1,"kernel_size":5,"dilation":1,
            "use_proj":True,"use_weights":True,"learn_attn_scale":False,
            "attn_drop":0.0,"proj_drop":0.0,"detach_sims":False,
            "qk_layer":True,"v_layer":True,"proj_layer":True,
            "sp_nftrs":None,"proj_attn_layer":False,
            "proj_attn_bias":False,"run_attn_search":True,
            "na_grid":"stnls","sp_type":"none"}

    def __init__(self,dim,**kwargs):
        super().__init__()

        # -- init --
        extract_self(self,kwargs,self.defs)

        # -- check superpixels --
        assert self.attn_type in ["soft","hard","hard+grad","na"]

        # -- attention modules --
        nheads = self.nheads#kwargs['nheads']
        kernel_size = self.kernel_size#kwargs['kernel_size']
        self.attn_rw = AttnReweight()
        self.na_search,self.na_agg = self.get_na_fxns(dim,nheads,kernel_size,kwargs)

        # -- [optional] project attention to create a weighted conv --
        assert self.run_attn_search or self.proj_attn_layer,"At least one must be true."
        if self.proj_attn_layer:
            bias = self.proj_attn_bias
            dim = kernel_size * kernel_size
            self.proj_attn = nn.Linear(dim,dim,bias=bias)
        else:
            self.proj_attn = nn.Identity()

    def get_na_fxns(self,dim,nheads,kernel_size,kwargs):
        if self.na_grid == "nat":
            kwargs_attn = get_fxn_kwargs(kwargs,NeighAttnMat.__init__)
            na_search = NeighAttnMat(dim,nheads,kernel_size,**kwargs_attn)
            kwargs_agg = get_fxn_kwargs(kwargs,NeighAttnAgg.__init__)
            na_agg = NeighAttnAgg(dim,nheads,kernel_size,**kwargs_agg)
            return na_search,na_agg
        elif self.na_grid == "stnls":
            kwargs_attn = get_fxn_kwargs(kwargs,StnlsNeighAttnMat.__init__)
            na_search = StnlsNeighAttnMat(dim,nheads,kernel_size,**kwargs_attn)
            kwargs_agg = get_fxn_kwargs(kwargs,StnlsNeighAttnAgg.__init__)
            na_agg = StnlsNeighAttnAgg(dim,nheads,kernel_size,**kwargs_agg)
            return na_search,na_agg
        else:
            raise ValueError(f"Unknown na grid [{self.na_grid}]")

    def attn_post_process(self,attn,wt):
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
        flows = flows.int()

        # -- compute attn differences  --
        ws = self.kernel_size
        wt = flows.shape[-4]//2
        x = x.permute(0,2,3,1) # t f h w -> t h w f
        attn,flows_k = self.na_search(x,flows)
        attn = self.attn_post_process(attn,wt)
        attn = self.attn_sign(attn)

        # -- reweight attn map --
        if self.attn_type == "hard":
            sH,sW = sims.shape[-2:]
            inds = sims.view(*sims.shape[:-2],-1).argmax(-1)
            binary = one_hot(inds,sH*sW).reshape_as(sims).type(sims.dtype)
            attn = self.attn_rw(attn,binary,self.normz_patch)
        elif self.attn_type in ["soft","hard+grad"] and "slic" in self.sp_type:
            attn = rearrange(attn,'1 hd t h w k -> t hd h w k')
            assert attn.ndim == 5,"No time dimension; b hd nh nw k"
            sims = sims.contiguous()
            attn = self.attn_rw(attn,sims,self.normz_patch)
        elif self.attn_type in ["soft","hard+grad"]:
            if "slic" in self.sp_type and sims.ndim==5:
                sims = rearrange(sims,'t h w nh nw -> t (nh nw) h w')
            attn = self.attn_rw_stnls(attn,sims,flows,ws,wt)
        elif self.attn_type == "na":
            attn = rearrange(attn,'1 hd t h w k -> t hd h w k')
            attn = attn.softmax(-1)
        else:
            raise ValueError(f"Uknown self.attn_type [{self.attn_type}]")

        # -- aggregate --
        x = self.na_agg(x,attn,flows_k)

        # -- prepare --
        x = x.permute(0,3,1,2)#.clone() # b h w f -> b f h w

        # assert not(th.any(th.isnan(x)).item()),"Must be no nan."
        return x

    def attn_sign(self,attn):
        # print("self.dist_type: ",self.dist_type)
        if self.dist_type == "prod":
            return attn
        elif self.dist_type == "l2":
            return -attn
        else:
            raise ValueError(f"Uknown dist type [{self.dist_type}]")

    def attn_rw_stnls(self,attn,sims,flows,ws,wt):

        # -- compute \sum_s p(s_i=s)p(s_j=s) --
        sims = sims[None,:].contiguous()
        search = stnls.search.NonLocalSearch(ws,wt,dist_type="prod",itype="int")
        sim_attn = search(sims,sims,flows)[0]

        # -- exp(-dists) --
        c = th.max(attn,dim=-1,keepdim=True).values
        attn = th.exp(attn-c)
        # -- exp(-dists) \sum_s p(s_i=s)p(s_j=s)  --
        attn = attn*sim_attn
        # -- normalize --
        attn = attn / (1e-10+attn.sum(-1,keepdim=True))
        attn = rearrange(attn,'1 hd t h w k -> t hd h w k')
        return attn
