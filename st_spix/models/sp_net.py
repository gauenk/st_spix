
# -- pytorch --
import torch
import torch as th
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.functional import pad,one_hot
from torch.nn.init import trunc_normal_

# -- basics --
import math
from einops import rearrange

# -- basic utils --
from st_spix.utils import extract_self

# -- superpixel utils --
from st_spix.spix_utils import run_slic,sparse_to_full
from st_spix.spix_utils import compute_slic_params

# -- bist --
import bist_cuda

# -- superpixel --
from .ssn_net import SsnUNet
from .attn_scale_net import AttnScaleNet
from .bass import run_bass,get_bass_sims,bass_kwargs
from st_spix.pwd.pair_wise_distance import PairwiseDistFunction


class SuperpixelNetwork(nn.Module):
    # sp_m is the "spatial" sigma
    defs = {"sp_type":None,"sp_niters":2,"sp_m":0.,"sp_stride":8,
            "sp_scale":1.,"sp_grad_type":"full","sp_nftrs":9,"unet_sm":True,
            "attn_type":None,"use_bass_prop":True,"sp_proj_nftrs":0,
            "bass_prop":False}
    defs.update(bass_kwargs)

    def __init__(self, dim, **kwargs):
        super().__init__()

        # -- init --
        extract_self(self,kwargs,self.defs)

        # -- check network types --
        assert self.sp_type in ["bass","slic","slic+lrn","ssn"]
        self.use_bass = "bass" in self.sp_type
        self.use_slic = "slic" in self.sp_type
        self.use_ssn = "ssn" in self.sp_type
        self.use_lmodel = "lrn" in self.sp_type
        self.use_sproj = self.sp_proj_nftrs > 0

        # -- input dimension --
        id_l0,id_l1,id_l2 = nn.Identity(),nn.Identity(),nn.Identity()
        self.ssn = SsnUNet(dim,9,self.sp_nftrs,self.unet_sm) if self.use_ssn else id_l0
        self.lmodel = AttnScaleNet(dim,2,self.sp_nftrs) if self.use_lmodel else id_l1
        spnf = self.sp_proj_nftrs
        self.sp_proj = SpixFeatureProjection(spnf) if self.use_sproj else id_l2

    def _reshape_sims(self,x,sims):
        if sims.ndim != 5:
            H = x.shape[-2]
            sH = H//self._get_stride()
            shape_str = 'b (sh sw) (h w) -> b h w sh sw'
            sims = rearrange(sims,shape_str,h=H,sh=sH)
        return sims

    def _get_stride(self):
        if hasattr(self.sp_stride,"__len__"):
            return self.sp_stride[0]
        else:
            return self.sp_stride

    def forward(self, x, fflow=None):

        # -- unpack --
        B,F,H,W = x.shape
        sp_stride = self._get_stride()
        sH = H//sp_stride
        sims,num_spixels,ftrs,s_sims = None,None,None,None
        with th.no_grad():
            x_for_iters = self.sp_proj(x)

        if self.use_slic:

            # -- use [learned or fixed] slic parameters --
            if self.use_lmodel:
                ssn_params = self.lmodel(x).reshape(x.shape[0],2,-1)
                m_est,temp_est = ssn_params[:,[0]],ssn_params[:,[1]]
                m_est = m_est.reshape((B,1,H,W))
            else:
                m_est,temp_est = self.sp_m,self.sp_scale

            # -- run slic iterations --
            output = run_slic(x_for_iters, x, self.sp_stride, self.sp_niters,
                              m_est, temp_est, self.sp_grad_type)
            s_sims, sims, num_spixels, ftrs = output
            sims = self._reshape_sims(x,sims)
            B,H,W,nH,nW = sims.shape
            spix = sims.reshape(B,H,W,nH*nW).argmax(-1).reshape(B,1,H,W)
            # sims = get_bass_sims(x,spix[:,0],temp_est) # debug only.
            # # print("sims.shape: ",sims.shape)

        elif self.use_bass:

            # -- use [learned or fixed] slic parameters --
            if self.use_lmodel:
                ssn_params = self.lmodel(x).reshape(x.shape[0],2,-1)
                m_est,temp_est = ssn_params[:,[0]],ssn_params[:,[1]]
                m_est = m_est.reshape((B,1,H,W))
            else:
                m_est,temp_est = self.sp_m,self.sp_scale

            # assert F==3,"Must be three channels for BASS."
            # kwargs = {"use_bass_prop":True,"niters":30,"niters_seg":4,
            #           "sp_size":15,"pix_var":0.1,"alpha_hastings":0.01,
            #           "potts":8.,"sm_start":0,"rgb2lab":False}
            kwargs = {"use_bass_prop":self.bass_prop,"niters":20,"niters_seg":4,
                      "sp_size":20,"sigma2_app":0.011,"sigma2_size":1.,
                      "alpha_hastings":0.,
                      "potts":10.,"sm_start":0,"rgb2lab":False}
            video_mode = self.bass_prop
            # x_for_iters = rearrange(x_for_iters,'b f h w -> b h w f').contiguous()
            fflow = rearrange(fflow,'b f h w -> b h w f').contiguous()
            with th.no_grad():
                x_for_iters = x_for_iters - x_for_iters.mean((1,2),keepdim=True)
                x_for_iters = x_for_iters/x_for_iters.std((1,2),keepdim=True)
                # spix = run_bass(x_for_iters,fflow,kwargs)
                # spix = run_bass(x_for_iters,fflow,kwargs)
                # print(spix.shape)
                sp_size = kwargs['sp_size']
                niters = sp_size
                potts = kwargs['potts']
                # sigma2_app = kwargs['sigma2_app']*kwargs['sigma2_app']
                sigma2_app = 0.008 # this is WAY bigger than it seems
                alpha = 0.

                # -- bist --
                x_for_iters = rearrange(x_for_iters,'t c h w -> t h w c')
                x_for_iters = x_for_iters.contiguous()/10.
                fflow = th.clamp(fflow,-20,20)
                # print("fflow.shape: ",fflow.shape)
                # print("stats: ",x_for_iters.min(),x_for_iters.max())
                # print("fflow: ",fnorm.min(),fnorm.max())
                spix = bist_cuda.bist_forward(x_for_iters,fflow,sp_size,niters,potts,
                                              sigma2_app,alpha,video_mode)
                spix = spix[:,None].contiguous()
            # sims = th.zeros(0)
            sims = get_bass_sims(x,spix,temp_est)
            # print(sims)
            # print("sims.shape: ",sims.shape)
            # exit()

        else:

            # -- directly predict slic probs from networks --
            sims = sparse_to_full(self.ssn(x),sp_stride)
            shape_str = 'b (sh sw) (h w) -> b h w sh sw'
            sims = rearrange(sims,shape_str,h=H,sh=sH)
            num_spixels, ftrs, s_sims = None, None, None

        # -- modify via attn type --
        if self.attn_type == "hard+grad":
            sH,sW = sims.shape[-2:]
            inds = sims.view(*sims.shape[:-2],-1).argmax(-1)
            binary = one_hot(inds,sH*sW)
            sims = binary.view(sims.shape).type(sims.dtype)
            sims = compute_slic_params(x, sims, self.sp_stride,
                                       self.sp_m, self.sp_scale)[1]
            shape_str = 'b (sh sw) (h w) -> b h w sh sw'
            sims = rearrange(sims,shape_str,h=H,sh=sH)

        return sims, spix, num_spixels, ftrs, s_sims


class SpixFeatureProjection(nn.Module):

    def __init__(self, out_dim, **kwargs):
        super().__init__()

        self.out_dim = out_dim
        self.svd_with_batch = False
        # -- init --
        # extract_self(self,kwargs,self.defs)


    def forward(self,x):

        # -- unpack --
        out_dim = self.out_dim
        B,F,H,W = x.shape
        if self.svd_with_batch:
            x = rearrange(x,'b f h w -> 1 (b h w) f')
        else:
            x = rearrange(x,'b f h w -> b (h w) f')
        # print("x.shape: ",x.shape)

        # -- normalize --
        x = x - x.mean(-1,keepdim=True)
        x = x/x.std(-1,keepdim=True)

        # -- svd --
        U, S, V = torch.linalg.svd(x, full_matrices=True)
        U_k = U[:, :, :out_dim]  # First k columns of U
        S_k = S[:,:out_dim]     # First k singular values
        V_k = V[:,:, :out_dim]  # First k columns of V

        # -- debug --
        # print("U.shape: ",U.shape)
        # print("Uk.shape: ",U_k.shape)
        # print("S.shape: ",S.shape)
        # print("Sk.shape: ",S_k.shape,torch.diag_embed(S_k).shape)

        # -- reconstruct --
        xr = torch.bmm(U_k, torch.diag_embed(S_k))  # Reduced features

        # -- reshape --
        if self.svd_with_batch:
            xr = rearrange(xr,'1 (b h w) f -> b f h w',b=B,h=H)
        else:
            xr = rearrange(xr,'b (h w) f -> b f h w',h=H)

        # # Display original and reduced shapes
        # print(f"Original shape: {X.shape}")
        # print(f"Reduced shape: {X_reduced.shape}")

        # # To calculate explained variance by the top k components:
        # explained_variance = (S_k ** 2).sum() / (S ** 2).sum()
        # print("(S_k ** 2).sum() / (S ** 2).sum(): ",(S_k ** 2).sum() / (S ** 2).sum())

        return xr
