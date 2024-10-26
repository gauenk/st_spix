"""

   BASS

"""

import torch as th
from einops import rearrange
import st_spix
from st_spix.prop import stream_bass
from st_spix.sp_video_pooling import video_pooling
from st_spix.sp_pooling import sp_pooling
import prop_cuda

bass_kwargs = {"use_bass_prop":False,"niters":30,"niters_seg":4,
               "sp_size":25,"pix_var":0.1,"alpha_hastings":0.01,
               "potts":10.,"sm_start":0}

def unpack_kwargs(kwargs):
    keys = ["niters","niters_seg","sm_start",
            "sp_size","pix_var","potts","alpha_hastings"]
    params = [kwargs[key] for key in keys]
    return params

def run_bass(vid,flows,kwargs):
    assert vid.shape[1] == 3,"Must use 3 features."
    vid = rearrange(vid,'t f h w -> t h w f').contiguous()
    use_bass_prop = kwargs['use_bass_prop']
    rgb2lab = kwargs['rgb2lab']
    # kwargs['rgb2lab'] = False
    if use_bass_prop:
        del kwargs['use_bass_prop']
        outs = stream_bass(vid,flow=flows,**kwargs)
        spix,params,children,missing,pmaps = outs
        spix = spix[:,None]
    else:
        # -- each independent spix --
        spix = []
        if rgb2lab:
            vid_lab = st_spix.utils.vid_rgb2lab_th(vid.clone(),normz=False)
        else:
            vid_lab = vid
        vid_lab = rearrange(vid_lab,'t f h w -> t h w f').contiguous()
        for img_t in vid:
            img_t = img_t[None,:]
            # img_t = rearrange(img,'f h w -> 1 h w f').contiguous()
            sm_start = 0
            # niters,niters_seg = sp_size,4 # a fun choice from BASS authors
            outs = unpack_kwargs(kwargs)
            niters,niters_seg,sm_start,sp_size,pix_var,potts,alpha_hastings = outs
            spix_t,params_t = prop_cuda.bass(img_t,niters,niters_seg,sm_start,
                                             sp_size,pix_var,potts,alpha_hastings)
            nspix_t = spix_t.max().item()+1
            spix.append(spix_t)
        spix = th.stack(spix)
    return spix

# def get_bass_sims(vid,spix,scale=1.):
#     means,down = sp_pooling(vid,spix)
#     pwd = th.cdist(down,down)
#     sims = th.exp(-scale*pwd)
#     return sims

def get_bass_sims(vid,spix,scale=1.):
    T,F,H,W = vid.shape
    use_video_pooling = False

    if use_video_pooling:
        means,down = video_pooling(vid[None,:],spix[None,:])
        vid = rearrange(vid,'t f h w -> 1 (t h w) f')
        pwd = th.cdist(vid,down)**2 # sum-of-squared differences
        pwd = rearrange(pwd,'1 (t h w) s -> t s h w',t=T,h=H)
    else:
        vid = rearrange(vid,'t f h w -> t h w f')
        means,down = sp_pooling(vid,spix)
        vid = rearrange(vid,'t h w f -> t (h w) f')
        pwd = th.cdist(vid,down)**2 # sum-of-squared differences
        pwd = rearrange(pwd,'t (h w) s -> t s h w',t=T,h=H)

    # -- normalize --
    sims = th.softmax(-scale*pwd,1)

    return sims

