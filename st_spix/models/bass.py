"""

   BASS

"""

import torch as th
from einops import rearrange
from st_spix.prop import stream_bass
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
    use_bass_prop = kwargs['use_bass_prop']
    if use_bass_prop:
        outs = stream_bass(vid,flow=flows,**kwargs)
        spix,params,children,missing,pmaps = outs
    else:
        # -- each independent spix --
        spix = []
        for img in vid:
            img_t = rearrange(img,'f h w -> 1 h w f').contiguous()
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

def get_bass_sims(vid,spix,kwargs):

    # for t in range(vid.shape[0]):
    pooled,downsampled = pooling(vid,spix,nspix)

    exit()
    pass


