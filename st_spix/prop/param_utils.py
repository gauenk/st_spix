
import torch
import prop_cuda

def unpack_spix_params_to_dict(params):
    unpacked_keys = ["mu_i","mu_s","sigma_s","logdet_Sigma_s",
                     "counts","prior_counts","ids"]
    unpacked = {key:getattr(params,key) for key in unpacked_keys}
    return unpacked

def unpack_spix_params_to_list(params):
    unpacked_keys = ["mu_i","mu_s","sigma_s","logdet_Sigma_s",
                     "counts","prior_counts","ids"]
    unpacked = [getattr(params,key) for key in unpacked_keys]
    return unpacked

def copy_spix_params(params):
    unpacked = unpack_spix_params_to_list(params)
    unpacked = [u.clone() for u in unpacked]
    return prop_cuda.SuperpixelParams(*unpacked)
