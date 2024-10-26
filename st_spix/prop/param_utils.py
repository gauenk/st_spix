
import torch
import prop_cuda

KEYS = ["mu_app", "sigma_app", "logdet_sigma_app", "prior_mu_app", "prior_sigma_app", "prior_mu_app_count", "prior_sigma_app_count", "mu_shape", "sigma_shape", "logdet_sigma_shape", "prior_mu_shape", "prior_sigma_shape", "prior_mu_shape_count", "prior_sigma_shape_count", "counts", "prior_counts", "ids"]

    # .def_readwrite("mu_app", &PySuperpixelParams::mu_app)
    # .def_readwrite("sigma_app", &PySuperpixelParams::sigma_app)
    # .def_readwrite("logdet_sigma_app", &PySuperpixelParams::logdet_sigma_app)
    # .def_readwrite("prior_mu_app", &PySuperpixelParams::prior_mu_app)
    # .def_readwrite("prior_sigma_app", &PySuperpixelParams::prior_sigma_app)
    # .def_readwrite("prior_mu_app_count", &PySuperpixelParams::prior_mu_app_count)
    # .def_readwrite("prior_sigma_app_count", &PySuperpixelParams::prior_sigma_app_count)
    # // -- shape --
    # .def_readwrite("mu_shape", &PySuperpixelParams::mu_shape)
    # .def_readwrite("sigma_shape", &PySuperpixelParams::sigma_shape)
    # .def_readwrite("logdet_sigma_shape", &PySuperpixelParams::logdet_sigma_shape)
    # .def_readwrite("prior_mu_shape", &PySuperpixelParams::prior_mu_shape)
    # .def_readwrite("prior_sigma_shape", &PySuperpixelParams::prior_sigma_shape)
    # .def_readwrite("prior_mu_shape_count", &PySuperpixelParams::prior_mu_shape_count)
    # .def_readwrite("prior_sigma_shape_count", &PySuperpixelParams::prior_sigma_shape_count)
    # // -- helpers --
    # .def_readwrite("counts", &PySuperpixelParams::counts)
    # .def_readwrite("prior_counts", &PySuperpixelParams::prior_counts)
    # .def_readwrite("ids", &PySuperpixelParams::ids);

def unpack_spix_params_to_dict(params):
    unpacked = {key:getattr(params,key) for key in KEYS}
    return unpacked

def unpack_spix_params_to_list(params):
    unpacked = [getattr(params,key) for key in KEYS]
    return unpacked

def spix_dict_to_list(spix_dict):
    unpacked = [spix_dict[k] for k in KEYS]
    return prop_cuda.SuperpixelParams(*unpacked)

def copy_spix_params(params):
    unpacked = unpack_spix_params_to_list(params)
    unpacked = [u.clone() for u in unpacked]
    return prop_cuda.SuperpixelParams(*unpacked)
