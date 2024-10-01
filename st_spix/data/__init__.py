
# -- datasets --
from .bsd500_seg import load_bsd500
from .simple import davis_example
from ..utils import optional

def load_data(cfg):
    dname = optional(cfg,"dname","default")
    if dname == "bsd500":
        load_test = optional(cfg,"data_load_test",False)
        return load_bsd500(cfg,load_test=load_test)
    elif dname == "davis":
        load_test = optional(cfg,"data_load_test",False)
        return load_davis(cfg,load_test=load_test)
    else:
        raise ValueError(f"Uknown model type [{dname}]")


import data_hub
from easydict import EasyDict as edict

def load_davis(cfg,load_test=False):

    # -- data config --
    dcfg = edict()
    dcfg.dname = "davis"
    dcfg.tr_set = "train-val" if load_test is False else "test"
    dcfg.sigma = optional(cfg,"sigma",0.001)
    dcfg.nframes = optional(cfg,"nframes",5)
    dcfg.isize = optional(cfg,"patch_size",128)

    # -- load images --
    device = "cuda:0"
    data, loaders = data_hub.sets.load(dcfg)
    if load_test is False:
        return loaders.tr,loaders.val
    else:
        return data.te,loaders.te
