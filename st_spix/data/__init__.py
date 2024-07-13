
# -- datasets --
from .bsd500_seg import load_bsd500
from .simple import davis_example
from ..utils import optional

def load_data(cfg):
    dname = optional(cfg,"dname","default")
    if dname == "bsd500":
        load_test = optional(cfg,"data_load_test",False)
        return load_bsd500(cfg,load_test=load_test)
    else:
        raise ValueError(f"Uknown model type [{dname}]")

