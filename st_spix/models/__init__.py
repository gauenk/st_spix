

# -- models --
from . import unet_ssn
from ..utils import optional,get_fxn_kwargs

def load_model(cfg):
    mname = optional(cfg,"mname","default")
    if mname == "unet_ssn":
        fxn = unet_ssn.UNetSsnNet.__init__
        kwargs = get_fxn_kwargs(cfg,fxn)
        model = unet_ssn.UNetSsnNet(**kwargs)
    else:
        raise ValueError(f"Uknown model type [{mname}]")
    return model

