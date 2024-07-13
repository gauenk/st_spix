
# -- losses --
from .sp_loss import SuperpixelLoss
from ..utils import optional

def load_loss(cfg):
    lname = optional(cfg,"lname","default")
    if lname == "spix_loss":
        spix_loss_type = optional(cfg,"spix_loss_type","cross")
        spix_loss_compat = optional(cfg,"spix_loss_compat",0.)
        loss = SuperpixelLoss(spix_loss_type,spix_loss_compat)
    else:
        raise ValueError(f"Uknown loss type [{lname}]")
    return loss
