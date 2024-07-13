
"""

Training

"""


# -- sys --
import os
import numpy as np
import pandas as pd

# -- training --
from st_spix.trte_spix import train

# -- caching results --
import cache_io


def main():

    # -- start info --
    verbose = True
    pid = os.getpid()
    print("PID: ",pid)
    version = "v1"

    # -- get experiments --
    def clear_fxn(num,cfg): return False
    exp_fn_list = [
        "exps/comp_graphs/train.cfg",
        "exps/comp_graphs/train_empty.cfg",
    ]
    exps,uuids = [],[]
    cache_fn = ".cache_io_exps/comp_graphs/train/"
    for exp_fn in exp_fn_list:
        _exps,_uuids = cache_io.train_stages.run(exp_fn,cache_fn,
                                                 fast=False,update=True,
                                                 cache_version=version)
        exps += _exps
        uuids += _uuids
    print("[original] Num Exps: ",len(exps))

    # -- run exps --
    results = cache_io.run_exps(exps,train.run,uuids=uuids,preset_uuids=True,
                                name=".cache_io/comp_graphs/train",
                                version=version,skip_loop=False,
                                clear_fxn=clear_fxn,
                                clear=False,enable_dispatch="slurm",
                                records_fn=
                                ".cache_io_pkl/comp_graphs/train.pkl",
                                records_reload=False,use_wandb=False,
                                proj_name="comp_graphs_train")



if __name__ == "__main__":
    main()
