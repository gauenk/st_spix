"""

   Development script to evaluate superpixels

"""

import st_spix
from st_spix.trte.test_spix import run as run_test
from easydict import EasyDict as edict

def main():

    # -- start info --
    verbose = True
    pid = os.getpid()
    print("PID: ",pid)

    # -- get/run experiments --
    def clear_fxn(num,cfg): return False
    refresh = True and not(read_testing)
    read_test = cache_io.read_test_config.run

    # -- load experiments --
    version = "v1"
    train_fn_list = ["exps/trte_deno/train_lin.cfg"]
    te_fn = "exps/trte_deno/test_shell.cfg"
    exps,uuids = [],[]
    for tr_fn in train_fn_list:
        is_empty = "empty" in tr_fn # special load; no network
        tr_exp = cache_io.fill_test_shell(tr_fn,te_fn)
        _exps = read_test(tr_exp,".cache_io_exps/trte_deno/test",
                          reset=refresh,skip_dne=False,keep_dne=is_empty)
        _exps,_uuids = cache_io.get_uuids(_exps,".cache_io/trte_deno/test",
                                          read=not(refresh),no_config_check=False)
        exps += _exps
        uuids += _uuids

    # -- run exps --
    results = cache_io.run_exps(exps,run_test,uuids=uuids,preset_uuids=True,
                                name=".cache_io/trte_deno/test",
                                version="v1",skip_loop=False,clear_fxn=clear_fxn,
                                clear=False,enable_dispatch="slurm",
                                records_fn=".cache_io_pkl/trte_deno/test.pkl",
                                records_reload=True and not(read_testing),
                                use_wandb=False,proj_name="trte_deno_test")
    print(results)
    exit()

    # # -- run exps --
    # run(cfg)
    # results = cache_io.run_exps(exps,train.run,uuids=uuids,preset_uuids=True,
    #                             name=".cache_io/trte_deno/train",
    #                             version=version,skip_loop=False,clear_fxn=clear_fxn,
    #                             clear=False,enable_dispatch="slurm",
    #                             records_fn=".cache_io_pkl/trte_deno/train.pkl",
    #                             records_reload=False,use_wandb=False,
    #                             proj_name="trte_deno_train")
    # cfg = edict()
    # # cfg.sp_type = "bass"
    # cfg.sp_type = "slic"

if __name__ == "__main__":
    main()
