import os
import copy
dcopy = copy.deepcopy
import pandas as pd
import torch as th
import numpy as np
from easydict import EasyDict as edict

from dev_basics.trte import bench
# from superpixel_paper.deno_trte.train import load_model,extract_defaults,config_via_spa
from st_spix.models import load_model
import st_spix.trte_utils as utils

from dev_basics.utils.timer import ExpTimer,TimeIt
from dev_basics.utils.gpu_mem import GpuMemer,MemIt


import cache_io

def run_bench(cfg):
    # cfg = extract_defaults(cfg)
    # load_model(cfg)
    # config_via_spa(cfg)
    vshape = (cfg.batch_size,3,96,96)

    device = "cuda"
    model = load_model(cfg).cuda()
    # -- load flow function --
    flow_fxn = utils.load_flow_fxn(cfg,device)
    vid = th.zeros(vshape).to(device)
    flows,fflow = flow_fxn(vid)
    # print(flows.shape,fflow.shape)

    # -- timer/memer --
    timer = ExpTimer()
    memer = GpuMemer()

    th.cuda.empty_cache()
    th.cuda.synchronize()
    # -- bench fwd --
    with TimeIt(timer,"fwd"):
        with MemIt(memer,"fwd"):
            output = model(vid,flows,fflow)['deno']

    th.cuda.synchronize()
    # -- compute grad --
    tgt = th.randn_like(output)
    loss = th.mean((tgt - output)**2)
    th.cuda.synchronize()

    # -- bench fwd --
    with TimeIt(timer,"bwd"):
        with MemIt(memer,"bwd"):
            loss.backward()

    # _fwd = model.forward

    # def fwd(vid):
    #     return _fwd(vid[0,0],flows,fflow)['deno']
    # model.forward = fwd
    # vshape = (cfg.batch_size,1,3,256,256)
    # summ = bench.summary_loaded(model,vshape,with_flows=False)
    # print(memer)

    summ = edict()
    summ.fwd_time = timer['fwd']
    summ.bwd_time = timer['bwd']
    summ.fwd_mem = memer['fwd']['alloc']
    summ.bwd_mem = memer['bwd']['alloc']
    return summ

def get_cfg_triple():
    cfg0 = edict({"mname":"sconv_deno","attn_type":"soft",
                  "dim":3,"sp_grad_type":"fixed_spix","lname":"deno",
                  "dname":"davis","patch_size":96,"nepochs":200,
                  "decays":[[75,150]],"kernel_size":15,"sigma":30,
                  "use_kernel_renormalize":False,
                  "use_kernel_reweight":True,
                  "seed":123,"spix_loss_type":"mse",
                  "sp_type":"bass","bass_prop":False,
                  "spix_loss_target":"pix",
                  "dist_type":"l2","tag":"v0.10","flow_method":"raft",
                  "window_time":0,"batch_size":30})
    cfg1 = dcopy(cfg0)
    # cfg1['mname'] = "sconv_deno" # testing the test :D
    cfg1['bass_prop'] = True
    cfg2 = dcopy(cfg0)
    cfg2['mname'] = "simple_conv"
    return cfg0,cfg1,cfg2

def main():
    cfg0,cfg1,cfg2 = get_cfg_triple()
    # model0 = load_model(cfg0).to(device)
    # model1 = load_model(cfg1).to(device)
    # model2 = load_model(cfg2).to(device)
    res2 = run_bench(cfg2)
    res0 = run_bench(cfg0)
    res1 = run_bench(cfg1)
    print(res0)
    print(res1)
    print(res2)

    # exp_fn = "exps/trte_deno/train_bench.cfg"
    # cache_fn = ".cache_io_exps/trte_deno/bench/"
    # exps,uuids = cache_io.train_stages.run(exp_fn,cache_fn,
    #                                        fast=False,update=True)
    # # for e in exps: e.batch_size = 1
    # # for e in exps: e.tag = "v0.01"
    # results = cache_io.run_exps(exps,run_bench,uuids=uuids,preset_uuids=True,
    #                             name=".cache_io/trte_deno/bench",
    #                             version="v1",skip_loop=False,
    #                             clear=False,enable_dispatch="slurm",
    #                             records_fn=".cache_io_pkl/trte_deno/bench.pkl",
    #                             records_reload=False,use_wandb=False,
    #                             proj_name="superpixels_deno_bench")
    # print(results['batch_size'].unique())

    # df = results.rename(columns={"spa_version":"spav","gen_sp_type":"gsp",
    #                              "timer_fwd_nograd":"t_fwd_ng",
    #                              "timer_fwd":"t_fwd",
    #                              "timer_bwd":"t_bwd",
    #                              "trainable_params":"params",
    #                              "learn_attn_scale":"las"})
    # df['params'] = df['params']*10**3
    # df['a_params'] = df['params'] - .195
    # fields0 = ["spav","gsp","las","params","a_params","t_fwd","t_bwd","alloc_fwd","alloc_bwd","seed"]
    # df = df[fields0]
    # fields = ["spav","gsp","las","params","a_params",
    #           "t_fwd","t_bwd","alloc_fwd","alloc_bwd"]

    # copy_fields = ["spav","gsp","las","params","a_params"]
    # ave_fields = ["t_fwd","t_bwd","alloc_fwd","alloc_bwd"]
    # df = df.groupby(copy_fields).agg({f:'mean' for f in ave_fields})
    # df['t_fwd'] = df['t_fwd']*10**3
    # df['t_bwd'] = df['t_bwd']*10**3
    # print(df)

    # df = df.reset_index(drop=False)
    # print(df)
    # # -- format --
    # fields = ave_fields
    # order_las = [True,True,False,False,False,True,False]
    # order_spav = ["ssna","ssna","ssna","ssna","sna","nat","nat"]
    # order_gsp = ["ssn","modulated","ssn","modulated","default","none","none"]
    # for f in fields:
    #     # df[['spav','las',f]]
    #     spav = df['spav'].to_numpy()
    #     las = df['las'].to_numpy()
    #     gsp = df['gsp'].to_numpy()
    #     finfo =  df[f].to_numpy()
    #     msg = ""
    #     for _las,_spav,_gsp in zip(order_las,order_spav,order_gsp):
    #         bool0 = np.logical_and(_las == las,_spav==spav)
    #         if _gsp != "none":
    #             bool1 = np.logical_and(_gsp == gsp,bool0)
    #         else:
    #             bool1 = bool0
    #         idx = np.where(bool1)[0]
    #         fmt = "%2.2f" % finfo[idx].item()
    #         msg += " " + str(fmt) + " &"
    #     print(msg)


    # for field,gdf in df.groupby(fields):
    #     print(fields)
    #     print(gdf)
    #     # df = df[fields]
    #     # print(df)

if __name__ == "__main__":
    main()
