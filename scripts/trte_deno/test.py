
"""

Training

"""


# -- sys --
import os
import numpy as np
import pandas as pd

# -- training --
from st_spix.trte import test

# -- caching results --
import cache_io


import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.patches import Rectangle

def viz_psnrs_across_frame_index(results):

    # -- .. --
    results = results.fillna(value=-1)
    # results = results[['bass_prop','sigma','name'] + pnames]
    results = results[['bass_prop','sigma','name',"deno_psnr"]]
    # print(results)

    # -- init plot --
    dpi = 300
    ginfo = {'wspace':0.01, 'hspace':0.1,
             "top":0.80,"bottom":0.18,"left":.10,"right":0.98}
    fig,ax = plt.subplots(1,1,figsize=(5,2.5),gridspec_kw=ginfo,dpi=dpi)
    # fig,ax = plt.subplots(1,1,figsize=(4,4))

    # -- sigma --
    colors = {10:"orange",20:"red",30:"blue"}
    linestyles = ['solid','dashed']
    for sigma,sdf in results.groupby("sigma"):
        color = colors[sigma]
        for bprop,bdf in sdf.groupby("bass_prop"):
            psnrs = np.stack(bdf["deno_psnr"].to_numpy())
            num = np.sum(psnrs>=0,axis=0)
            psnrs_nan = np.where(psnrs >= 0, psnrs, np.nan)
            means = np.nanmean(psnrs_nan, axis=0)
            cut = np.max(np.where(num == 30)[0])+1
            means = means[:cut]
            ls = linestyles[0] if bprop is True else linestyles[1]
            ax.plot(np.arange(len(means)),means,
                    label=sigma,color=color,linestyle=ls)

    # -- axis labels --
    ax.set_ylabel("PSNR",fontsize=12)
    ax.set_xlabel("Frame Index",fontsize=12)

    # -- custom legend --

    # Create the custom lines for the legend
    line_bist = Line2D([0], [0], color='black', linestyle='--', label="BIST")
    line_bass = Line2D([0], [0], color='black', linestyle='-', label="BASS")
    # Create custom lines for the color legend
    line_orange = Line2D([0],[0],color='orange', linestyle='-', label=r"$10$")
    line_red = Line2D([0], [0], color='red', linestyle='-', label=r"$20$")
    line_blue = Line2D([0], [0], color='blue', linestyle='-', label=r"$30$")


    # # Add the first legend for line style
    # legend1 = ax.legend(handles=[line_bist, line_bass],
    #                     bbox_to_anchor=(0.15, 1.18),  # Shift to the top-right
    #                     ncol=2,frameon=False,fontsize=12,title_fontsize=12,
    #                     title="Line Style", loc='upper left',framealpha=0.)

    # # Add the second legend for color
    # legend2 = ax.legend(handles=[line_orange, line_red, line_blue],
    #                     bbox_to_anchor=(1.0, 0.8),  # Shift to the top-left
    #                     ncol=3,frameon=False,fontsize=12,title_fontsize=12,
    #                     title=r"$\sigma^2$", loc='upper right',framealpha=0.)

    # # Add both legends to the axes


    # Legend Elements
    legend1 = ax.legend(handles=[line_bist, line_bass],
                        bbox_to_anchor=(-0.1, 1.40),  # Shift to the top-right
                        ncol=2,frameon=False,fontsize=12,title_fontsize=12,
                        title="Method", loc='upper left',framealpha=0.)
    ax.add_artist(legend1)

    # legend_elements = [
    #     Line2D([0], [0], color='black', linestyle='dashed', label='BIST'),
    #     Line2D([0], [0], color='black', label='BASS')
    # ]

    # Add square/block markers for the sigma values
    legend_elements = [
        Rectangle((0, 0), 1, 1, color='orange', label='10'),
        Rectangle((0, 0), 1, 1, color='red', label='20'),
        Rectangle((0, 0), 1, 1, color='blue', label='30')
    ]


    # ....
    # line_legend = plt.legend(handles=legend_elements[:2], loc='upper left')
    # plt.gca().add_artist(line_legend)
    plt.legend(handles=legend_elements, loc='upper left',
               bbox_to_anchor=(0.45, 1.38),  # Shift to the top-right
               title='Noise Intensity', framealpha=0., ncol=3)

    # -- save --
    fname = "output/viz_psnrs_across_frame_index.png"
    plt.savefig(fname,dpi=300,transparent=True)

def main():

    # -- start info --
    verbose = True
    pid = os.getpid()
    print("PID: ",pid)
    read_testing = False

    # -- get/run experiments --
    refresh = True and not(read_testing)
    def clear_fxn(num,cfg): return False
    read_test = cache_io.read_test_config.run

    # -- load experiments --
    train_fn_list = [
        # "exps/trte_deno/train.cfg",
        # "exps/trte_deno/train_spix.cfg",
        # "exps/trte_deno/train_empty.cfg",
        "exps/trte_deno/train_sconv.cfg",
    ]
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

        # -- remove bass_prop --
        # del tr_exp["train_grid"]["mesh0"]["listed8"]["bass_prop"]

        # -- make bass prop false --
        # tr_exp["train_grid"]["mesh0"]["listed8"]["bass_prop"] = False
        # tr_exp = cache_io.fill_test_shell(tr_fn,te_fn)
        # _exps = read_test(tr_exp,".cache_io_exps/trte_deno/test",
        #                   reset=refresh,skip_dne=False,keep_dne=is_empty)
        # _exps,_uuids = cache_io.get_uuids(_exps,".cache_io/trte_deno/test",
        #                                   read=not(refresh),no_config_check=False)
        # exps += _exps
        # uuids += _uuids


    # -- info --
    # print("len(exps): ",len(exps))
    # exit()

    # -- run exps --
    results = cache_io.run_exps(exps,test.run,uuids=uuids,preset_uuids=True,
                                name=".cache_io/trte_deno/test",
                                version="v1",skip_loop=False,clear_fxn=clear_fxn,
                                clear=False,enable_dispatch="slurm",
                                records_fn=".cache_io_pkl/trte_deno/test.pkl",
                                records_reload=True and not(read_testing),
                                use_wandb=False,proj_name="trte_deno_test")

    # -- view results --
    # pnames = ["deno_psnr_%d"%i for i in range(130)]
    # results[pnames] = results["deno_psnr"].to_list()
    results = results.fillna(value=-1)
    viz_psnrs_across_frame_index(results)
    # results = results[['bass_prop','sigma','name'] + pnames]
    # return


    # -- save! --
    rename = {"spix_loss_type":"spix_l","deno_spix_alpha":"spix_a","spix_loss_compat":"spix_c","use_kernel_reweight":"krw","net_depth":"nd","kernel_size":"ks"}
    results = results.rename(columns=rename)


    psnrs = np.stack(results['deno_psnr'].to_numpy())
    psnrs = np.where(psnrs >= 0, psnrs, np.nan)
    psnrs = np.nanmean(psnrs, axis=1)
    results = results.drop('deno_psnr', axis=1)
    results['deno_psnr'] = psnrs

    ssims = np.stack(results['deno_ssim'].to_numpy())
    ssims = np.where(ssims >= 0, ssims, np.nan)
    ssims = np.nanmean(ssims, axis=1)
    results = results.drop('deno_ssim', axis=1)
    results['deno_ssim'] = ssims

    print(results[['deno_psnr','deno_ssim']])
    print(results.columns)
    vfields0 = ["asa","br","bp"]
    # vfields0 = ["asa","br"]
    vfields1 = ["pooled_psnr","pooled_ssim"]
    vfields2 = ["deno_psnr","deno_ssim"]
    vfields = vfields0 + vfields1 + vfields2
    # gfields = ["attn_type"]
    # gfields = ["sp_type","spix_l","spix_a","spix_c"]
    gfields = ["mname","sigma","bass_prop","krw","nd","ks"]
    results = results[vfields+gfields]
    # results = results.iloc[:10]
    print(results)
    results0 = results.groupby(gfields, as_index=False).agg(
        {k:['mean'] for k in vfields0})
    print(results0)
    # results1 = results.groupby(gfields, as_index=False).agg(
    #     {k:['mean','std'] for k in vfields1})
    # print(results1)
    results2 = results.groupby(gfields, as_index=False).agg(
        {k:['mean','std'] for k in vfields2})
    print(results2)



if __name__ == "__main__":
    main()
