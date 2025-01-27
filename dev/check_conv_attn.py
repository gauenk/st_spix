
# -- basic --
import os
import sys
import time
import math
import glob
import copy
import logging
import importlib
import argparse, yaml
from tqdm import tqdm
dcopy = copy.deepcopy
from pathlib import Path
from easydict import EasyDict as edict
import numpy as np

# -- pytorch --
import torch
import torch as th
import torch.nn as nn
import torch.nn.functional as F
from torch.optim.lr_scheduler import MultiStepLR, StepLR
from einops import rearrange,repeat

# -- helper --
from dev_basics.utils.timer import ExpTimer

# -- project imports --
from st_spix.data import load_data
from st_spix.losses import load_loss
from st_spix.models import load_model
import st_spix.trte_utils as utils
import st_spix.utils as base_utils
from st_spix import metrics
# from st_spix import isp
from st_spix.trte.train import load_flow_fxn



def get_cfg_pair():
    cfg0 = edict({"mname":"sconv_deno","attn_type":"soft",
                  "dim":3,"sp_grad_type":"fixed_spix","lname":"deno",
                  "dname":"davis","patch_size":96,"nepochs":200,
                  "decays":[[75,150]],"kernel_size":15,"sigma":30,
                  "use_kernel_renormalize":False,
                  "use_kernel_reweight":True,
                  "seed":123,"spix_loss_type":"mse",
                  "sp_type":"bass","spix_loss_target":"pix",
                  "dist_type":"l2","tag":"v0.10","flow_method":"raft","window_time":0})
    cfg1 = dcopy(cfg0)
    # cfg1['mname'] = "sconv_deno" # testing the test :D
    cfg1['mname'] = "simple_conv"
    return cfg0,cfg1

def share_weights(model0,model1):
    with th.no_grad():
        for name0,param0 in model0.named_parameters():
            for name1,param1 in model1.named_parameters():
                if name0 == name1:
                    param1.data[...] = th.randn_like(param1.data)
                    param0.data[...] = param1.data[...]

def share_weights_conv_lin(model_lin,model_conv):
    with th.no_grad():
        for name0,layer0 in model_lin.named_modules():
            # print(name0)
            for name1,layer1 in model_conv.named_modules():
                # print(name0,name1)
                if name0 == "sconv.0.linear" and name1 == "conv.0":
                    iweight = th.randn_like(layer1.weight)
                    ibias = th.randn_like(layer1.bias)
                    layer0.weight[...] = iweight.reshape_as(layer0.weight)
                    layer0.bias[...] = ibias.reshape_as(layer0.bias)
                    layer1.weight[...] = iweight
                    layer1.bias[...] = ibias

def main():

    device = "cuda"
    cfg0,cfg1 = get_cfg_pair()
    model0 = load_model(cfg0).to(device)
    model1 = load_model(cfg1).to(device)
    share_weights(model0,model1)
    share_weights_conv_lin(model0,model1)
    flow_fxn = load_flow_fxn(cfg0,device)


    x = th.randn((2,3,96,96)).to(device)
    flows,fflow = flow_fxn(x)
    out0 = model0(x,flows,fflow)["deno"]
    out1 = model1(x,flows,fflow)["deno"]
    delta = th.mean((out0 - out1)**2).item()
    print("delta: ",delta)

if __name__ == "__main__":
    main()
