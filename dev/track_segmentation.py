"""

   Track a Segmentation using Superpixel Propogatino

"""

import numpy as np
import torch as th
from einops import rearrange,repeat

import matplotlib.pyplot as plt

def run_tracking(video):


    kwargs = {"use_bass_prop":False,"niters":30,"niters_seg":4,
              "sp_size":15,"pix_var":0.1,"alpha_hastings":0.01,
              "potts":8.,"sm_start":0}
    spix = run_bass(x,fflow,kwargs)
    sims = get_bass_sims(x,spix)


    pass

def main():
    pass

if __name__ == "__main__":
    main()
