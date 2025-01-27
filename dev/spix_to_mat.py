"""

   Convert a directory of segmentations into a ".mat" file

"""

import os
import pandas as pd
import numpy as np
import numpy.random as npr
from scipy.io import savemat
# import h5py
import argparse
from pathlib import Path

def read_spix(root):
    # -- read all image files --
    root = Path(root)
    files = []
    for fn in root.iterdir():
        suffix = fn.suffix
        if not(suffix in [".csv"]):
            continue
        files.append(fn.name)

    # -- sort by number --
    files = sorted(files, key=lambda x: int(os.path.splitext(x)[0]))

    # -- sort by number --
    vid = []
    for fn in files:
        img = np.array(pd.read_csv(root/fn,header=None))
        vid.append(img)
    vid = np.stack(vid, axis=-1).astype(np.uint32)
    print(vid.shape)
    return vid

def main():

    # -- Create an argument parser --
    parser = argparse.ArgumentParser(description='Save frames to a MATLAB v7.3 .mat file.')
    parser.add_argument('filepath', type=str, help='Path to the directory of superpixel segmentations')
    parser.add_argument('outname', type=str, help='The name of the output file')
    args = parser.parse_args()

    # -- ensure output format --
    outname = args.outname
    if not outname.endswith(".mat"):
        outname = outname + ".mat"

    # -- read files --
    spix = read_spix(args.filepath)

    # -- Save frames to the specified .mat file --
    savemat(outname, {'svMap': spix})
    # with h5py.File(outname, 'w') as f:
    #     f.create_dataset('svMap', data=spix, dtype='uint32')

    # -- info --
    print(f'Successfully saved frames from {args.filepath} to {outname}')

if __name__ == "__main__":
    main()
