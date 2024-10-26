"""

   A nice vizualition of a cute image in pieces...

   Cartoon superpixels

"""

import torch as th
import numpy as np
from einops import rearrange,repeat
from pathlib import Path
from functools import reduce
from torchvision.transforms import InterpolationMode
from torchvision.transforms.functional import resize



def read_puzzle_pieces():
    import torchvision.io as tvio
    root = Path("/home/gauenk/Documents/packages/st_spix/data/figures/")
    name = "puzzle%d.png"
    segs = []
    for idx in range(3):
        if idx == 2: continue
        fn = root / (name % idx)
        img = tvio.read_image(fn)/255.
        img = img[:3].mean(0)
        img = 1.*(img < 0.5)
        xargs,yargs = th.where(img>0.5)
        print(xargs)
        print(yargs)
        # print("img.shape: ",img.shape)
        # img = resize(img,(256,256),interpolation=InterpolationMode.BILINEAR)

        # print("img.shape: ",img.shape)
        # exit()
    exit()

    #     print(img.shape)
    #     F,H,W = img.shape
    #     vid.append(img)
    # for idx in range(len(vid)):
    #     vid[idx] = vid[idx][:,:minH,:minW]
    # vid = th.stack(vid)
    # # vid = resize(vid,(352,504)).to("cuda")
    # vid = resize(vid,(352,352)).to("cuda")
    # return vid


def read_elephants():
    import torchvision.io as tvio
    root = Path("/home/gauenk/Documents/packages/st_spix/data/figures/")
    name = "elephant_frame%d.png"
    vid = []
    # minH,minW = 704,1008
    minH,minW = 704,704
    for idx in range(3):
        if idx == 2: continue
        fn = root / (name % idx)
        img = tvio.read_image(fn)/255.
        print(img.shape)
        F,H,W = img.shape
        minH = H if H < minH else minH
        minW = W if W < minW else minW
        vid.append(img)
    for idx in range(len(vid)):
        vid[idx] = vid[idx][:,:minH,:minW]
    vid = th.stack(vid)
    # vid = resize(vid,(352,504)).to("cuda")
    vid = resize(vid,(352,352)).to("cuda")
    return vid



def main():
    pieces = read_puzzle_pieces()
    print(pieces.shape)

if __name__ == "__main__":
    main()
