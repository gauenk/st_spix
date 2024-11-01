"""

   Possible issue with original split

"""
import torch as th
import torchvision.utils as tv_utils
from pathlib import Path

def main():
    # -- root --
    root = Path("output/inspect_prop_split")
    if not root.exists(): root.mkdir()

    # -- read --
    spix = th.load("output/spix_t.pth")[0]
    print(spix.max())
    prop_seg = th.load("prop_seg.pth").state_dict()['0']

    # -- compare --
    a = th.sum(spix==96)
    b = th.sum(prop_seg==96)
    c = th.sum(prop_seg==96+112)
    print(a,b,c,b+c)

    # -- compare --
    prop_seg[th.where(prop_seg>=112)] -= 112
    delta = th.abs(spix != prop_seg)*1.

    # -- save --
    print(root)
    tv_utils.save_image(delta[None,None,:],root/"delta.png")

if __name__ == "__main__":
    main()
