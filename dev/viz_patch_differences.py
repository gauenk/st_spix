


import torch as th
import numpy as np
from einops import rearrange

from pathlib import Path


import torchvision.utils as tv_utils
import torchvision.io as tvio

from torchvision.transforms.functional import InterpolationMode
from torchvision.transforms.functional import resize as _resize
def resize(img,size): return _resize(img,size,InterpolationMode.NEAREST)

def main():

    # -- save --
    root = Path("./output/viz_patch_differences")
    if not root.exists(): root.mkdir(parents=True)

    # -- get image --
    img_fn = Path("../superpixel_paper/data/extra/bes.jpg")
    img = tvio.read_image(str(img_fn))[None,]/255.
    sw,sh = 566,1116-10
    img = img[:,:,566:1196+20,1116-10:1946+10]
    H,W = img.shape[-2:]
    print("img.shape: ",img.shape)

    # -- cropped --
    tv_utils.save_image(img,str(root/"cropped.png"))

    # -- get patches --
    ps = 11
    unfold = th.nn.Unfold(ps, dilation=1, padding=ps//2, stride=1)
    patches = rearrange(unfold(img),'1 (c ph pw) (h w) -> h w c ph pw',
                        h=H,ph=ps,pw=ps)
    # ps = 64
    # unfold = th.nn.Unfold(ps, dilation=1, padding=ps//2, stride=ps)
    # patches = rearrange(unfold(img),'1 (c ph pw) l -> l c ph pw',c=3,ph=ps)
    # print("patches.shape: ",patches.shape)

    # -- query patch --
    og_x,og_y = 1241,644
    q_x,q_y = og_x - sw,og_y-sh
    q_x,q_y = 130,130
    query = patches[q_y,q_x]
    _query = resize(query,(128,128))
    # _query = resize(patches[100],(128,128))
    tv_utils.save_image(_query,str(root/"query.png"))

    # -- search --
    print(query.shape,patches.shape)
    delta = th.mean((query[None,None] - patches).abs(),dim=(-3,-2,-1))
    delta = delta.ravel()
    vals,inds = th.topk(delta,k=5,largest=False)
    vals,inds = vals[1:],inds[1:]
    # print(vals)
    inds_w = inds % W
    inds_h = inds // W
    # print(inds_h)
    # print(inds_w)

    # -- viz keys & differences --
    query = query[None,:]
    keys = []
    deltas = []
    for (ind_h,ind_w) in zip(inds_h,inds_w):
        key = patches[ind_h,ind_w][None,:]
        print(th.abs(query - key).mean())
        keys.append(key)
        delta = th.abs(query - key).mean(-3,keepdim=True)
        deltas.append(delta)
    keys = th.cat(keys)
    deltas = th.cat(deltas)
    deltas = deltas / deltas.max()

    # -- save --
    _deltas = resize(deltas,(128,128))
    tv_utils.save_image(_deltas,str(root/"deltas.png"))
    _keys = resize(keys,(128,128))
    tv_utils.save_image(_keys,str(root/"keys.png"))
    for i in range(len(_deltas)):
        tv_utils.save_image(_deltas[[i]],str(root/("deltas_%d.png"%i)))
        tv_utils.save_image(_keys[[i]],str(root/("keys_%d.png"%i)))



if __name__ == "__main__":
    main()
