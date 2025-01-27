

import os
import torch as th
import numpy as np
from einops import rearrange


from typing import Any, Callable, List, Optional
from torch import nn, Tensor

from pathlib import Path
import torchvision.utils as tv_utils
import torchvision.io as tvio

from torchvision.transforms.functional import InterpolationMode
from torchvision.transforms.functional import resize as _resize
def resize(img,size): return _resize(img,size,InterpolationMode.NEAREST)


def neigh_attention(input: Tensor, ps: int) -> Tensor:
    pass

def global_attention(img: Tensor, ps: int) -> Tensor:

    # -- get patches --
    B,F,H,W = img.shape
    msg = '1 f l -> l f'
    # unfold = th.nn.Unfold(ps, dilation=1, padding=ps//2, stride=1)
    # patches = rearrange(unfold(img),msg).to("cuda")
    patches = rearrange(img,'1 f h w -> (h w) f').to("cuda")

    # -- attention map --
    attn = th.cdist(patches,patches) # l x l
    attn = th.softmax(-20.*attn,1)
    deno = attn @ patches
    deno = deno.cpu()

    # -- shape for output --
    deno = rearrange(deno,'(h w) f -> 1 f h w',h=H)

    return deno


def shifted_window_attention(
    input: Tensor,
    window_size: List[int],
) -> Tensor:
    """
    Window based multi-head self attention (W-MSA) module with relative position bias.
    It supports both of shifted and non-shifted window.
    Args:
        input (Tensor[N, H, W, C]): The input tensor or 4-dimensions.
        qkv_weight (Tensor[in_dim, out_dim]): The weight tensor of query, key, value.
        proj_weight (Tensor[out_dim, out_dim]): The weight tensor of projection.
        relative_position_bias (Tensor): The learned relative position bias added to attention.
        window_size (List[int]): Window size.
        num_heads (int): Number of attention heads.
        shift_size (List[int]): Shift size for shifted window attention.
        attention_dropout (float): Dropout ratio of attention weight. Default: 0.0.
        dropout (float): Dropout ratio of output. Default: 0.0.
        qkv_bias (Tensor[out_dim], optional): The bias tensor of query, key, value. Default: None.
        proj_bias (Tensor[out_dim], optional): The bias tensor of projection. Default: None.
        logit_scale (Tensor[out_dim], optional): Logit scale of cosine attention for Swin Transformer V2. Default: None.
        training (bool, optional): Training flag used by the dropout parameters. Default: True.
    Returns:
        Tensor[N, H, W, C]: The output tensor after shifted window attention.
    """
    num_heads = 1
    B, H, W, C = input.shape
    # pad feature maps to multiples of window size
    pad_r = (window_size[1] - W % window_size[1]) % window_size[1]
    pad_b = (window_size[0] - H % window_size[0]) % window_size[0]
    x = F.pad(input, (0, 0, 0, pad_r, 0, pad_b))
    _, pad_H, pad_W, _ = x.shape
    shift_size = th.tensor([8,8])

    shift_size = shift_size.copy()
    # If window size is larger than feature size, there is no need to shift window
    if window_size[0] >= pad_H:
        shift_size[0] = 0
    if window_size[1] >= pad_W:
        shift_size[1] = 0

    # cyclic shift
    if sum(shift_size) > 0:
        x = torch.roll(x, shifts=(-shift_size[0], -shift_size[1]), dims=(1, 2))

    # partition windows
    num_windows = (pad_H // window_size[0]) * (pad_W // window_size[1])
    x = x.view(B, pad_H // window_size[0], window_size[0], pad_W // window_size[1], window_size[1], C)
    x = x.permute(0, 1, 3, 2, 4, 5).reshape(B * num_windows, window_size[0] * window_size[1], C)  # B*nW, Ws*Ws, C

    # multi-head attention
    qkv = th.cat([x,x,x])
    qkv = qkv.reshape(x.size(0), x.size(1), 3, num_heads, C // num_heads).permute(2, 0, 3, 1, 4)
    q, k, v = qkv[0], qkv[1], qkv[2]
    q = q * (C // num_heads) ** -0.5
    attn = q.matmul(k.transpose(-2, -1))

    if sum(shift_size) > 0:
        # generate attention mask
        attn_mask = x.new_zeros((pad_H, pad_W))
        h_slices = ((0, -window_size[0]), (-window_size[0], -shift_size[0]), (-shift_size[0], None))
        w_slices = ((0, -window_size[1]), (-window_size[1], -shift_size[1]), (-shift_size[1], None))
        count = 0
        for h in h_slices:
            for w in w_slices:
                attn_mask[h[0] : h[1], w[0] : w[1]] = count
                count += 1
        attn_mask = attn_mask.view(pad_H // window_size[0], window_size[0], pad_W // window_size[1], window_size[1])
        attn_mask = attn_mask.permute(0, 2, 1, 3).reshape(num_windows, window_size[0] * window_size[1])
        attn_mask = attn_mask.unsqueeze(1) - attn_mask.unsqueeze(2)
        attn_mask = attn_mask.masked_fill(attn_mask != 0, float(-100.0)).masked_fill(attn_mask == 0, float(0.0))
        attn = attn.view(x.size(0) // num_windows, num_windows, num_heads, x.size(1), x.size(1))
        attn = attn + attn_mask.unsqueeze(1).unsqueeze(0)
        attn = attn.view(-1, num_heads, x.size(1), x.size(1))

    attn = F.softmax(attn, dim=-1)
    # attn = F.dropout(attn, p=attention_dropout, training=training)
    x = attn.matmul(v).transpose(1, 2).reshape(x.size(0), x.size(1), C)
    # x = F.linear(x, proj_weight, proj_bias)
    # x = F.dropout(x, p=dropout, training=training)

    # reverse windows
    x = x.view(B, pad_H // window_size[0], pad_W // window_size[1], window_size[0], window_size[1], C)
    x = x.permute(0, 1, 3, 2, 4, 5).reshape(B, pad_H, pad_W, C)

    # reverse cyclic shift
    if sum(shift_size) > 0:
        x = torch.roll(x, shifts=(shift_size[0], shift_size[1]), dims=(1, 2))

    # unpad features
    x = x[:, :H, :W, :].contiguous()
    return x


def save_swin_patches(root,img):
    H,W = img.shape[-2:]
    a = img[...,:H//2,:W//2]
    b = img[...,H//2:,:W//2]
    c = img[...,:H//2,W//2:]
    d = img[...,H//2:,W//2:]
    print("a.shape: ",a.shape)
    print("b.shape: ",b.shape)
    tv_utils.save_image(resize(a,(256,256)),str(root/"swin_a.png"))
    tv_utils.save_image(resize(b,(256,256)),str(root/"swin_b.png"))
    tv_utils.save_image(resize(c,(256,256)),str(root/"swin_c.png"))
    tv_utils.save_image(resize(d,(256,256)),str(root/"swin_d.png"))


def main():

    # -- save --
    print("PID: ",os.getpid())
    root = Path("./output/viz_deno_spaces")
    if not root.exists(): root.mkdir(parents=True)

    # -- get image --
    img_fn = Path("data/crop_cat_chicken/image-026.png")
    img = tvio.read_image(str(img_fn))[None,]/255.
    # sw,sh = 566,1116-10
    img = img[:,:,40:40+160,120:120+160]
    H,W = img.shape[-2:]
    print("img.shape: ",img.shape)
    noisy = img + (20./255.)*th.randn_like(img)

    # -- save em --
    tv_utils.save_image(resize(img,(256,256)),str(root/"clean.png"))
    tv_utils.save_image(noisy,str(root/"noisy.png"))
    # noisy = img + (20./255.)*th.randn_like(img) # so effect is easier to see.

    # # -- save swin layers --
    # shift_size = 8
    # img = th.roll(img, shifts=(-shift_size, -shift_size), dims=(2, 3))
    # tv_utils.save_image(img,str(root/"swin.png"))

    save_swin_patches(root,img)

    # -- denoise with opts --
    # ps = 3
    # deno_g = global_attention(noisy,ps)
    # # deno_s = shifted_window_attention(
    # # deno_n = deno_g

    # -- info --
    # print("deno_g.shape: ",deno_g.shape)

    # -- save --
    # tv_utils.save_image(deno_g,str(root/"deno_g.png"))
    # tv_utils.save_image(deno_s,str(root/"deno_s.png"))
    # tv_utils.save_image(deno_n,str(root/"deno_n.png"))


if __name__ == "__main__":
    main()
