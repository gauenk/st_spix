
import torch
from einops import rearrange
import torch as th
import bass_cuda

def pooling(tensor,spix,nspix=None):

    # -- unpack --
    if nspix is None: nspix = int((spix.max()+1).item())
    dtype = tensor.dtype
    device = tensor.device
    tensor = rearrange(tensor,'b f h w -> b h w f')
    B,H,W,F = tensor.shape

    # -- allocate --
    down = th.zeros((B,nspix,F),device=device,dtype=dtype)
    counts = th.zeros((B,nspix),device=device,dtype=dtype)

    # -- prepare for scatter --
    tensor = tensor.reshape(B,-1,F)
    spix = spix.reshape(B,-1).long()
    spix_e = spix[:,:,None].expand(-1,-1,F)
    assert th.all(spix>=0)

    # -- scatter add --
    down = th.scatter_add(down,1,spix_e,tensor)
    counts = th.scatter_add(counts,1,spix,th.ones_like(tensor[:,:,0]))
    down = down / (counts[:,:,None] + 1e-8)
    # print(down.shape)
    # print(th.mean(1.*th.isnan(down[...,0])))
    # print(th.sum(1.*th.isnan(down[...,0])))
    down[th.isnan(down)] = 0.
    # exit()

    # -- gather --
    pooled = th.gather(down,1,spix_e).reshape(B,H,W,F)
    pooled = rearrange(pooled,'b h w f -> b f h w')

    return pooled,down

class SuperpixelPooling(torch.autograd.Function):

    @staticmethod
    def forward(ctx, tensor, spix, nspix=None):
        if nspix is None: nspix = (spix.max()+1).item()
        fwd = bass_cuda.sp_pooling_fwd
        tensor = rearrange(tensor,'b f h w -> b h w f').contiguous()
        pooled,downsampled,counts = fwd(tensor.contiguous(),spix.contiguous(),nspix)
        pooled = rearrange(pooled,'b h w f -> b f h w').contiguous()
        # print("tensor.shape, pooled.shape, downsampled.shape: ",
        #       tensor.shape, pooled.shape, downsampled.shape)

        # -- compare --
        # ds_v0 = downsampled
        # counts_v0 = counts
        # B,H,W,F = tensor.shape
        # ds_v1 = th.zeros_like(ds_v0)
        # counts_v1 = th.zeros_like(counts_v0)
        # img_r = tensor.reshape(B,-1,F)

        # inds_r = spix.reshape(B,-1)[:,:,None].expand(-1,-1,F).long()
        # ds_v1 = th.scatter_add(ds_v1,1,inds_r,img_r)

        # inds_r = spix.reshape(B,-1).long()
        # ones = th.ones_like(img_r[:,:,0])
        # counts_v1 = th.scatter_add(counts_v1,1,inds_r,ones)
        # print("counts_v0.shape: ",counts_v0.shape)
        # print("counts_v1.shape: ",counts_v1.shape)
        # diff = th.sum((counts_v1 - counts_v0)**2).item()
        # print("diff: ",diff)

        # ds_v1 = ds_v1 / (counts[:,:,None] + 1e-10)

        # # ds_v1 = th.scatter(ds_v1,1,spix_r,img_r)
        # print("ds_v1.shape: ",ds_v1.shape)
        # diff = th.sum((ds_v1 - ds_v0)**2).item()
        # print("diff: ",diff)
        # exit()

        # -- end compare --
        ctx.save_for_backward(tensor,counts,spix)
        ctx.nspix = nspix
        ctx.shape = tensor.shape
        # print("pooled.shape: ",pooled.shape)
        # print("downsampled.shape: ",downsampled.shape)
        return pooled,downsampled

    @staticmethod
    def backward(ctx, pooled_grad, ds_grad):

        # -- unpack --
        tensor,counts,spix = ctx.saved_tensors
        nspix = ctx.nspix
        B,H,W,F = ctx.shape

        # -- gather counts --
        spix_r = spix.reshape(B,-1).long()
        counts = th.gather(counts,1,spix_r).reshape(B,H,W)

        # -- gather dowsampled grads --
        ds_grad = ds_grad.reshape(B,-1,F)
        inds = spix_r[:,:,None].expand(-1,-1,F)
        ds_grad = th.gather(ds_grad,1,inds).reshape(B,H,W,F)

        # -- guess --
        tensor_grad = (pooled_grad+ds_grad) / (counts+1e-10)

        # exit()
        # # ds_grad = th.gather(ds_grad,spix[None,:].repeat(3,1,1))

        # # bwd = bass_cuda.sp_pooling_bwd
        # # tensor_grad = bwd(pooled_grad,ds_grad,
        # #                   counts,spix,B,H,W,F,nspix)
        return tensor_grad, None, None

