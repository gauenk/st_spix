
import torch as th
from einops import rearrange
import prop_cuda

def video_pooling(tensor,spix):
    return SuperpixelPooling.apply(tensor,spix)

class SuperpixelPooling(th.autograd.Function):

    @staticmethod
    def forward(ctx, tensor, spix):
        assert tensor.ndim == 5,"Must be 5-dims: batch, frames, ftrs, height, width"

        # -- forward --
        fwd = prop_cuda.sp_video_pooling
        tensor = rearrange(tensor,'b t f h w -> b t h w f').contiguous()
        pooled,downsampled,counts = fwd(tensor.contiguous(),spix.contiguous())
        pooled = rearrange(pooled,'b t h w f -> b t f h w').contiguous()

        # -- end compare --
        ctx.save_for_backward(spix)
        return pooled,downsampled,counts

    @staticmethod
    def backward(ctx, pooled_grad, ds_grad):
        assert pooled_grad.ndim == 5,"Must be 5-dims: batch, frames, ftrs, height, width"

        # -- backward --
        spix = ctx.saved_tensors[0]
        fwd = prop_cuda.sp_video_pooling
        pooled_grad = rearrange(pooled_grad,'b t f h w -> b t h w f').contiguous()
        img_grad,downsampled,counts = fwd(pooled_grad.contiguous(),spix.contiguous())
        img_grad = rearrange(img_grad,'b t h w f -> b t f h w').contiguous()

        # -- backward downsampled --
        if not(ds_grad is None) or th.any(ds_grad.abs()>0):
            fwd = prop_cuda.downsampled_video_to_pooled
            img_grad_ds = fwd(ds_grad,spix)
            img_grad += rearrange(img_grad_ds,'b t h w f -> b t f h w').contiguous()

        return img_grad, None

