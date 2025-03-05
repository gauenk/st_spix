import torch as th
from einops import rearrange
import torch.nn.functional as th_f
from torchvision.utils import draw_segmentation_masks

from st_spix.sp_pooling import pooling

from skimage.segmentation import mark_boundaries

def img4bass(img):
    img = rearrange(img,'... f h w -> ... h w f').flip(-1)
    img = img.contiguous()
    if img.ndim == 3: img = img[None,:]
    if img.max() <= 2:
        img = (255.*img).type(th.uint8)
    return img


def viz_spix(img_batch,spix_batch,nsp):
    masks = []
    img_batch = (img_batch*255.).clip(0,255).type(th.uint8)
    for img, spix in zip(img_batch, spix_batch):
        # _masks = img
        # masks.append(img)

        # _masks = th_f.one_hot(spix,nsp).T.bool()
        # _masks = rearrange(_masks,'c (h w) -> c h w',h=img.shape[1])
        # masks.append(draw_segmentation_masks(img, masks=_masks, alpha=0.5))

        img = rearrange(img,'c h w -> h w c').cpu().numpy()/255.
        spix = rearrange(spix,'(h w) -> h w',h=img.shape[0]).cpu().numpy()
        # print(img.shape,spix.shape)
        _masks = mark_boundaries(img,spix,mode="subpixel")
        _masks = th.from_numpy(_masks)
        _masks = rearrange(_masks,'h w c -> c h w')
        masks.append(_masks)

        # _masks = _masks[:3]
        # print("_masks.shape: ",_masks.shape)
        # masks.append(_masks)
    masks = th.stack(masks)#/255.
    # masks = th.stack(masks)/(1.*nsp)
    # masks = masks / masks.max()
    return masks


def spix_pool_vid(vid,spix):
    # -- prepare --
    # vid = rearrange(vid,'b f h w -> b h w f')
    vid = vid.contiguous()
    spix = spix.contiguous()
    nspix = spix.max()+1
    # print("labels.shape: ",labels.shape)

    # -- run --
    # import st_spix_cuda
    # fxn = st_spix_cuda.sp_pooling_fwd
    # pooled,downsampled = fxn(labels,spix,nspix)
    pooled,downsampled = pooling(vid,spix,nspix)

    return pooled,downsampled

# def sp_pool_from_spix(labels,spix,version="v1",return_ds=False):
#     if version == "v0":
#         return sp_pool_from_spix_v0(labels,spix)
#     elif version == "v1":
#         return sp_pool_from_spix_v1(labels,spix,return_ds)
#     else:
#         raise ValueError(f"Uknown spix pooling version [{version}]")

# def sp_pool_from_spix_v0(labels,spix):
#     sims_hard = th_f.one_hot(spix.long())*1.
#     sims_hard = rearrange(sims_hard,'b h w nsp -> b nsp (h w)')
#     labels_sp = sp_pool(labels,sims_hard)
#     return labels_sp

# def sp_pool_from_spix_v1(labels,spix,return_ds=False):

#     # -- prepare --
#     # labels = rearrange(labels,'b f h w -> b h w f')
#     labels = labels.contiguous()
#     spix = spix.contiguous()
#     nspix = spix.max()+1
#     # print("labels.shape: ",labels.shape)

#     # -- run --
#     # import st_spix_cuda
#     # fxn = st_spix_cuda.sp_pooling_fwd
#     # pooled,downsampled = fxn(labels,spix,nspix)

#     print("hey-yo.")
#     th.cuda.synchronize()
#     print(labels.shape,spix.shape,spix.max())
#     pooled,downsampled = pooling(labels,spix,nspix)
#     print("b.")
#     th.cuda.synchronize()
#     print("bye.")
#     # print("labels.shape,pooled.shape,downsampled.shape: ",
#     #       labels.shape,pooled.shape,downsampled.shape)
#     # exit()

#     # -- return --
#     # pooled = rearrange(pooled,'b h w f -> b f h w')
#     if return_ds:
#         return pooled,downsampled
#     else:
#         return pooled

# def sp_pool(labels,sims,re_expand=True):
#     assert re_expand == True,"Only true for now."

#     # -- normalize across #sp for each pixel --
#     sims_nmz = sims / (1e-15+sims.sum(-1,keepdim=True))# (B,NumSpix,NumPix) -> (B,NS,NP)
#     sims = sims.transpose(-1,-2)

#     # -- prepare labels --
#     W = labels.shape[-1]
#     labels = rearrange(labels,'b f h w -> b (h w) f')

#     # -- compute "superpixel pooling" --
#     labels_sp = sims @ (sims_nmz @ labels)

#     # -- reshape --
#     labels_sp = rearrange(labels_sp,'b (h w) f -> b f h w',w=W)
#     # print("a: ",labels.min(),labels.max())
#     # print("b: ",labels_sp.min(),labels_sp.max())

#     return labels_sp

# def sp_pool_v0(img_batch,spix_batch,sims,S,nsp,method):
#     pooled = []
#     for img, spix in zip(img_batch, spix_batch):
#         img = rearrange(img,'f h w -> h w f')
#         pool = sp_pool_img(img,spix,sims,S,nsp,method)
#         pooled.append(rearrange(pool,'h w f -> f h w'))
#     pooled = th.stack(pooled)
#     return pooled

# def sp_pool_img_v0(img,spix,sims,S,nsp,method):
#     H,W,F = img.shape
#     if method in ["ssn","sna"]:
#         sH,sW = (H+1)//S,(W+1)//S # add one for padding
#     else:
#         sH,sW = H//S,W//S # no padding needed

#     is_tensor = th.is_tensor(img)
#     if not th.is_tensor(img):
#         img = th.from_numpy(img)
#         spix = th.from_numpy(spix)

#     img = img.reshape(-1,F)
#     spix = spix.ravel()
#     # N = nsp
#     N = len(th.unique(spix))
#     assert N <= (sH*sW)

#     # -- normalization --
#     counts = th.zeros((sH*sW),device=spix.device)
#     ones = th.ones_like(img[:,0])
#     counts = counts.scatter_add_(0,spix,ones)

#     # -- pooled --
#     pooled = th.zeros((sH*sW,F),device=spix.device)
#     for fi in range(F):
#         pooled[:,fi] = pooled[:,fi].scatter_add_(0,spix,img[:,fi])

#     # -- exec normz --
#     pooled = pooled/counts[:,None]

#     # -- post proc --
#     pooled = pooled.reshape(sH,sW,F)
#     if not is_tensor:
#         pooled = pooled.cpu().numpy()

#     return pooled

def to_th(tensor):
    return th.from_numpy(tensor)
def swap_c(img):
    return rearrange(img,'... h w f -> ... f h w')

def mark_spix_vid(vid,spix,mode=None,color="blue"):
    if color == "blue":
        color = (0,0,1)
    else:
        color = (1,0,0)
    if mode is None:
        mode = "inner"
    # mode = "thick"
    # mode = "subpixel"
    if vid.ndim == 5:
        # print("Only using first video batch.")
        vid = vid[0]
    marked = []
    if th.is_tensor(spix):
        spix = spix.detach().cpu().numpy()
    if th.is_tensor(vid):
        vid = vid.detach().cpu().numpy()
    for ix,spix_t in enumerate(spix):
        img = rearrange(vid[ix],'f h w -> h w f')
        if mode is None:
            marked_t = mark_boundaries(img,spix_t,color=color,mode=mode)
        else:
            marked_t = mark_boundaries(img,spix_t,color=color,mode=mode)
        marked_t = to_th(swap_c(marked_t))
        marked.append(marked_t)
    marked = th.stack(marked)
    return marked

def fill_spix(vid,spix,spix_id):
    vid[:,0][th.where(spix == spix_id)] = 0.
    vid[:,1][th.where(spix == spix_id)] = 0.
    vid[:,2][th.where(spix == spix_id)] = 1.

def to_np(tensor):
    return tensor.cpu().numpy()
def to_th(tensor):
    return th.from_numpy(tensor)
def swap_c0(img):
    return rearrange(img,'... h w f -> ... f h w')
def swap_c3(img):
    return rearrange(img,'... f h w -> ... h w f')
def mark_spix(img,spix):
    marked = to_th(swap_c0(mark_boundaries(to_np(img),to_np(spix))))
    if th.any(spix==-1):
        args = th.where(spix==-1)
        marked[0][args] = 0
        marked[1][args] = 0
        marked[2][args] = 1.
    return marked

def mark_border(vid,border,color_index=0):
    vid = vid.clone()
    # print("vid.shape: ",vid.shape)
    # print("border.shape: ",border.shape)
    inds = th.where(border>0)
    for c in range(3):
        vid[:,c][inds] = c==color_index
    return vid

# def spix_to_params(vid,spix):
#     from .flow_utils import index_grid
#     T,F,H,W = vid.shape
#     print("vid.shape: ",vid.shape)
#     grid = index_grid(H,W,dtype=th.float,device="cuda",normalize=True)
#     print("grid.shape: ",grid.shape)
#     pooled,downsampled = pooling(grid,spix,nspix)


