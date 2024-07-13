import torch as th
import numpy as np
import pandas as pd
import st_spix
import st_spix_cuda
import st_spix_original_cuda
from st_spix import flow_utils as futils
import torchvision.io as iio
from einops import rearrange,repeat
from skimage.segmentation import mark_boundaries
import torchvision.utils as tv_utils
import torch.nn.functional as th_f

import seaborn as sns
import matplotlib.pyplot as plt


import stnls

from dev_basics import flow
from dev_basics.utils.timer import ExpTimer

def sp_pool(labels,sims,re_expand=True):

    # -- normalize across #sp for each pixel --
    sims_nmz = sims / (1e-20+sims.sum(-1,keepdim=True))# (B,NumSpix,NumPix) -> (B,NS,NP)
    sims = sims.transpose(-1,-2)

    # -- prepare labels --
    W = labels.shape[-2]
    labels = rearrange(labels,'b h w f -> b (h w) f')

    # -- compute "superpixel pooling" --
    if re_expand:
        labels_sp = sims @ (sims_nmz @ labels)
        labels_sp = rearrange(labels_sp,'b (h w) f -> b h w f',w=W)
    else:
        labels_sp = sims_nmz @ labels

    return labels_sp

def save_flow(flow,name):
    flow = th.cat([flow,th.zeros_like(flow[:1])],0)[None,:]
    flow = flow.abs()/flow.abs().max()
    tv_utils.save_image(flow,name)

def swap_c(img):
    return rearrange(img,'h w f -> f h w')
def to_th(tensor):
    return th.from_numpy(tensor)

def pool_flow(spix,fflow):
    H,W = fflow.shape[-2:]
    fflow = rearrange(fflow,'f h w -> 1 h w f')
    sims_hard = th_f.one_hot(spix.long())*1.
    sims_hard = rearrange(sims_hard,'b h w nsp -> b nsp (h w)')
    fflow_s = sp_pool(fflow,sims_hard)
    fflow_s = rearrange(fflow_s,'1 h w f -> f h w')
    fflow = rearrange(fflow,'1 h w f -> f h w')
    return fflow_s

def shift_spix_flow(spix,fflow):
    H,W = fflow.shape[-2:]
    fflow = rearrange(fflow,'f h w -> 1 1 (h w) 1 f')
    fflow = fflow.flip(-1)
    fflow[...,0] = -fflow[...,0]
    fflow = th.cat([th.zeros_like(fflow[...,:1]),fflow],-1)
    ones = th.ones_like(fflow[...,0])
    spix = rearrange(spix,'1 h w -> 1 1 1 1 h w')
    # vid # [B HD T F H W] or [B T F' H W] with F' = (HD F) and HD = inds.shape[1]
    # weights.shape # B,HD,Q,K
    # inds.shape # B,HD,Q,K,3

    stacking = stnls.agg.NonLocalGather(1,1,itype="float")
    stack = stacking(spix*1.,ones,fflow).squeeze()[None,:]
    stack = stack.int()
    # print(stack.shape)
    # # rearrange(stack,'b 1 (h w) 1 f -> b h w f',
    # exit()
    return stack

def shift_spix_flow_v1(spix,flows_k,ws):
    # flows = rearrange(flows,'b hd t h w k f')
    ones = th.ones_like(flows_k[...,0])
    T = flows_k.shape[2]
    spix = repeat(spix,'1 h w -> 1 t 1 h w',t=T)
    # vid # [B HD T F H W] or [B T F' H W] with F' = (HD F) and HD = inds.shape[1]
    # weights.shape # B,HD,Q,K
    # inds.shape # B,HD,Q,K,3
    stacking = stnls.agg.NonLocalGather(1,1,itype="float")
    K = spix.max()
    half = ws*ws
    stack = stacking(1.*spix,ones,flows_k)[:,0,half:,1].int()
    # print("stack.shape: ",stack.shape)
    stack = th.mode(stack,1).values
    stack = stack[0] # no batch
    return stack

def shift_spix_flow_v2(spix,fflow,ws):
    H,W = fflow.shape[-2:]
    fflow = rearrange(fflow,'f h w -> 1 1 (h w) 1 f')
    fflow = fflow.flip(-1)
    fflow[...,0] = -fflow[...,0]
    # print("fflow.shape: ",fflow.shape)
    fflow[...] = fflow.mean(1,keepdim=True)
    fflow = th.cat([th.zeros_like(fflow[...,:1]),fflow],-1)
    ones = th.ones_like(fflow[...,0])
    spix = rearrange(spix,'1 h w -> 1 1 1 1 h w')
    stacking = stnls.agg.NonLocalGather(1,1,itype="float")
    stack = stacking(spix*1.,ones,fflow).squeeze()[None,:]
    stack = stack.int()
    return stack

def shift_mean_flow(spix,fflow,means):
    H,W = fflow.shape[-2:]
    fflow = rearrange(fflow,'f h w -> 1 h w f')
    sims_hard = th_f.one_hot(spix.long())*1.
    sims_hard = rearrange(sims_hard,'b h w nsp -> b nsp (h w)')
    fflow_s = sp_pool(fflow,sims_hard,re_expand=False)
    # print(means.shape)
    # print(fflow_s.shape)
    # exit()
    fflow_s[:,:,0] = -fflow_s[:,:,0]
    means[None,:,-2:] = means[None,:,-2:] + fflow_s
    return means

def shift_mean_flow_v1(spix,fflow,means):
    H,W = fflow.shape[-2:]
    fflow = rearrange(fflow,'f h w -> 1 h w f')
    sims_hard = th_f.one_hot(spix.long())*1.
    sims_hard = rearrange(sims_hard,'b h w nsp -> b nsp (h w)')
    fflow_s = sp_pool(fflow,sims_hard,re_expand=False)
    fflow_s[...] = fflow_s.mean(1,keepdim=True)
    # print(means.shape)
    # print(fflow_s.shape)
    # exit()
    fflow_s[:,:,0] = -fflow_s[:,:,0]
    means[None,:,-2:] = means[None,:,-2:] + fflow_s
    return means

def remap_spix(spix,device="cuda"):
    B,H,W = spix.shape
    num, letter = pd.factorize(spix.cpu().numpy().ravel())
    spix_remap = th.from_numpy(num).to(device).reshape((B,H,W)).type(th.int)
    # print(spix_remap.max(),spix_remap.min(),len(th.unique(spix_remap)))
    return spix_remap

def compute_means(labels,sims):
    # -- normalize across #sp for each pixel --
    sims_nmz = sims / (1e-20+sims.sum(-1,keepdim=True))# (B,NumSpix,NumPix) -> (B,NS,NP)
    sims = sims.transpose(-1,-2)

    # -- prepare labels --
    W = labels.shape[-2]
    labels = rearrange(labels,'b h w f -> b (h w) f')

    # -- compute "superpixel pooling" --
    labels_sp = sims_nmz @ labels

    return labels_sp

def compute_cov2d(labels,pooled,sims,prior_count):

    # -- normalize across #sp for each pixel --
    count = 1e-20 + sims.sum(-1,keepdim=True) + prior_count*50 - 3
    sims_nmz = sims / count#(B,NumSpix,NumPix) -> (B,NS,NP)
    sims = sims.transpose(-1,-2)
    prior_sigma_s_2 = prior_count*2

    # -- prepare labels --
    W = labels.shape[-2]
    pooled = rearrange(pooled,'b h w f -> b (h w) f')
    labels = rearrange(labels,'b h w f -> b (h w) f')
    diff00 = (labels[:,:,0] - pooled[:,:,0])**2+prior_sigma_s_2
    diff11 = (labels[:,:,1] - pooled[:,:,1])**2+prior_sigma_s_2
    diff01 = (labels[:,:,0] - pooled[:,:,1])*(labels[:,:,1] - pooled[:,:,0])
    det = diff11 * diff00 - diff01**2
    diffs = th.stack([diff11,diff01,diff00,th.log(det+1e-15)],-1)
    print(diffs.shape)

    # -- compute "superpixel pooling" --
    labels_sp = sims @ (sims_nmz @ diffs)
    labels_sp[...,:3] = labels_sp[...,:3]/(det[...,None]+1e-20)

    # -- reshape --
    labels_sp = rearrange(labels_sp,'b (h w) f -> b h w f',w=W)

    return labels_sp

def run_stnls(vid,acc_flows,ws,ps,full_ws=False):
    s0 = 1
    s1 = 1
    wt = 1
    k = -1
    vid = rearrange(vid,'b t f h w -> b t f h w').contiguous()
    # print(acc_flows.shape)
    # print("vid.shape: ",vid.shape)
    # exit()
    search_p = stnls.search.PairedSearch(ws,ps,k,
                                         nheads=1,dist_type="l2",
                                         stride0=s0,stride1=s1,
                                         self_action=None,use_adj=False,
                                         full_ws=full_ws,itype="float")
    _,flows_k = search_p.paired_vids(vid,vid,acc_flows,wt,skip_self=True)
    return flows_k

# -- read/init --
device = "cuda"
# vid = st_spix.data.davis_example()[:1,:3]
vid = st_spix.data.davis_example()[[0],:3]
# print("vid.shape: ",vid.shape)
img= (th.clip(vid[0,0],0.,1.)*255.).type(th.uint8)
img = rearrange(img,'f h w -> 1 h w f').to(device)
B,H,W,F = img.shape

img1= (th.clip(vid[0,1],0.,1.)*255.).type(th.uint8)
img1 = rearrange(img1,'f h w -> 1 h w f').to(device)
B,H,W,F = img.shape

# img = iio.read_image("./images/126039.jpg")#[:,:256,:256]
# img = img.flip(0)
# img[...] = 254


# print(img.shape)
# exit()
# means = th.zeros((B,H,W,5),dtype=th.float).to(device)
# cov = th.zeros((B,H,W,3),dtype=th.float).to(device)
# spix = th.zeros((B,H,W),dtype=th.int).to(device)
# print("init cuda device: ",spix.sum())
# print(spix.shape)
npix_in_side = 40
prior_sigma_s = npix_in_side**4
prior_count = npix_in_side**4
i_std = 0.018
# alpha,beta = 0.5,0.5
# alpha,beta = 2.,2.
alpha,beta = 0.001,10.

# -- init timer --
timer = ExpTimer()

timer.sync_start("flow")
flows = flow.run(vid,sigma=0.0,ftype="cv2",rescale=True)
fflow = flows.fflow[0,0]
acc_flows = stnls.nn.search_flow(flows.fflow,flows.bflow,1,1)
# print(acc_flows.shape)
# exit()
timer.sync_stop("flow")

# -- run --
st_spix.utils.seed_everything(0)
timer.sync_start("dev_bass")
spix,means,cov,counts = st_spix_original_cuda.bass_forward(img,npix_in_side,
                                                           i_std,alpha,beta)
timer.sync_stop("dev_bass")

st_spix.utils.seed_everything(0)
timer.sync_start("core_bass")
spix,means,cov,counts = st_spix_cuda.bass_forward(img,npix_in_side,
                                           i_std,alpha,beta)
timer.sync_stop("core_bass")

st_spix.utils.seed_everything(0)
timer.sync_start("dev_bass_2")
_spix,_means,_cov,_counts = st_spix_original_cuda.bass_forward(img,npix_in_side,
                                                       i_std,alpha,beta)
timer.sync_stop("dev_bass_2")

st_spix.utils.seed_everything(0)
timer.sync_start("a")
spix,means,cov,counts = st_spix_original_cuda.bass_forward(img,npix_in_side,
                                                           i_std,alpha,beta)
timer.sync_stop("a")



_spix_remap = remap_spix(_spix)
K = _spix_remap.max()+1
max_SP = K
spix,means,cov,counts = st_spix_cuda.bass_forward_refine(img,_spix_remap,
                                                         _means,_cov,_counts,
                                                         npix_in_side,
                                                         i_std,alpha,beta,0,K,max_SP)
_means_mod = _means
save_flow(fflow,"output/test_bass/fflow.png")
fflow_p = pool_flow(_spix_remap,fflow)
ws,ps = 7,5
# print("flows.fflow.shape,fflow_p.shape: ",flows.fflow.shape,fflow_p.shape)
flows.fflow[0,0] = fflow_p
acc_flows = stnls.nn.search_flow(flows.fflow,flows.bflow,1,1)
# exit()
flows_k = run_stnls(vid,acc_flows,ws,ps)
fflow_k = rearrange(flows_k[0,0,0,:,:,0,:2],'h w f -> f h w')
fflow_kp = pool_flow(_spix_remap,fflow_k)
print("flows_k.shape: ",flows_k.shape)
print("fflow_k.shape,fflow_p.shape,fflow.shape: ",fflow_k.shape,fflow_p.shape,fflow.shape)
# _means_mod = shift_mean_flow(_spix_remap,fflow_p,_means.clone())
_means_mod = shift_mean_flow_v1(_spix_remap,fflow_p,_means.clone())
# _spix_shift = shift_spix_flow(_spix_remap,fflow_p)
# _spix_shift = shift_spix_flow_v1(_spix_remap,flows_k,ws)
_spix_shift = shift_spix_flow_v2(_spix_remap,fflow_kp,ws)
save_flow(fflow_p,"output/test_bass/fflow_ave.png")
# _means_mod = shift_mean_flow(_spix_remap,fflow,_means.clone())

niters = 0
timer.sync_start("b")
spix_2,means_2,cov_2,counts_2 = st_spix_cuda.bass_forward_refine(img,_spix_shift,
                                                                 _means_mod,
                                                                 _cov,_counts,
                                                                 npix_in_side,
                                                                 i_std,alpha,beta,
                                                                 niters,K,max_SP)
timer.sync_stop("b")


timer.sync_start("c")
niters = 6
# _counts[...]= 0.
spix,means,counts,cov = st_spix_cuda.bass_forward_refine(img1,_spix_shift,
                                                         _means_mod,
                                                         _cov,_counts,
                                                         npix_in_side,
                                                         i_std,alpha,beta,
                                                         niters,K,max_SP)
timer.sync_stop("c")

print(timer)


# -- compare --
_spix = remap_spix(_spix)
img_alpha = th.cat([img/255.,1.*th.ones_like(img[...,:1])],-1)
img_alpha = rearrange(img_alpha,'b h w f -> b f h w')
warped_img = futils.flow_warp(img_alpha, fflow_kp[None,:], interp_mode='bilinear',
                       padding_mode='reflection', align_corners=True)
print("warped_img.shape: ",warped_img.shape)
tv_utils.save_image(warped_img,"./output/test_bass/warped.png")
# spix = remap_spix(spix)
print("Comparing # of Spix: ",len(th.unique(spix)),len(th.unique(_spix)))
args = th.where(spix != _spix)
print(spix[args])
print(_spix[args])
tv_utils.save_image(1.*(spix!=_spix),"./output/test_bass/spix.png")
# original spix on frame 0
marked0 = mark_boundaries(img[0].cpu().numpy(),_spix[0].cpu().numpy())
# init spix on frame 0
marked1 = mark_boundaries(img[0].cpu().numpy(),_spix_shift[0].cpu().numpy())
# init spix on frame 1
marked2 = mark_boundaries(img1[0].cpu().numpy(),spix_2[0].cpu().numpy())
# new spix on frame 1
marked3 = mark_boundaries(img1[0].cpu().numpy(),spix[0].cpu().numpy())
tv_utils.save_image(to_th(swap_c(marked0))[None,:],"./output/test_bass/marked0.png")
tv_utils.save_image(to_th(swap_c(marked1))[None,:],"./output/test_bass/marked1.png")
tv_utils.save_image(to_th(swap_c(marked2))[None,:],"./output/test_bass/marked2.png")
tv_utils.save_image(to_th(swap_c(marked3))[None,:],"./output/test_bass/marked3.png")
tv_utils.save_image(vid[0,:2],"./output/test_bass/vid.png")
print("vid[0,:2].shape,warped_img.shape: ",vid[0,:2].shape,warped_img.shape)
delta = vid[0,:2]-warped_img[None,:,:3]
delta = th.abs(delta)/delta.abs().max()
print("delta.shape: ",delta.shape)
tv_utils.save_image(delta[0],"./output/test_bass/delta.png")

print(spix.shape,_spix.shape)
print(means.shape,_means.shape)
print(th.sum(spix!=_spix))
assert th.mean(1.*(spix-_spix)**2).item() < 1e-5,"Equal."
print(means[0][0])
print(_means[0])
assert th.mean((means[0]-_means)**2).item() < 1e-5,"Equal."
assert th.mean((cov-_cov)**2).item() < 1e-5,"Equal."
# assert th.mean((sprobs-_sprobs)**2).item() < 1e-5,"Equal."
# assert th.mean((ids-_ids)**2).item() < 1e-5,"Equal."
exit()

K = 10

# -- remap --
timer.sync_start("remap")
num, letter = pd.factorize(spix.cpu().numpy().ravel())
num = th.from_numpy(num).to(device).reshape((B,H,W))
spix = num
timer.sync_stop("remap")

# -- hard sims --
timer.sync_start("sims_hard")
sims_hard = th_f.one_hot(spix.long())*1.
sims_hard = rearrange(sims_hard,'b h w nsp -> b nsp (h w)')
timer.sync_stop("sims_hard")

# -- pool via superpixel assignment --
img = img.flip(-1)
print(len(spix.unique()))
print("sims_hard.shape: ",sims_hard.shape)
print(img.max(),img.min())
timer.sync_start("img_pool")
pooled = sp_pool(img/255.,sims_hard)
timer.sync_stop("img_pool")
print(pooled.max(),pooled.min())

# -- grid --
timer.sync_start("grid_pool")
grid = th.cartesian_prod(th.arange(H)/1., th.arange(W)/1.).flip(-1)
grid = grid.reshape(1,H,W,-1).repeat(B,1,1,1).to(device)
grid_means = compute_means(grid,sims_hard)
timer.sync_stop("grid_pool")
print(cov[:10])
print(grid_means.shape)

print(timer)
# exit()

# -- pool xy grid --
# grid = th.cartesian_prod(th.arange(H)/(H-1.), th.arange(W)/(W-1.)).flip(-1)
grid = th.cartesian_prod(th.arange(H)/1., th.arange(W)/1.).flip(-1)
grid = grid.reshape(1,H,W,-1).repeat(B,1,1,1).to(device)
grid_pooled = sp_pool(grid,sims_hard)
# exit()

# -- compute cov --
grid_cov = compute_cov2d(grid,grid_pooled,sims_hard,prior_count)
print("cov: ",cov[:10,0])
print("grid.")
print(grid_cov[0,0,0])
print(grid_cov[0,:3,:3,0])
print("ratio.")
print(grid_cov[0,0,0][None,:]/cov[:10,:])
# print(grid_cov.shape)
exit()


# -- pool via marked boundaries --
marked = mark_boundaries(img[0].cpu().numpy(),spix[0].cpu().numpy())
marked = th.from_numpy(marked)

# -- save --
tv_utils.save_image(swap_c(marked)[None,],"./test_api/marked.png")
tv_utils.save_image(swap_c(pooled[0])[None,],"./test_api/sims_hard.png")
tv_utils.save_image(spix[None,]*1./spix.max(),"./test_api/spix.png")

# -- stat 1 [how ''flat'' is the distribution?]  --
prior_lam = 1./K
_sprobs = th.softmax(prior_lam*sprobs,-1)
print("entropy: ",th.mean(_sprobs * th.log(_sprobs+1e-20)))
freq,intervals = th.histogram(_sprobs.ravel().cpu())
sprobs_r = _sprobs.ravel().cpu()
nz = th.where(sprobs_r>.1)
sns.histplot(sprobs_r[nz])
plt.savefig("./test_api/hist.png")
plt.close("all")
# print(freq)
# print(intervals)
# print(sprobs[64,64])

# -- stat 2 [how sharp is each "K" distribution?]  --
K = sprobs.shape[-1]
prior_lam = 5e-3
_sprobs = th.softmax(prior_lam*sprobs,-1).reshape(-1,K)
_sprobs = -th.sort(-_sprobs,-1).values
_means = _sprobs.mean(0).cpu().numpy()
_stds = _sprobs.std(0).cpu().numpy()
print(_means)
print(_stds)
x = np.arange(len(_means))
plt.errorbar(x, _means, yerr=_stds, fmt='o')
plt.savefig("./test_api/sharp.png")
plt.close("all")

num = 50
plt.errorbar(x[:num], _means[:num],
             yerr=_stds[:num], fmt='o')
plt.savefig("./test_api/sharp_50.png")
plt.close("all")

# -- stat 3 [limit before softmax]  --
topK = 9
K = sprobs.shape[-1]
print("K: ",K,1./K)
# prior_lam = 5e-3
prior_lam = 1./K
print(sprobs.shape)
_sprobs = th.softmax(prior_lam*sprobs[...,:topK],-1).reshape(-1,topK)
_sprobs = -th.sort(-_sprobs,-1).values
_means = _sprobs.mean(0).cpu().numpy()
_stds = _sprobs.std(0).cpu().numpy()
print(_means)
print(_stds)
x = np.arange(len(_means))
plt.errorbar(x, _means, yerr=_stds, fmt='o')
plt.savefig("./test_api/topk_sharp.png")
plt.close("all")

freq,intervals = th.histogram(_sprobs.ravel().cpu())
sprobs_r = _sprobs.ravel().cpu()
nz = th.where(sprobs_r>.1)
sns.histplot(sprobs_r[nz])
plt.savefig("./test_api/topk_hist.png")
plt.close("all")
