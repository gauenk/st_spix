
# -- basic --
import gc
import glob,os

from scipy.io import loadmat
from skimage.segmentation import mark_boundaries

import torch
import torch as th
import numpy as np
from torchvision.utils import save_image
import torch.nn.functional as thF


from pathlib import Path
from einops import rearrange
from easydict import EasyDict as edict

# -- load models --
from st_spix.models import load_model as _load_model

# -- project imports --
import st_spix
from st_spix import metrics
from st_spix.data import load_data
from st_spix.losses import load_loss
from st_spix.models import load_model
import st_spix.trte_utils as utils
import st_spix.utils as base_utils

# SAVE_ROOT = Path("output/eval_superpixels/")

def run(cfg):

    # -- init experiment --
    defs = {"data_path":"./data/","data_augment":False,
            "patch_size":128,"data_repeat":1,"colors":3,
            "use_connected":False,"save_output":False,"seed":0,
            "num_samples":0,"load_checkpoint":True,
            "flow_method":"raft","window_time":1}
    cfg = base_utils.extract_defaults(cfg,defs)
    device = "cuda"
    save_root = Path(cfg.save_root) / cfg.tr_uuid

    # -- seed --
    base_utils.seed_everything(cfg.seed)

    # -- dataset --
    cfg.data_load_test = True
    dset,dataloader = load_data(cfg)

    # -- noise function --
    sigma = base_utils.optional(cfg,'sigma',0.)
    ntype = base_utils.optional(cfg,'noise_type',"gaussian")
    def pre_process(x):
        if ntype == "gaussian":
            return x + (sigma/255.)*th.randn_like(x),{}
        elif ntype == "isp":
            return isp.run_unprocess(x)
        else:
            raise ValueError(f"Uknown noise type [{ntype}]")
    def post_process(deno,noise_info):
        if ntype == "gaussian":
            return deno
        elif ntype == "isp":
            keys = ['red_gain','blue_gain','cam2rgb']
            args = [noise_info[k] for k in keys]
            return isp.run_process(deno,*args)
        else:
            raise ValueError(f"Uknown noise type [{ntype}]")

    # -- load model --
    model = load_model(cfg)

    # -- restore from checkpoint --
    chkpt_root = utils.get_ckpt_root(cfg.tr_uuid,cfg.base_path)
    chkpt,chkpt_fn = utils.get_checkpoint(chkpt_root)

    # -- checking --
    no_chkpt = not(chkpt is None)
    is_empty = "empty" in cfg.mname
    # print(no_chkpt,is_empty,cfg.mname)
    assert no_chkpt or is_empty,"Must have a checkpoint loaded for testing."

    # -- load --
    if cfg.load_checkpoint:
        loaded_epoch = utils.load_checkpoint(chkpt,model,skip_module=False)
        print("Restoring State from [%s]" % chkpt_fn)

    # -- to device --
    model = model.to(device)
    model = model.eval()

    # -- load flow function --
    flow_fxn = utils.load_flow_fxn(cfg,device)

    # -- init info --
    ifields = ["asa","br","bp","nsp_og","nsp","hw","name",
               "pooled_psnr","pooled_ssim","entropy",
               "deno_psnr","deno_ssim"]
    info = edict()
    for f in ifields: info[f] = []

    # -- each sample --
    data_iter = iter(dataloader)
    for ix in range(len(dataloader)):

        # -- garbage collect --
        gc.collect()
        th.cuda.empty_cache()

        # -- unpack --
        batch = next(data_iter)
        img,seg = batch['clean'][0],batch['seg'][0]
        img,seg = img.to(device)/255.,seg.to(device)
        # print(img.shape,seg.shape)

        # -- ... --
        # img = img[:4]
        # seg = seg[:4]

        # -- accomodate "small" GPU memory --
        # -- reduce the size of the flow so we can process the 64x64 regions --
        # img = thF.interpolate(img, scale_factor=0.5, mode='bilinear', align_corners=False)
        # seg = thF.interpolate(seg, scale_factor=0.5, mode='bilinear', align_corners=False)
        # print(img.shape,seg.shape)
        # exit()

        # img, seg = img[:,:,:-1,:-1], seg[:,:-1,:-1]
        name = dset.vid_names[ix]
        B,F,H,W = img.shape
        # print(img.shape)
        # assert B == 1,"Testing metrics are not currently batch-able."
        # print(img.shape)
        # exit()

        # -- optional noise --
        noisy,ninfo = pre_process(img)

        # -- compute flows --
        flows,fflow = flow_fxn(noisy)

        """

        TODO: we need to estimate spix before forward pass of NN on the full image.

        """

        # -- get superpixels --
        with th.no_grad():
            spix = model.get_spix(noisy,flows,fflow)[-1]

        # -- compute superpixels --
        with th.no_grad():
            output = crop_test(model,noisy,flows,fflow,spix,ninfo)
            # output =  {"deno":model(noisy,flows,fflow,ninfo,sims),"sims":None}
        # with th.no_grad():
        #     output = model(noisy,flows,fflow,ninfo)['deno']
            # output = []
            # for t in range(len(noisy)):
            #     output.append(model(noisy[[t]],flows[:,[t]],fflow[[t]],ninfo)['deno'])
            # output = th.cat(output)
        # print("output.shape: ",output.shape)
        # exit()
        # -- unpack --
        deno = base_utils.optional(output,'deno',None)
        sims = base_utils.optional(output,'sims',None)

        # -- optionally post-process --
        deno = post_process(deno,ninfo)
        if not(sims is None) and (sims.ndim == 5):
            sims = rearrange(sims,'b h w sh sw -> b (sh sw) (h w)')

        # -- get superpixel --
        if not(sims is None):
            spix = sims.argmax(1).reshape(B,H,W)
            spix_og = spix.clone()
            if cfg.use_connected:
                if th.is_tensor(spix): spix = spix.cpu().numpy().astype(np.int64)
                else: spix = spix.astype(np.int64)
                cmin = cfg.connected_min
                cmax = cfg.connected_max
                spix = connected_sp(spix,cmin,cmax)

            # -- spix pooled --
            # _img = img[0].cpu().numpy().transpose(1,2,0)
            # print("img.shape: ",img.shape)
            entropy = th.mean(-sims*th.log(sims+1e-15)).item()
            pooled = st_spix.spix_utils.sp_pool(img,sims)

        else:
            spix = th.zeros_like(img[:,0]).long()
            spix_og = spix.clone()
            entropy = -1.
            pooled = th.zeros_like(img)

        # -- save --
        if cfg.save_output:
            # save_dir = save_root / ("s_%d"%cfg.S) / cfg.method
            save_dir = save_root / "spix"
            if not save_dir.exists():
                save_dir.mkdir(parents=True)
            _img = img[0].cpu().numpy().transpose(1,2,0)
            viz = mark_boundaries(_img,spix,mode="subpixel")
            viz = th.from_numpy(viz.transpose(2,0,1))
            save_fn = save_dir / ("%s.jpg"%name)
            save_image(viz,save_fn)


        # -- save --
        if cfg.save_output:
            # save_dir = save_root / ("s_%d"%cfg.S) / cfg.method
            save_dir = save_root / "pooled"
            if not save_dir.exists():
                save_dir.mkdir(parents=True)
            save_fn = save_root / ("%s_s.jpg"%name)
            print("Saved @ [%s]"%save_fn)
            save_image(smoothed,save_fn)

        # -- eval deno --
        deno_psnr = metrics.compute_psnrs(img,deno,div=1.)
        deno_ssim = metrics.compute_ssims(img,deno,div=1.)

        # -- eval pool --
        pooled_psnr = metrics.compute_psnrs(img,pooled,div=1.)
        pooled_ssim = metrics.compute_ssims(img,pooled,div=1.)

        # -- expand to too big --
        out = expand_ndarrays(130,deno_psnr,deno_ssim,pooled_psnr,pooled_ssim)
        deno_psnr,deno_ssim,pooled_psnr,pooled_ssim = out

        # -- eval & collect info --
        iinfo = edict()
        for f in ifields: iinfo[f] = []
        iinfo.asa = -1#metrics.compute_asa(spix[0],seg[0])
        iinfo.br = -1#metrics.compute_br(spix[0],seg[0],r=1)
        iinfo.bp = -1#metrics.compute_bp(spix[0],seg[0],r=1)
        iinfo.nsp = int(len(th.unique(spix)))
        iinfo.nsp_og = int(len(th.unique(spix_og)))
        iinfo.deno_psnr = deno_psnr
        iinfo.deno_ssim = deno_ssim
        iinfo.pooled_psnr = pooled_psnr
        iinfo.pooled_ssim = pooled_ssim
        iinfo.entropy = entropy
        iinfo.hw = img.shape[-2]*img.shape[-1]
        iinfo.name = name
        print(iinfo)
        for f in ifields: info[f].append(iinfo[f])
        if cfg.num_samples > 0 and ix >= cfg.num_samples:
            break
        exit()

    return info

def expand_ndarrays(size,*ndarrays):
    out = []
    for ndarray in ndarrays:
        ndarray_e = -np.ones(size)
        ndarray_e[:len(ndarray)] = ndarray
        out.append(ndarray_e)
    return out

def crop_test(model,img,flows,fflow,spix,ninfo,cropsize=64,overlap=0.10):
    fwd_fxn = lambda a,b,c,d,e: model(a,b,c,d,e)['deno']
    # deno = run_grouped_spatial_chunks(fwd_fxn,img,flows,fflow,ninfo,cropsize,overlap)
    # return {"deno":deno,"spix":None}


    T = img.shape[0]
    if T > 30:
        imgH = img[:T//2]
        flowsH = flows[:T//2]
        fflowH = fflow[:T//2]
        spixH = spix[:T//2]
        # print(imgH.shape)
        deno0 = run_grouped_spatial_chunks(fwd_fxn,imgH,flowsH,fflowH,
                                           spixH,ninfo,cropsize,overlap)
        # print(deno0.shape)

        imgH = img[T//2:]
        flowsH = flows[T//2:]
        fflowH = fflow[T//2:]
        spixH = spix[T//2:]
        # print(imgH.shape)
        deno1 = run_grouped_spatial_chunks(fwd_fxn,imgH,flowsH,fflowH,
                                           spixH,ninfo,cropsize,overlap)
        # print(deno1.shape)
        deno = th.cat([deno0,deno1])

    else:
        deno = run_grouped_spatial_chunks(fwd_fxn,img,flows,fflow,spix,
                                          ninfo,cropsize,overlap)

    return {"deno":deno,"sims":None}

# -- simpler one --
def run_grouped_spatial_chunks(fwd_fxn,img,flows,fflow,spix,ninfo,size,overlap):

    # -- imports --
    from dev_basics.net_chunks.shared import get_chunks

    # -- unpack --
    shape = img.shape
    B,F,H,W = img.shape

    # -- alloc --
    deno = th.zeros(shape,device=img.device)
    count = th.zeros((1,1,H,W),device=img.device)

    # -- get chunks --
    h_chunks = get_chunks(H,size,overlap)
    w_chunks = get_chunks(W,size,overlap)

    # -- loop --
    for h_chunk in h_chunks:
        for w_chunk in w_chunks:

            # -- forward --
            img_chunk = img[...,h_chunk:h_chunk+size,w_chunk:w_chunk+size]
            flows_chunk = flows[...,h_chunk:h_chunk+size,w_chunk:w_chunk+size]
            fflow_chunk = fflow[...,h_chunk:h_chunk+size,w_chunk:w_chunk+size]
            spix_chunk =  spix[...,h_chunk:h_chunk+size,w_chunk:w_chunk+size]
            deno_chunk = fwd_fxn(img_chunk,flows_chunk,fflow_chunk,spix_chunk,ninfo)

            # -- fill --
            sizeH,sizeW = deno_chunk.shape[-2:]
            deno[...,h_chunk:h_chunk+sizeH,w_chunk:w_chunk+sizeW] += deno_chunk
            count[...,h_chunk:h_chunk+sizeH,w_chunk:w_chunk+sizeW] += 1
    # -- normalize --
    deno = deno / count
    return deno

