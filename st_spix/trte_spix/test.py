
# -- basic --
import gc
import glob,os

from scipy.io import loadmat
from skimage.segmentation import mark_boundaries

import torch
import torch as th
import numpy as np
from torchvision.utils import save_image

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
            "num_samples":0,"load_checkpoint":True}
    cfg = base_utils.extract_defaults(cfg,defs)
    device = "cuda"
    save_root = Path(cfg.save_root) / cfg.tr_uuid

    # -- seed --
    base_utils.seed_everything(cfg.seed)

    # -- dataset --
    cfg.data_load_test = True
    dset,dataloader = load_data(cfg)

    # -- load model --
    model = load_model(cfg)

    # -- restore from checkpoint --
    chkpt_root = utils.get_ckpt_root(cfg.tr_uuid,cfg.base_path)
    chkpt,chkpt_fn = utils.get_checkpoint(chkpt_root)
    assert not(chkpt is None),"Must have a checkpoint loaded for testing."
    if cfg.load_checkpoint:
        # a = model.ftrs.decoder1.dec1conv2.weight.clone()
        loaded_epoch = utils.load_checkpoint(chkpt,model)
        # b = model.ftrs.decoder1.dec1conv2.weight.clone()
        # print(th.mean((a-b)**2))
        # exit()
        print("Restoring State from [%s]" % chkpt_fn)

    model = model.to(device)
    model = model.eval()

    # -- init info --
    ifields = ["asa","br","bp","nsp_og","nsp","hw","name",
               "psnr","ssim","entropy"]
    info = edict()
    for f in ifields: info[f] = []

    # -- each sample --
    for ix,(img,seg) in enumerate(dataloader):

        # -- garbage collect --
        gc.collect()
        th.cuda.empty_cache()

        # -- unpack --
        img, seg = img.to(device)/255., seg.to(device)
        img, seg = img[:,:,:-1,:-1], seg[:,:-1,:-1]
        name = dset.names[ix]
        B,F,H,W = img.shape
        assert B == 1,"Testing metrics are not currently batch."
        print(img.shape,seg.shape)

        # -- compute superpixels --
        with th.no_grad():
            sims = model.get_superpixel_probs(img)

        # -- get superpixel --
        spix = sims.argmax(1).reshape(B,H,W)
        spix_og = spix.clone()
        if cfg.use_connected:
            if th.is_tensor(spix): spix = spix.cpu().numpy().astype(np.int64)
            else: spix = spix.astype(np.int64)
            cmin = cfg.connected_min
            cmax = cfg.connected_max
            spix = connected_sp(spix,cmin,cmax)

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


        # -- spix pooled --
        # _img = img[0].cpu().numpy().transpose(1,2,0)
        # print("img.shape: ",img.shape)
        entropy = th.mean(-sims*th.log(sims+1e-15)).item()
        pooled = st_spix.sp_pool(img,sims)

        # -- save --
        if cfg.save_output:
            # save_dir = save_root / ("s_%d"%cfg.S) / cfg.method
            save_dir = save_root / "pooled"
            if not save_dir.exists():
                save_dir.mkdir(parents=True)
            save_fn = save_root / ("%s_s.jpg"%name)
            print("Saved @ [%s]"%save_fn)
            save_image(smoothed,save_fn)

        # -- eval --
        # print(img.shape,pooled.shape)
        # print(metrics.compute_psnrs(img,pooled,div=1.))
        # exit()
        psnr = metrics.compute_psnrs(img,pooled,div=1.).item()
        ssim = metrics.compute_ssims(img,pooled,div=1.).item()
        # print(spix.shape,seg.shape)

        # -- eval & collect info --
        iinfo = edict()
        for f in ifields: iinfo[f] = []
        iinfo.asa = metrics.compute_asa(spix[0],seg[0])
        iinfo.br = metrics.compute_br(spix[0],seg[0],r=1)
        iinfo.bp = metrics.compute_bp(spix[0],seg[0],r=1)
        iinfo.nsp = int(len(th.unique(spix)))
        iinfo.nsp_og = int(len(th.unique(spix_og)))
        iinfo.psnr = psnr
        iinfo.ssim = ssim
        iinfo.entropy = entropy
        iinfo.hw = img.shape[-2]*img.shape[-1]
        iinfo.name = name
        print(iinfo)
        for f in ifields: info[f].append(iinfo[f])
        if cfg.num_samples > 0 and ix >= cfg.num_samples:
            break

    return info
