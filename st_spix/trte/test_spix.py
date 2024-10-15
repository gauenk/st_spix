"""

        Test Superpixels


"""

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
# from st_spix.models import load_model as _load_model
from st_spix.models import SuperpixelNetwork

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
    defs = {"patch_size":0,"data_repeat":1,"colors":3,
            "use_connected":False,"save_output":False,"seed":0,
            "num_samples":0,"load_checkpoint":True,
            "save_root":"output/test_spix/",
            "group_uuid":"dev","dname":"davis"}
    cfg = base_utils.extract_defaults(cfg,defs)
    device = "cuda"
    save_root = Path(cfg.save_root) / cfg.group_uuid

    # -- seed --
    base_utils.seed_everything(cfg.seed)

    # -- dataset --
    cfg.data_load_test = False # want val data, not test
    cfg.tr_set = "val"
    dset,dataloader = load_data(cfg)

    # -- load model --
    spcfg = base_utils.extract_defaults(cfg,SuperpixelNetwork.defs)
    model = SuperpixelNetwork(3,**spcfg)
    parameters = list(model.parameters())

    # -- saved model IO --
    if len(parameters) > 0:
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
            loaded_epoch = utils.load_checkpoint(chkpt,model)
            print("Restoring State from [%s]" % chkpt_fn)

    # -- to device --
    model = model.to(device)
    model = model.eval()

    # -- init info --
    ifields = ["asa","br","bp","nsp_og","nsp","hw",
               "pooled_psnr","pooled_ssim","entropy"]
    info = edict()
    for f in ifields: info[f] = []

    # -- each sample --
    # name = "davis"
    for ix,batch in enumerate(dataloader):

        # -- garbage collect --
        gc.collect()
        th.cuda.empty_cache()

        # -- unpack --
        name = "index_%d" % batch['index']
        img,seg = batch['clean'][0],batch['seg'][0]
        img,seg = img.to(device)/255.,seg[:,0].to(device)
        B,F,H,W = img.shape
        # assert B == 1,"Testing metrics are not currently batch-able."
        # print(img.shape,seg.shape)
        # exit()

        # -- compute superpixels --
        with th.no_grad():
            output = model(img)

        # -- unpack --
        sims, spix = output[:2]
        if not(sims is None) and (sims.ndim == 5):
            sims = rearrange(sims,'b h w sh sw -> b (sh sw) (h w)')

        # -- get superpixel --
        spix_og = spix.clone()
        if not(sims is None):
            # spix = sims.argmax(1).reshape(B,H,W)
            # spix_og = spix.clone()
            if cfg.use_connected:
                if th.is_tensor(spix): spix = spix.cpu().numpy().astype(np.int64)
                else: spix = spix.astype(np.int64)
                cmin = cfg.connected_min
                cmax = cfg.connected_max
                spix = connected_sp(spix,cmin,cmax)

            # -- spix pooled --
            entropy = th.mean(-sims*th.log(sims+1e-15)).item()
            # pooled = st_spix.spix_utils.sp_pool(img,sims)
            pooled,down = st_spix.video_pooling(img[None,],spix[None,])
            pooled = pooled[0]
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

        # -- eval pool --
        pooled_psnr = metrics.compute_psnrs(img,pooled,div=1.)
        pooled_ssim = metrics.compute_ssims(img,pooled,div=1.)
        def a2v(fxn,spix,seg,**kwargs): # apply to video
            return [fxn(spix[t],seg[t],**kwargs) for t in range(spix.shape[0])]
        spix = spix[:,0].contiguous()

        # -- eval & collect info --
        iinfo = edict()
        for f in ifields: iinfo[f] = []
        iinfo.asa = a2v(metrics.compute_asa,spix,seg)
        iinfo.br = a2v(metrics.compute_br,spix,seg,r=1)
        iinfo.bp = a2v(metrics.compute_bp,spix,seg,r=1)
        iinfo.nsp = int(len(th.unique(spix)))
        iinfo.nsp_og = int(len(th.unique(spix_og)))
        iinfo.pooled_psnr = pooled_psnr
        iinfo.pooled_ssim = pooled_ssim
        iinfo.entropy = entropy
        iinfo.hw = img.shape[-2]*img.shape[-1]
        print(iinfo)
        for f in ifields: info[f].append(iinfo[f])
        if cfg.num_samples > 0 and ix >= cfg.num_samples:
            break

    return info
