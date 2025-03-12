

import os,glob
import numpy as np
import pandas as pd
from pathlib import Path
from easydict import EasyDict as edict

import torch as th
import h5py
from PIL import Image
from einops import rearrange

from st_spix.sp_pooling import sp_pooling
from st_spix.sp_video_pooling import video_pooling
from skvideo import measure



def cvlbmap(ndarray):
    codes,_ = pd.factorize(ndarray.ravel())
    return codes.reshape(ndarray.shape)


def read_spix(root,vname,offset):
    nframes = len(glob.glob(str(root/"*_params.csv")))
    spix = []
    for frame_index in range(offset,nframes+offset):
        fn = root / ("%05d.csv" % frame_index)
        spix_t = pd.read_csv(str(fn),header=None).to_numpy()
        spix.append(spix_t)
    spix = np.stack(spix)
    return spix

def read_anno_video(root,vname,offset):
    nframes = len(glob.glob(str(root/"border*png")))
    vid = []
    for frame_index in range(offset,nframes+offset):
        fn = root / ("border_%05d.png" % frame_index)
        img = np.array(Image.open(fn).convert("RGB"))/255.
        vid.append(img)
    vid = np.stack(vid)
    return vid

def segSizeByFrame(svmap,seg_ids):
    # max_id = np.max(seg_ids)+1
    max_id = np.max(seg_ids)
    frameNum = svmap.shape[2]
    hists = np.zeros((len(seg_ids)-1,frameNum))
    # hists = np.zeros((len(seg_ids),frameNum))
    # print(hists.shape,max_id,len(seg_ids))
    # # exit()
    for i in range(frameNum):
        tmp = np.histogram(svmap[:,:,i],seg_ids)
        hists[:,i] = np.histogram(svmap[:,:,i],seg_ids)[0]
        # hists[:,i] = np.bincount(svmap[:,:,i],minlength=seg_ids.max())[0]
    return hists
# function segSize = segSizeByFrame(segMap, segList)
# % This function is used to extract size information of supervoxels.
# % segMap: 3D video volume with supervoxel index.
# % segList: a list of supervoxel index.
# % segSize: a matrix containing size information per frame of supervoxels.

# frameNum = size(segMap,3);
# segSize = zeros(length(segList),frameNum);
# for i=1:frameNum
#     thisMap = segMap(:,:,i);
#     segSize(:,i) = histc(thisMap(:), segList);
# end

def count_spix(spix):

    # -- setup --
    if not th.is_tensor(spix):
        spix = th.from_numpy(spix)
    spix = spix.long()
    device = spix.device
    nframes,H,W = spix.shape
    spix = spix.reshape(nframes,-1)
    nspix = spix.max().item()+1

    # -- allocate --
    counts = th.zeros((nframes, nspix), dtype=th.int32, device=device)
    ones = th.ones_like(spix, dtype=th.int32, device=device)
    if spix.min() < 0:
        print("invalid spix!")
        print(th.where(spix == spix.min()))
        exit()
    counts.scatter_add_(1, spix, ones)

    return counts


def computeSummary(vid,seg,spix):
    # print("spix.min(),spix.max(): ",spix.min(),spix.max())

    # -- setup --
    spix = spix - spix.min() # fix offset by one
    summ = edict()
    spix_ids = np.unique(spix)
    gtpos = np.arange(seg.shape[0])
    # sizes = segSizeByFrame(spix, np.arange(0, spix.max()+2))
    # seg_sizes = segSizeByFrame(seg, np.arange(0,seg.max()+2))[1:]
    sizes = count_spix(spix)
    seg_sizes = count_spix(seg)[:,1:] # skip 0 for seg
    # gt_ids = np.unique(seg)[1:] # skip 0 for seg
    gt_ids = np.unique(seg) # keep 0 for seg

    # -- info --
    device = "cuda:0"
    vid = th.from_numpy(vid).to(device).float()
    spix = th.from_numpy(spix).to(device).int()
    seg = th.from_numpy(seg).to(device).int()
    sizes = sizes.to(device)
    seg_sizes = seg_sizes.to(device)
    # print(vid.shape,spix.shape,seg.shape,sizes.shape,seg_sizes.shape)
    # exit()

    # -- summary --
    summ.tex,summ.szv = scoreMeanDurationAndSizeVariation(sizes)
    # summ.tex,summ.szv = scoreMeanDurationAndSizeVariation_v0(spix+1)
    summ.ev = scoreExplainedVariance(vid,spix)
    summ.ev3d = scoreExplainedVariance3D(vid,spix)
    summ.pooling,pooled = scoreSpixPoolingQuality(vid,spix)

    # -- rearrange --
    # spix = rearrange(spix,'t h w -> h w t')
    # seg = rearrange(seg,'t h w -> h w t')
    # sizes = sizes.cpu().numpy().T
    # seg_sizes = seg_sizes.cpu().numpy().T

    # mode = "3d"
    # summ.ue3d,summ.sa3d = scoreUnderErrorAndSegAccuracy(spix, sizes,
    #                                                     seg, seg_sizes,
    #                                                     gt_ids, gtpos, mode)
    # mode = "2d"
    # summ.ue2d,summ.sa2d = scoreUnderErrorAndSegAccuracy(spix, sizes,
    #                                                     seg, seg_sizes,
    #                                                     gt_ids, gtpos, mode)

    # print(summ.ue2d,summ.sa2d)
    # mode = "2d_old"
    # summ.ue2d,summ.sa2d = scoreUnderErrorAndSegAccuracy(spix, sizes,
    #                                                     seg, seg_sizes,
    #                                                     gt_ids, gtpos, mode)
    # print(summ.ue2d,summ.sa2d)

    outs = scoreUEandSA(spix, sizes, seg, seg_sizes, gt_ids, gtpos)
    # print(outs)
    summ.ue2d,summ.sa2d,summ.ue3d,summ.sa3d = outs
    # exit()

    # -- summary stats --
    summ.spnum = spix.max().item()+1
    # summ.ave_nsp = averge_unique_spix(spix)
    summ.ave_nsp = averge_unique_spix_v1(sizes)
    # v0 = averge_unique_spix(rearrange(spix.cpu().numpy(),'t h w -> h w t'))
    # print(summ.ave_nsp,v0)
    # exit()
    # print(summ)


    # -- optional strred --
    summ.strred0 = -1
    summ.strred1 = -1
    strred = True
    if strred:
        _pooled = rearrange(pooled,'t c h w -> t h w c')
        _pooled = rgb_to_luminance_torch(_pooled)
        _vid = rgb_to_luminance_torch(vid)
        _,score0,score1 = measure.strred(_vid.cpu().numpy(),_pooled.cpu().numpy())
        summ.strred0 = float(score0)
        summ.strred1 = float(score1)
    # exit()

    return summ


def rgb_to_luminance_torch(rgb):
    # Rec. 709 coefficients (sRGB)
    weights = th.tensor([0.2126, 0.7152, 0.0722], dtype=rgb.dtype, device=rgb.device)
    return th.tensordot(rgb, weights, dims=([-1], [0]))[...,None]  # Sum along RGB channels

def averge_unique_spix_v1(sizes):
    return th.mean(1.*th.sum(sizes>0,-1)).item() # sum across spix; ave across time

def averge_unique_spix(spix):
    nsp = []
    for t in range(spix.shape[-1]):
        nsp.append(len(np.unique(spix[:,:,t])))
    return np.mean(nsp)

def scoreSpixPoolingQualityByFrame(vid,spix,metric="psnr"):
    # -- setup --
    device = "cuda:0"
    if not th.is_tensor(vid):
        vid = th.tensor(vid).to(device).double()
    if not th.is_tensor(spix):
        spix = th.tensor(spix.astype(np.int32)).to(device)

    # -- pooling --
    pooled,down = sp_pooling(vid,spix)
    vid = rearrange(vid,'t h w f -> t f h w')
    pooled = rearrange(pooled,'t h w f -> t f h w')
    from st_spix import metrics
    if metric == "psnr":
        res = metrics.compute_psnrs(vid,pooled,div=1.)
    elif metric == "ssim":
        res = metrics.compute_ssims(vid,pooled,div=1.)
    else:
        raise ValueError(f"Uknown metric name [{metric}]")
    return res,pooled

def scoreSpixPoolingQuality(vid,spix):
    # -- setup --
    device = "cuda:0"
    if not th.is_tensor(vid):
        vid = th.tensor(vid).to(device).double()
    if not th.is_tensor(spix):
        spix = th.tensor(spix.astype(np.int32)).to(device)

    # -- pooling --
    pooled,down = sp_pooling(vid,spix)
    vid = rearrange(vid,'t h w f -> t f h w')
    pooled = rearrange(pooled,'t h w f -> t f h w')
    from st_spix import metrics
    psnr = metrics.compute_psnrs(vid,pooled,div=1.).mean().item()
    return psnr,pooled

def scoreExplainedVariance(vid,spix):
    # roughly: var(sp_mean) / var(pix)
    # but sp_mean is computed per frame

    # -- setup --
    device = "cuda:0"
    if not th.is_tensor(vid):
        vid = th.tensor(vid).to(device).double()
    if not th.is_tensor(spix):
        spix = th.tensor(spix.astype(np.int32)).to(device)

    # -- Global mean --
    mean_global = vid.mean(dim=(1,2),keepdim=True)  # Mean across pixels (dim=0)

    # -- pixels vs mean --
    pix2mean = ((vid - mean_global) ** 2).sum((-1,-2,-3))  # Sum over color channels

    # -- sp-aves v.s. mean --
    pooled,down = sp_pooling(vid,spix)
    vid = rearrange(vid,'t h w f -> t f h w')
    pool2mean = ((pooled - mean_global)**2).sum((-1,-2,-3))
    score = (pool2mean / (pix2mean+1e-10)).mean().item()
    return score

def scoreExplainedVariance3D(vid,spix):
    # roughly: var(sp_mean) / var(pix)
    # but sp_mean is computed across all frames


    # -- setup --
    device = "cuda:0"
    if not th.is_tensor(vid):
        vid = th.tensor(vid).to(device).double()
    if not th.is_tensor(spix):
        spix = th.tensor(spix.astype(np.int32)).to(device)

    # -- get global mean per frame --
    vid = 255.*vid
    mean_global = vid.mean(dim=(1,2),keepdim=True)  # Mean across pixels

    # -- compare unnormalized variance of pixels --
    pix2mean = ((vid - mean_global) ** 2).sum((-1,-2,-3))  # Sum over color channels

    # -- unnormalized variance of superpixels --
    pooled,down,cnts = video_pooling(vid[None,:],spix[None,:])
    pooled = pooled[0]
    pooled2mean = ((pooled - mean_global)**2).sum((-1,-2,-3))

    # -- divide and average across # of frames --
    score = (pooled2mean/(pix2mean+1e-10)).mean().item()

    return score


def scoreMeanDurationAndSizeVariation(counts):

    # -- temporal extent; how long is each pixel alive? --
    TEX = th.mean(1.*(counts>0),1).mean().item()

    # -- how much does each superpixel change shape? --
    T,S = counts.shape
    counts = counts.to("cuda")
    mask = counts > 0
    num_valid = th.sum(mask, dim=0)
    num_valid_s = th.clamp(num_valid, min=1)

    # -- compute unbiased variance --
    sum_counts = th.sum(counts * mask, dim=0)  # Sum along T (ignoring zeros)
    sum_counts_sq = th.sum((counts**2) * mask, dim=0)  # Sum of squares along T
    variance = (sum_counts_sq / num_valid_s) - (sum_counts / num_valid_s) ** 2
    correction = num_valid_s/th.clamp(num_valid_s-1,min=1)
    variance = correction * variance
    stds = th.sqrt(variance)

    # -- compute average variance of only valid points --
    args = th.where(num_valid>1)
    stds = stds[args]
    SZV = th.mean(stds)
    # if th.isnan(SZV):
    #     SZV = th.tensor([0.])
    #     # print(num_valid[11])
    #     # print(sum_counts[11])
    #     # print(sum_counts_sq[11])
    #     # N = num_valid[11]
    #     # print("."*10)
    #     # print(sum_counts_sq[11]/N)
    #     # print((sum_counts[11]/N)**2)
    #     # print(variance[11])
    #     # # print(sum_counts)
    #     # # print(sum_counts_sq)
    #     # # print(variance)
    #     # print("SZV is nan!")
    #     # exit()
    SZV = SZV.item()
    # num_valid = num_valid[args]
    # SZV = ((num_valid * stds).sum()/th.sum(num_valid)).item() # weighted by size
    return TEX, SZV


def scoreMeanDurationAndSizeVariation_v0(svMap):

    # -- setup --
    svMap = svMap.astype(np.int32)
    sv_labels = np.unique(svMap)  # Avoid zero as a label
    frame_num = svMap.shape[2]
    sv_labels = th.tensor(sv_labels).to("cuda")
    svMap = th.tensor(svMap.astype(np.int64)).to("cuda")
    # print(svMap.shape)
    # exit()

    # -- get counts --
    from dev_basics.net_chunks.shared import get_chunks
    size = 8
    overlap = 0.
    H,W,T = svMap.shape
    S = len(sv_labels)
    counts = th.zeros((S,T)).to("cuda")
    num_h = (H-1)//size+1
    num_w = (W-1)//size+1
    for hi in range(num_h):
        for wi in range(num_w):
            h_chunk = hi*size
            w_chunk = wi*size
            svChunk = svMap[h_chunk:h_chunk+size,w_chunk:w_chunk+size]
            counts += (svChunk[...,None] == sv_labels).sum((0,1)).T

    # Temporal Extend
    # TEX = (th.sum(counts>0)/counts.numel()).item() # the same
    TEX = th.mean(1.*(counts>0),0).mean().item()
    exit()

    # Size Variation
    stds = th.zeros(S).to("cuda")
    for s in range(S):
        count_s = counts[s][counts[s]>0]
        valid = len(count_s)>0
        stds[s] = th.std(count_s) if valid else 0.
        # stds[s] = _std if not th.isnan(_std) else 0.
    SZV = th.mean(stds).item()

    return TEX, SZV

def scoreUEandSA(spix, counts, gtSeg, gtSize, gtList, gtPos):
    """
    This function is used to score 3D Undersegmentation Error and 3D
    Segmentation Accuracy for supervoxels.
    """

    # In case ground-truth is sparsely annotated
    spix = spix[gtPos, :, :] # T H W
    if spix.shape != gtSeg.shape:
        print('Error: gtSeg and spix dimension mismatch!')
        return -1., -1., -1., -1.

    # -- init --
    device = spix.device
    T,H,W = spix.shape
    # frameNum = gtSeg.shape[0]
    T,S = counts.shape
    numGT = len(gtList)
    gtUE = th.zeros((T,numGT),device=device)
    gtUE3D = th.zeros((numGT),device=device)
    gtSA = th.zeros((T,numGT),device=device)
    gtSA3D = th.zeros((numGT),device=device)

    # gtUE = th.zeros((len(gtList), frameNum),device=device)
    # gtSA = th.zeros((len(gtList), frameNum),device=device)

    # -- setup for alt --
    counts = counts.long()
    spix = spix.long()
    gtSeg = gtSeg.long()
    gtSize = gtSize.long()
    invalid = spix.max().item()+1

    # th.from_numpy(counts).T.long()
    # spix = th.from_numpy(spix).clone().long()
    # spix = rearrange(spix,'h w t -> t h w')
    # gtSeg = th.from_numpy(gtSeg).clone().long()
    # gtSize = th.from_numpy(gtSize).clone().long()
    # gtSeg = rearrange(gtSeg,'h w t -> t h w')
    # T,S = counts.shape
    # invalid = spix.max()+1

    for i in range(len(gtList)): # number of masks

        # -- get counts overlapping with mask --
        gt_id = gtList[i]
        invalid_mask = gtSeg != int(gt_id)
        spix_i = spix.clone()
        spix_i[th.where(invalid_mask)] = invalid
        in_counts = count_spix(spix_i)[:,:S].long() # remove invalid
        # assert th.all(in_counts <= counts).item() # true
        # print(in_counts.shape,counts.shape)


        # -- get spix ids and counts which overlap with the gtSeg --
        # mask = (gtSeg == gtList[i])
        # svOnMask = spix[mask]
        # svOnID, svOnSize = np.unique(svOnMask, return_counts=True)

        # -- compute ue --
        # gtUE_i = th.zeros((T,S),device=device,dtype=th.long)
        # args = th.where(in_counts>0)
        # gtUE_i[args] = counts[args]
        # gtUE[:,i] = gtUE_i.sum(-1)

        # -- corrected ue --
        out_counts = th.zeros((T,S),device=device,dtype=th.long)
        min_counts = th.zeros((T,S),device=device,dtype=th.long)
        args = th.where(in_counts>0)
        out_counts[args] = counts[args] - in_counts[args]
        min_counts[args] = th.minimum(out_counts[args],in_counts[args])
        gtUE[:,i] = min_counts.sum(-1)
        # print(counts[args])
        # print(in_counts[args])
        # print(out_counts[args])
        # print(min_counts[args])
        # exit()

        # -- compute ue 3D --
        min_counts_s = th.zeros((S),device=device,dtype=th.long)
        out_counts_s = out_counts.sum(0)
        in_counts_s = in_counts.sum(0)
        args = th.where(in_counts_s>0)
        min_counts_s[args] = th.minimum(out_counts_s[args],in_counts_s[args])
        gtUE3D[i] = min_counts_s.sum()

        # -- compute sa 2d --
        gtSA_i = th.zeros((T,S),device=device,dtype=th.long)
        args = th.where(in_counts >= (0.5 * counts))
        gtSA_i[args] = in_counts[args]
        gtSA[:,i] = gtSA_i.sum(-1)

        # -- compute sa 3d --
        in_counts = in_counts.sum(0)
        args = th.where(in_counts >= (0.5 * counts.sum(0)))
        gtSA3D[i] = in_counts[args].sum()

        # -- ue and sa --
        # gtUE[i] = counts[svOnID].sum()  # Corrected for zero-based indexing
        # gtSA[i] = svOnSize[svOnSize >= 0.5 * counts[svOnID]].sum()

    # UE = how many extra pixels for all spix in the GT.
    # SA = how many pixels are more than half in the GT
    # UE_2d = th.mean((gtUE - gtSize) / gtSize)
    # SA_2d = th.mean(gtSA / gtSize)

    # -- remove gtlabel "0" for SA --
    # print(gtSA.shape,gtSA3D.shape)
    gtSA = gtSA[:,1:]
    gtSA3D = gtSA3D[1:]
    # print(gtSA.shape,gtSA3D.shape)
    # print(gtSize.shape)

    # -- 2d metrics ["masked" average dropping the "0" frames --
    # print(gtUE)
    # print(gtSize)
    # print(gtUE.shape,gtSize.shape)
    gtSize_mask = gtSize + (gtSize == 0)
    # print(gtSize,gtSize_mask)
    # UE_2d = th.mean(th.sum((gtUE - gtSize) / gtSize_mask, axis=0) / th.sum(gtSize > 0, axis=0))
    # UE_2d = th.mean(th.sum(gtUE / gtSize_mask, axis=0) / th.sum(gtSize > 0, axis=0))
    UE_2d = th.mean(gtUE / (H*W))
    SA_2d = th.mean(th.sum(gtSA / gtSize_mask, axis=0) / th.sum(gtSize > 0, axis=0))

    # -- 3d metrics --
    gtUE = gtUE.sum(0)
    gtSA = gtSA.sum(0)
    gtSize = gtSize.sum(0)
    # print(gtUE)
    # print(gtUE3D)
    # print(gtSA)
    # print(gtSA3D)
    # print(gtSize)
    assert th.all(gtSize>0).item()
    # UE_3d = th.mean((gtUE - gtSize) / gtSize)
    # UE_3d = th.mean(gtUE / gtSize)
    # UE_3d = th.mean(gtUE / gtSize)
    UE_3d = th.mean(gtUE3D / (T*H*W))
    SA_3d = th.mean(gtSA3D / gtSize)

    return UE_2d.item(),SA_2d.item(),UE_3d.item(),SA_3d.item()


def scoreUnderErrorAndSegAccuracy(spix, counts, gtSeg, gtSize, gtList, gtPos, mode):
    """
    This function is used to score 3D Undersegmentation Error and 3D
    Segmentation Accuracy for supervoxels.
    """

    # -- ensure shape --
    assert counts.shape[-1] == spix.shape[-1]

    # In case ground-truth is sparsely annotated
    spix = spix[:, :, gtPos] # H W T
    if spix.shape != gtSeg.shape:
        print('Error: gtSeg and spix dimension mismatch!')
        return None, None
    spix = spix.astype(np.uint32)

    # -- 3d ue and sa --
    if mode.lower() == '3d':
        # counts = counts.sum(axis=1)
        # gtSize = gtSize.sum(axis=1)
        # gtUE = np.zeros(len(gtList))
        # gtSA = np.zeros(len(gtList))

        # -- init --
        device = spix.device
        frameNum = gtSeg.shape[2]
        gtUE = th.zeros((len(gtList), frameNum),device=device)
        gtSA = th.zeros((len(gtList), frameNum),device=device)

        # -- setup for alt --
        counts = th.from_numpy(counts).T.long()
        spix = th.from_numpy(spix).clone().long()
        spix = rearrange(spix,'h w t -> t h w')
        gtSeg = th.from_numpy(gtSeg).clone().long()
        gtSize = th.from_numpy(gtSize).clone().long()
        gtSeg = rearrange(gtSeg,'h w t -> t h w')
        T,S = counts.shape
        invalid = spix.max()+1

        for i in range(len(gtList)): # number of masks

            # -- get counts overlapping with mask --
            gt_id = gtList[i]
            invalid_mask = gtSeg != int(gt_id)
            spix_i = spix.clone()
            spix_i[th.where(invalid_mask)] = invalid
            on_counts = count_spix(spix_i)[:,:-1].long() # remove invalid

            # -- get spix ids and counts which overlap with the gtSeg --
            # mask = (gtSeg == gtList[i])
            # svOnMask = spix[mask]
            # svOnID, svOnSize = np.unique(svOnMask, return_counts=True)

            # -- compute ue --
            gtUE_i = th.zeros(T,S).long()
            args = th.where(on_counts>0)
            gtUE_i[args] = counts[args]
            gtUE[i] = gtUE_i.sum()

            # -- compute sa --
            gtSA_i = th.zeros(T,S).long()
            args = th.where(on_counts >= (0.5 * counts))
            gtSA_i[args] = on_counts[args]
            gtSA[i] = gtSA_i.sum()

            # -- ue and sa --
            # gtUE[i] = counts[svOnID].sum()  # Corrected for zero-based indexing
            # gtSA[i] = svOnSize[svOnSize >= 0.5 * counts[svOnID]].sum()

        # UE = how many extra pixels for all spix in the GT.
        # SA = how many pixels are more than half in the GT
        gtSize = gtSize.sum(-1)
        assert th.all(gtSize>0).item()
        # gtSize_mask = gtSize + (gtSize == 0)
        UE = th.mean((gtUE - gtSize) / gtSize)
        SA = th.mean(gtSA / gtSize)

    elif mode.lower() == '2d':

        # -- init --
        frameNum = gtSeg.shape[2]
        gtUE = np.zeros((len(gtList), frameNum))
        gtSA = np.zeros((len(gtList), frameNum))
        ue = 0.

        # -- setup for alt --
        counts = th.from_numpy(counts).T.long()
        spix = th.from_numpy(spix).clone().long()
        spix = rearrange(spix,'h w t -> t h w')
        gtSeg = th.from_numpy(gtSeg).clone().long()
        gtSeg = rearrange(gtSeg,'h w t -> t h w')
        T,S = counts.shape
        invalid = spix.max()+1


        for i,gt_id in enumerate(gtList):

            # -- get counts overlapping with mask --
            invalid_mask = gtSeg != int(gt_id)
            spix_i = spix.clone()
            spix_i[th.where(invalid_mask)] = invalid
            on_counts = count_spix(spix_i)[:,:-1].long() # remove invalid

            # -- compute ue --
            gtUE_i = th.zeros(T,S).long()
            args = th.where(on_counts>0)
            gtUE_i[args] = counts[args]
            gtUE[i] = gtUE_i.sum(-1)

            # -- compute sa --
            gtSA_i = th.zeros(T,S).long()
            args = th.where(on_counts >= (0.5 * counts))
            gtSA_i[args] = on_counts[args]
            gtSA[i] = gtSA_i.sum(-1)

        # -- finalize --
        gtSize_mask = gtSize + (gtSize == 0)
        UE = np.mean(np.sum((gtUE - gtSize) / gtSize_mask, axis=1) / np.sum(gtSize > 0, axis=1))
        SA = np.mean(np.sum(gtSA / gtSize_mask, axis=1) / np.sum(gtSize > 0, axis=1))

    elif mode.lower() == '2d_old':

        frameNum = gtSeg.shape[2]
        gtUE = np.zeros((len(gtList), frameNum))
        gtSA = np.zeros((len(gtList), frameNum))
        ue = 0.

        for i in range(frameNum):
            thisGtMap = gtSeg[:, :, i]
            thisSvMap = spix[:, :, i]
            GtOn = np.unique(thisGtMap)

            for j in range(len(gtList)):
                if gtList[j] not in GtOn:
                    # print("continuing: ",i,j,GtOn)
                    continue
                mask = (thisGtMap == gtList[j])
                svOnMask = thisSvMap[np.where(mask)]
                svOnID, svOnSize = np.unique(svOnMask, return_counts=True)
                # if i < 30:
                #     print(i,svOnID[:10],counts[svOnID, i].sum())
                gtUE[j, i] = counts[svOnID, i].sum()  # Corrected indexing
                gtSA[j, i] = svOnSize[svOnSize >= 0.5 * counts[svOnID, i]].sum()

        gtSize_mask = gtSize + (gtSize == 0)
        # print("\n")
        # print(gtUE[0,:10])
        # print(gtSA[0,:10])
        UE = np.mean(np.sum((gtUE - gtSize) / gtSize_mask, axis=1) / np.sum(gtSize > 0, axis=1))
        SA = np.mean(np.sum(gtSA / gtSize_mask, axis=1) / np.sum(gtSize > 0, axis=1))

    else:
        print('Error: unknown mode.')
        return None, None, None, None

    # print(UE,SA)
    return UE.item(), SA.item()#, onMapUE, onMapSA

def undersegmentation_error_v(ground_truth, superpixels):
    # Unique labels for ground truth and superpixel segments
    sp_labels = np.unique(superpixels)
    gt_masks = ground_truth[..., None] == 1

    # Create a mask for each superpixel label
    sp_masks = 1.*(superpixels[..., None] == sp_labels)

    # Calculate the intersection between each ground truth mask and each superpixel mask
    intersections = np.einsum('ijk,ijtm->ktm', gt_masks, sp_masks)
    spix = superpixels


def undersegmentation_error(ground_truth, superpixels):
    """
    Calculate the undersegmentation error for a given superpixel segmentation.

    Args:
        ground_truth: A 2D numpy array representing the ground truth segmentation.
        superpixels: A 2D numpy array representing the superpixel segmentation.

    Returns:
        The undersegmentation error.
    """
    # Unique labels for ground truth and superpixel segments
    gtseg = ground_truth
    spix = superpixels
    gt_labels = np.unique(ground_truth)
    sp_labels = np.unique(spix)

    # nspix = len(sp_labels)
    # print(spix.shape,gtseg.shape)
    # living,counts = np.unique(spix[np.where(gtseg==1)],return_counts=True)
    # _living,_counts = np.unique(spix,return_counts=True)
    # # counts = np.histogram(spix[np.where(gtseg==1)],nspix+1)[0]
    # print(living)
    # print(counts)

    # print(_living[:10])
    # print(_counts[:10])

    # # counts = np.histogram(spix,nspix)[0]


    # exit()

    # Initialize undersegmentation error
    ue = 0.0

    # Create a mask for each ground truth label
    # gt_masks = 1.*(ground_truth[..., None] == gt_labels)
    gt_masks = ground_truth[..., None] == 1

    # Create a mask for each superpixel label
    sp_masks = 1.*(superpixels[..., None] == sp_labels)

    # Calculate the intersection between each ground truth mask and each superpixel mask
    intersections = np.einsum('ijk,ijm->km', gt_masks, sp_masks)
    spix = superpixels

    # print(intersections)
    # gt_mask = gt_masks[:,:,0]
    # for sp_label in sp_labels:
    #     spix_l = spix==sp_label
    #     sp_size = spix_l.sum()
    #     inter = np.logical_and(gt_mask,spix_l).sum()
    #     union = np.logical_or(gt_mask,spix_l).sum()
    #     sp_out = sp_spix - inter
    #     sp_in = inter
    #     minp = min(sp_out,sp_in)
    #     if inter > 0:
    #         print(sp_label,union,inter)

    # Calculate the area of each ground truth and superpixel segment
    gt_areas = np.sum(gt_masks, axis=(0, 1))
    sp_areas = np.sum(sp_masks, axis=(0, 1))
    # print(gt_areas.shape)
    # print(sp_areas.shape)
    # print(intersections.shape)
    # exit()

    # Calculate the union for each ground truth-superpixel pair
    # unions = gt_areas[:, None] + sp_areas[None, :] - intersections
    unions = gt_areas[0] + sp_areas - intersections
    # print(unions.shape)
    # print(intersections)
    # s,e = 136,142
    # print(unions[0][s:e])
    # print(intersections[0][s:e])
    # print(gt_areas)
    # print(sp_areas[136:142])

    # Calculate undersegmentation error by accumulating differences
    # print("unions.shape: ",unions.shape)
    ue = np.sum((unions - intersections)[intersections > 0])
    ue = ue / ground_truth.size
    # print(ue)
    # # exit()

    # Normalize by the total number of pixels
    return ue

def read_svmap(fname):
    svmap = np.array(h5py.File(fname)['svMap'])
    svmap = rearrange(svmap,'t w h -> h w t')
    return svmap

def read_seg(root):

    # -- sort by number --
    files = [f.name for f in root.iterdir()]
    files = sorted(files, key=lambda x: int(os.path.splitext(x)[0]))

    # -- read --
    seg = []
    for fn in files:
        img_f = np.array(Image.open(root/fn).convert("L"))
        img_f = (img_f >= 128).astype(np.float32)
        seg.append(img_f)
    seg = np.stack(seg,-1)
    return seg

def read_vid(root):

    # -- sort by number --
    files = [f.name for f in root.iterdir() if str(f).endswith(".png")]
    files = sorted(files, key=lambda x: int(os.path.splitext(x)[0]))

    # -- read --
    seg = []
    for fn in files:
        img_f = np.array(Image.open(root/fn).convert("RGB"))/255.
        seg.append(img_f)
    seg = np.stack(seg)
    return seg

def get_video_names(dname):
    if "segtrack" in dname.lower():
        return get_segtrackerv2_videos()
    elif "davis" in dname.lower():
        return get_davis_videos()
    else:
        raise KeyError(f"Uknown dataset name [{dname}]")

def get_segtrackerv2_videos():
    root = Path("/home/gauenk/Documents/packages/superpixel-benchmark/docker/in/SegTrackv2/GroundTruth/")
    vid_names = list([v.name for v in root.iterdir()])
    # vid_names = ["frog_2","girl"]
    return vid_names

def get_davis_videos():
    try:
        # names = np.loadtxt("/app/in/DAVIS/ImageSets/2017/train-val.txt",dtype=str)
        names = np.loadtxt("/app/in/DAVIS/ImageSets/2017/val.txt",dtype=str)
    except:
        # fn = "/home/gauenk/Documents/data/davis/DAVIS/ImageSets/2017/train-val.txt"
        fn = "/home/gauenk/Documents/data/davis/DAVIS/ImageSets/2017/val.txt"
        names = np.loadtxt(fn,dtype=str)
    # names = names[:10]
    # names = ["bmx-trees","breakdance"]
    # names = ["bmx-trees"]
    # names = names[:4]
    # names = ["bike-packing","blackswan","bmx-trees",
    #          "breakdance","camel","car-roundabout"]
    # names = ["bmx-trees","car-roundabout"]
    # names = ["bike-packing","bmx-trees"]
    # names = ["car-roundabout"]
    # names = ["bmx-trees"]
    # exit()
    return names

# def get_segtrackerv2_videos():
#     root = Path("/home/gauenk/Documents/packages/superpixel-benchmark/docker/in/SegTrackv2/GroundTruth/")
#     vid_names = list([v.name for v in root.iterdir()])
#     # vid_names = ["frog_2","girl"]
#     return vid_names


