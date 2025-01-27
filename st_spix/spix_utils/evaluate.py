

import os
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


def cvlbmap(ndarray):
    codes,_ = pd.factorize(ndarray.ravel())
    return codes.reshape(ndarray.shape)

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
    counts = th.zeros((nframes, nspix), dtype=th.int32)
    ones = th.ones_like(spix, dtype=th.int32)
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
    gt_ids = np.unique(seg)[1:] # skip 0 for seg


    # -- summary --
    summ.tex,summ.szv = scoreMeanDurationAndSizeVariation(sizes)
    # summ.tex,summ.szv = scoreMeanDurationAndSizeVariation_v0(spix+1)
    summ.ev = scoreExplainedVariance(vid,spix)
    summ.ev3d = scoreExplainedVariance3D(vid,spix)
    summ.pooling = scoreSpixPoolingQuality(vid,spix)

    # -- rearrange --
    spix = rearrange(spix,'t h w -> h w t')
    seg = rearrange(seg,'t h w -> h w t')
    sizes = sizes.cpu().numpy().T
    seg_sizes = seg_sizes.cpu().numpy().T

    mode = "3d"
    summ.ue3d,summ.sa3d = scoreUnderErrorAndSegAccuracy(spix, sizes,
                                                        seg, seg_sizes,
                                                        gt_ids, gtpos, mode)
    mode = "2d"
    summ.ue2d,summ.sa2d = scoreUnderErrorAndSegAccuracy(spix, sizes,
                                                        seg, seg_sizes,
                                                        gt_ids, gtpos, mode)
    # print(summ.ue2d,summ.sa2d)
    # mode = "2d_old"
    # summ.ue2d,summ.sa2d = scoreUnderErrorAndSegAccuracy(spix, sizes,
    #                                                     seg, seg_sizes,
    #                                                     gt_ids, gtpos, mode)
    # print(summ.ue2d,summ.sa2d)

    # -- summary stats --
    summ.spnum = spix.max().item()+1
    summ.ave_nsp = averge_unique_spix(spix)
    # print(summ)

    # exit()

    return summ

def averge_unique_spix(spix):
    nsp = []
    for t in range(spix.shape[-1]):
        nsp.append(len(np.unique(spix[:,:,t])))
    return np.mean(nsp)

def scoreSpixPoolingQuality(vid,spix):
    device = "cuda:0"
    vid = th.tensor(vid).to(device)
    spix = th.tensor(spix.astype(np.int32)).to(device)

    pooled,down = sp_pooling(vid,spix)
    vid = rearrange(vid,'t h w f -> t f h w')
    pooled = rearrange(pooled,'t h w f -> t f h w')
    from st_spix import metrics
    psnr = metrics.compute_psnrs(vid,pooled,div=1.).mean()
    return psnr

def scoreExplainedVariance(vid,spix):
    # roughly: var(sp_mean) / var(pix)
    # but sp_mean is computed per frame

    # -- setup --
    device = "cuda:0"
    vid = th.tensor(vid).to(device).double()
    spix = th.tensor(spix.astype(np.int32)).to(device)
    # spix = rearrange(spix,'h w b -> b h w')

    # -- Global mean --
    # vid = 255.*vid
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
    vid = th.tensor(vid).to(device).float()
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
    SZV = th.mean(stds).item()
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

def scoreUnderErrorAndSegAccuracy(svMap, svSize, gtMap, gtSize, gtList, gtPos, mode):
    """
    This function is used to score 3D Undersegmentation Error and 3D
    Segmentation Accuracy for supervoxels.
    """

    # -- ensure shape --
    assert svSize.shape[-1] == svMap.shape[-1]

    # In case ground-truth is sparsely annotated
    svMap = svMap[:, :, gtPos] # H W T
    if svMap.shape != gtMap.shape:
        print('Error: gtMap and svMap dimension mismatch!')
        return None, None
    svMap = svMap.astype(np.uint32)

    # -- 3d ue and sa --
    if mode.lower() == '3d':
        svSize = svSize.sum(axis=1)
        gtSize = gtSize.sum(axis=1)
        gtUE = np.zeros(len(gtList))
        gtSA = np.zeros(len(gtList))

        for i in range(len(gtList)): # number of masks

            # -- get spix ids and counts which overlap with the gtMap --
            mask = (gtMap == gtList[i])
            svOnMask = svMap[mask]
            svOnID, svOnSize = np.unique(svOnMask, return_counts=True)

            # -- ue and sa --
            gtUE[i] = svSize[svOnID].sum()  # Corrected for zero-based indexing
            gtSA[i] = svOnSize[svOnSize >= 0.5 * svSize[svOnID]].sum()

        # UE = how many extra pixels for all spix in the GT.
        # SA = how many pixels are more than half in the GT
        UE = np.mean((gtUE - gtSize) / gtSize)
        SA = np.mean(gtSA / gtSize)

    elif mode.lower() == '2d':

        frameNum = gtMap.shape[2]
        gtUE = np.zeros((len(gtList), frameNum))
        gtSA = np.zeros((len(gtList), frameNum))
        ue = 0.

        # -- setup for alt --
        counts = th.from_numpy(svSize).T.long()
        spix = th.from_numpy(svMap).clone().long()
        spix = rearrange(spix,'h w t -> t h w')
        gtSeg = th.from_numpy(gtMap).clone().long()
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
            cond_ue = 0
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
        frameNum = gtMap.shape[2]
        gtUE = np.zeros((len(gtList), frameNum))
        gtSA = np.zeros((len(gtList), frameNum))
        ue = 0.

        for i in range(frameNum):
            thisGtMap = gtMap[:, :, i]
            thisSvMap = svMap[:, :, i]
            GtOn = np.unique(thisGtMap)

            for j in range(len(gtList)):
                if gtList[j] not in GtOn:
                    # print("continuing: ",i,j,GtOn)
                    continue
                mask = (thisGtMap == gtList[j])
                svOnMask = thisSvMap[np.where(mask)]
                svOnID, svOnSize = np.unique(svOnMask, return_counts=True)
                # if i < 30:
                #     print(i,svOnID[:10],svSize[svOnID, i].sum())
                gtUE[j, i] = svSize[svOnID, i].sum()  # Corrected indexing
                gtSA[j, i] = svOnSize[svOnSize >= 0.5 * svSize[svOnID, i]].sum()

        gtSize_mask = gtSize + (gtSize == 0)
        print("\n")
        # print(gtUE[0,:10])
        print(gtSA[0,:10])
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
