"""

   Vizualize SLIC

"""

import torch as th
import torchvision.io as iio
import torchvision.utils as tv_utils
import torch.nn.functional as th_f
from spix_paper.spix_utils.slic_img_iter import *

def main():

    # -- setup --
    fname = "../superpixel_paper/output/figures/cute_chick_cat.png"
    pix_ftrs = iio.read_image(fname)[None,:]
    stoken_size = 12
    M = 1.
    n_iter = 20
    print(pix_ftrs.shape)
    exit()

    # -- unpack --
    B = pix_ftrs.shape[0]
    height, width = pix_ftrs.shape[-2:]
    if not hasattr(stoken_size,"__len__"):
        stoken_size = [stoken_size,stoken_size]
    sheight, swidth = stoken_size[0],stoken_size[1]
    nsp_height = height // sheight
    nsp_width = width // swidth
    nsp = nsp_height * nsp_width
    full_grad = grad_type == "full"

    # -- add grid --
    if th.is_tensor(M): M = M[:,None]
    pix_ftrs = append_grid(pix_ftrs[:,None],M/stoken_size[0],normz=True)[:,0]
    shape = pix_ftrs.shape

    # -- init centroids/inds --
    sftrs, ilabel = init_centroid(pix_ftrs, nsp_width, nsp_height)
    abs_indices = get_abs_indices(ilabel, nsp_width)
    mask = (abs_indices[1] >= 0) * (abs_indices[1] < nsp)
    pix_ftrs = pix_ftrs.reshape(*pix_ftrs.shape[:2], -1)
    permuted_pix_ftrs = pix_ftrs.permute(0, 2, 1).contiguous()
    coo_inds = abs_indices[:,mask]
    # print(coo_inds.shape,mask.shape,permuted_pix_ftrs.shape)

    # -- determine grad --
    with torch.set_grad_enabled(False):
        for k in range(n_iter):

            # # -- compute all affinities  --
            # pwd_fxn = PairwiseDistFunction.apply
            # dist_matrix = pwd_fxn(pix_ftrs, sftrs, ilabel, nsp_width, nsp_height)
            # print(dist_matrix)
            # print(dist_matrix.std())

            # # -- sample only relevant affinity --
            # sparse_sims = (-sm_scale*dist_matrix).softmax(1)
            # reshaped_sparse_sims = sparse_sims.reshape(-1)
            # sparse_sims = torch.sparse_coo_tensor(abs_indices[:,mask],
            #                                       reshaped_sparse_sims[mask])
            # sims = sparse_sims.to_dense().contiguous()

            # -- compute all affinities  --
            sparse_sims, sims = _update_sims(pix_ftrs,sftrs,ilabel,
                                             nsp_width,nsp_height,
                                             sm_scale,coo_inds,mask)
            # -- update centroids --
            if k < n_iter - 1:
                sftrs = _update_sftrs(sims,permuted_pix_ftrs)

    pass

if __name__ == "__main__":
    main()
