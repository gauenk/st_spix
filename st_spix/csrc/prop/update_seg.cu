
#ifndef OUT_OF_BOUNDS_LABEL
#define OUT_OF_BOUNDS_LABEL -1
#endif
#define THREADS_PER_BLOCK 512

#include <assert.h>
#include <stdio.h>
#include "seg_utils.h"
#include "update_seg.h"

/**********************************************
***********************************************


         Segmentation Update Kernel


***********************************************
**********************************************/


/*
* Update the superpixel labels for pixels 
* that are on the boundary of the superpixels
* and on the (xmod3, ymod3) position of 3*3 block
*/

__global__
void update_seg_subset(float* img, int* seg, bool* border,
                       spix_params* sp_params, const float3 pix_cov,
                       const float logdet_pix_cov,  const float potts,
                       const int npix, const int nbatch,
                       const int xdim, const int ydim, const int nftrs,
                       const int xmod3, const int ymod3){

    int label_check;
    int idx = threadIdx.x + blockIdx.x*blockDim.x;
   // idx = idx_img;

    int pix_idx = idx; 
    if (pix_idx>=npix)  return;
    // todo; add batch info here.

    int x = pix_idx % xdim;  
    if (x % 2 != xmod3) return;
    int y = pix_idx / xdim;   
    if (y % 2 != ymod3) return;
    
    if (border[pix_idx]==0) return;
    // strides of 2*2

    //float beta = 4;
    //printf("(%d, %d) - %d, %d, %d \n", x,y , idx_cache,threadIdx.x );
    const bool x_greater_than_1 = (x>1);
    const bool y_greater_than_1 = (y>1);
    const bool x_smaller_than_xdim_minus_1 = x<(xdim-1);
    const bool y_smaller_than_ydim_minus_1 = y<(ydim-1);
    if ((!x_greater_than_1)||(!y_greater_than_1)||(!x_smaller_than_xdim_minus_1)||(!y_smaller_than_ydim_minus_1)) return;
   
    bool nbrs[9];
    //float potts_term[4];
    //potts_term[0] = potts_term[1] = potts_term[2] = potts_term[3] = 0;
    bool isNvalid = 0;
    bool isSvalid = 0;
    bool isEvalid = 0;
    bool isWvalid = 0; 
    //printf("Potts: %f", potts);

    // -- count neighbors --
    int count_diff_nbrs_N=0;
    int count_diff_nbrs_S=0;
    int count_diff_nbrs_E=0;
    int count_diff_nbrs_W=0;

    //-- init max --
    float2 res_max;
    res_max.x = -9999;

    // -- read superpixel labels --
    int NW =__ldg(&seg[pix_idx-xdim-1]);
    int N = __ldg(&seg[pix_idx-xdim]);
    int NE = __ldg(&seg[pix_idx-xdim+1]);
    int W = __ldg(&seg[pix_idx-1]);
    int E = __ldg(&seg[pix_idx+1]);
    int SW = __ldg(&seg[pix_idx+xdim-1]);
    int S = __ldg(&seg[pix_idx+xdim]);
    int SE =__ldg(&seg[pix_idx+xdim+1]);  

    //N :
    set_nbrs(NW, N, NE,  W, E, SW, S, SE, N, nbrs);
    count_diff_nbrs_N = ischangbale_by_nbrs(nbrs);
    isNvalid = nbrs[8];
    if(!isNvalid) return;
    
    //W :
    set_nbrs(NW, N, NE,  W, E, SW, S, SE, W, nbrs);
    count_diff_nbrs_W = ischangbale_by_nbrs(nbrs);
    isWvalid = nbrs[8];
    if(!isWvalid) return;

    //S :
    set_nbrs(NW, N, NE,  W, E, SW, S, SE, S, nbrs);
    count_diff_nbrs_S = ischangbale_by_nbrs(nbrs);
    isSvalid = nbrs[8];
    if(!isSvalid) return;

    //E:
    set_nbrs(NW, N, NE,  W, E, SW, S, SE, E, nbrs);
    count_diff_nbrs_E = ischangbale_by_nbrs(nbrs);
    isEvalid = nbrs[8];
    if(!isEvalid) return;
   
    // -- index image --
    float* imgC = img + idx * 3;

    // -- compute posterior --
    label_check = N;
    assert(label_check >= 0);
    res_max = calc_joint(imgC,seg,x,y,sp_params,label_check,
                                  pix_cov,logdet_pix_cov,
                                  count_diff_nbrs_N,potts,res_max);
    label_check = S;
    assert(label_check >= 0);
    if(label_check!=N)
      res_max = calc_joint(imgC,seg,x,y,sp_params,label_check,
                                    pix_cov,logdet_pix_cov,
                                    count_diff_nbrs_S,potts,res_max);

    label_check = W;
    assert(label_check >= 0);
    if ( (label_check!=S)&&(label_check!=N))
      res_max = calc_joint(imgC,seg,x,y,sp_params,label_check,
                                    pix_cov,logdet_pix_cov,
                                    count_diff_nbrs_W,potts,res_max);
    
    label_check = E;
    assert(label_check >= 0);
    if((label_check!=W)&&(label_check!=S)&&(label_check!=N))
      res_max = calc_joint(imgC,seg,x,y,sp_params,label_check,
                                    pix_cov,logdet_pix_cov,
                                    count_diff_nbrs_E,potts,res_max);
    seg[pix_idx] = res_max.y;
    return;
}



__device__ float2 calc_joint(float* imgC, int* seg, int width_index, int height_index,
                             spix_params* sp_params, int seg_idx,
                             float3 pix_var, float logdet_pix_var,
                             float neigh_neq, float beta, float2 res_max){

    // -- init res --
    float res = -1000; // some large negative number // why?
    /* float res = 0.; */

    // -- appearance --
    const float x0 = __ldg(&imgC[0])-__ldg(&sp_params[seg_idx].mu_app.x);
    const float x1 = __ldg(&imgC[1])-__ldg(&sp_params[seg_idx].mu_app.y);
    const float x2 = __ldg(&imgC[2])-__ldg(&sp_params[seg_idx].mu_app.z);
    /* const float sigma_a_x = __ldg(&sp_params[seg_idx].sigma_app.x); */
    /* const float sigma_a_y = __ldg(&sp_params[seg_idx].sigma_app.y); */
    /* const float sigma_a_z = __ldg(&sp_params[seg_idx].sigma_app.z); */
    const float sigma_a_x = 1./pix_var.x;
    const float sigma_a_y = 1./pix_var.y;
    const float sigma_a_z = 1./pix_var.z;
    /* const float logdet_sigma_app = __ldg(&sp_params[seg_idx].logdet_sigma_app); */
    const float logdet_sigma_app = 3.*log(sigma_a_x);

    // -- shape --
    const int d0 = width_index - __ldg(&sp_params[seg_idx].mu_shape.x);
    const int d1 = height_index - __ldg(&sp_params[seg_idx].mu_shape.y);
    const float sigma_s_x = __ldg(&sp_params[seg_idx].sigma_shape.x);
    const float sigma_s_y = __ldg(&sp_params[seg_idx].sigma_shape.y);
    const float sigma_s_z = __ldg(&sp_params[seg_idx].sigma_shape.z);
    const float logdet_sigma_shape = __ldg(&sp_params[seg_idx].logdet_sigma_shape);

    // -- appearance [sigma is actually \sigma^2] --
    // res = res - x0*x0 - x1*x1 - x2*x2;
    res = res - x0*x0/sigma_a_x - x1*x1/sigma_a_y - x2*x2/sigma_a_z;
    res = res - logdet_sigma_app;

    // -- shape [sigma is actually \Sigma^{(-1)}, the inverse] --
    res = res - d0*d0*sigma_s_x - d1*d1*sigma_s_z - 2*d0*d1*sigma_s_y; // sign(s_y) = -1
    res = res - logdet_sigma_shape;

    // -- prior --
    /* res = res - sp_params[seg_idx].prior_lprob; */

    // -- potts term --
    res = res - beta*neigh_neq;

    // -- update res --
    if( res>res_max.x ){
      res_max.x = res;
      res_max.y = seg_idx;
    }

    return res_max;
}


/**********************************************
***********************************************


              Main Function


***********************************************
**********************************************/

__host__ void update_seg(float* img, int* seg, bool* border,
                         spix_params* sp_params, const int niters,
                         const float3 pix_cov, const float logdet_pix_cov,
                         const float potts, const int npix,
                         int nbatch, int xdim, int ydim, int nftrs){
    
    int num_block = ceil( double(npix) / double(THREADS_PER_BLOCK) ); 
    dim3 ThreadPerBlock(THREADS_PER_BLOCK,1);
    dim3 BlockPerGrid(num_block,nbatch);
    for (int iter = 0 ; iter < niters; iter++){
        cudaMemset(border, 0, npix*sizeof(bool));
        find_border_pixels<<<BlockPerGrid,ThreadPerBlock>>>(seg,border,npix,xdim,ydim);
        for (int xmod3 = 0 ; xmod3 <2; xmod3++){
            for (int ymod3 = 0; ymod3 <2; ymod3++){
                update_seg_subset<<<BlockPerGrid,ThreadPerBlock>>>(img, seg, \
                     border, sp_params, pix_cov, logdet_pix_cov, potts,\
                     npix, nbatch, xdim, ydim, nftrs, xmod3, ymod3);
            }
        }
    }
    cudaMemset(border, 0, npix*sizeof(bool));
    find_border_pixels<<<BlockPerGrid,ThreadPerBlock>>>(\
           seg, border, npix, xdim, ydim);
}


