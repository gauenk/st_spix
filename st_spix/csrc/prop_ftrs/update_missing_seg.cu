
#ifndef OUT_OF_BOUNDS_LABEL
#define OUT_OF_BOUNDS_LABEL -1
#endif
#define THREADS_PER_BLOCK 512

#include <assert.h>
#include <stdio.h>
#include "seg_utils.h"
#include "update_missing_seg.h"

/**********************************************
***********************************************


         Segmentation Update Kernel


***********************************************
**********************************************/


/*
* Update the superpixel labels for pixels 
* that are on the boundary of the superpixels
* and on the (width_mod, height_mod) position of 3*3 block
*/

__global__
void update_missing_seg_subset(float* img, int* seg, bool* border, bool* missing,
                               superpixel_params* sp_params, const float3 pix_var,
                               const float logdet_pix_var,  const float potts,
                               const int npix, const int nbatch,
                               const int width, const int height, const int nftrs,
                               const int width_mod, const int height_mod){

    int label_check;
    int idx = threadIdx.x + blockIdx.x*blockDim.x;
   // idx = idx_img;
    int pix_idx = idx; 
    if (pix_idx>=npix)  return;
    // todo; add batch info here.

    int x = pix_idx % width;  
    if (x % 2 != width_mod) return;
    int y = pix_idx / width;   
    if (y % 2 != height_mod) return;
    
    if (border[pix_idx]==0) return;
    // strides of 2*2

    //float beta = 4;
    //printf("(%d, %d) - %d, %d, %d \n", x,y , idx_cache,threadIdx.x );
    const bool x_greater_than_1 = (x>1);
    const bool y_greater_than_1 = (y>1);
    const bool x_smaller_than_width_minus_1 = x<(width-1);
    const bool y_smaller_than_height_minus_1 = y<(height-1);
    if ((!x_greater_than_1)||(!y_greater_than_1)||(!x_smaller_than_width_minus_1)||(!y_smaller_than_height_minus_1)) return;
   
    bool nbrs[9];
    bool isNvalid = 0;
    bool isSvalid = 0;
    bool isEvalid = 0;
    bool isWvalid = 0; 

    // -- count neighbors --
    int count_diff_nbrs_N=0;
    int count_diff_nbrs_S=0;
    int count_diff_nbrs_E=0;
    int count_diff_nbrs_W=0;

    //-- init max --
    float2 res_max;
    res_max.x = -9999;
    res_max.y = __ldg(&seg[pix_idx]);

    // -- read if missing --
    bool mNW = missing[pix_idx-width-1];
    bool mN = missing[pix_idx-width];
    bool mNE = missing[pix_idx-width+1];
    bool mW = missing[pix_idx-1];
    bool mE = missing[pix_idx+1];
    bool mSW = missing[pix_idx+width-1];
    bool mS = missing[pix_idx+width];
    bool mSE = missing[pix_idx+width+1];

    // // -- read superpixel labels --
    // int NW = (mNW==1) ? __ldg(&seg[pix_idx-width-1]) : -1;
    // int N = (mN == 1) ? __ldg(&seg[pix_idx-width]) : -1;
    // int NE =(mNE == 1) ? __ldg(&seg[pix_idx-width+1]) : -1;
    // int W = (mW == 1) ? __ldg(&seg[pix_idx-1]) : -1;
    // int E = (mE == 1) ? __ldg(&seg[pix_idx+1]) : -1;
    // int SW =(mSW == 1)? __ldg(&seg[pix_idx+width-1]) : -1;
    // int S = (mS == 1) ? __ldg(&seg[pix_idx+width]) : -1;
    // int SE =(mSE == 1)? __ldg(&seg[pix_idx+width+1]): -1;  

    int NW =__ldg(&seg[pix_idx-width-1]);
    int N = __ldg(&seg[pix_idx-width]);
    int NE = __ldg(&seg[pix_idx-width+1]);
    int W = __ldg(&seg[pix_idx-1]);
    int E = __ldg(&seg[pix_idx+1]);
    int SW = __ldg(&seg[pix_idx+width-1]);
    int S = __ldg(&seg[pix_idx+width]);
    int SE =__ldg(&seg[pix_idx+width+1]);  


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
    // assert(label_check >= 0);
    if (mN == 1)
      res_max = cal_prop_likelihood(imgC,seg,x,y,sp_params,label_check,
                                    pix_var,logdet_pix_var,
                                    count_diff_nbrs_N,potts,res_max);
    label_check = S;
    // assert(label_check >= 0);
    if( (label_check!=N) &&(mS==1) )
    res_max = cal_prop_likelihood(imgC,seg,x,y,sp_params,label_check,
                                  pix_var,logdet_pix_var,
                                  count_diff_nbrs_S,potts,res_max);

    label_check = W;
    // assert(label_check >= 0);
    if ( (label_check!=S)&&(label_check!=N)&&(mW==1))   
      res_max = cal_prop_likelihood(imgC,seg,x,y,sp_params,label_check,
                                   pix_var,logdet_pix_var,
                                   count_diff_nbrs_W,potts,res_max);
    
    label_check = E;
    // assert(label_check >= 0);
    if((label_check!=W)&&(label_check!=S)&&(label_check!=N)&&(mE==1))
      res_max = cal_prop_likelihood(imgC,seg,x,y,sp_params,label_check,
                                   pix_var,logdet_pix_var,
                                   count_diff_nbrs_E,potts,res_max);
    seg[pix_idx] = res_max.y;
    return;
}



/**********************************************
***********************************************


              Main Function


***********************************************
**********************************************/

__host__ void update_missing_seg(float* img, int* seg, bool* border, bool* missing,
                                 superpixel_params* sp_params, const int niters,
                                 const float3 pix_ivar, const float logdet_pix_var,
                                 const float potts, const int npix,
                                 int nbatch, int width, int height, int nftrs){
    
    int num_block = ceil( double(npix) / double(THREADS_PER_BLOCK) ); 
    dim3 ThreadPerBlock(THREADS_PER_BLOCK,1);
    dim3 BlockPerGrid(num_block,nbatch);
    // printf("pix_var.x,potts: %2.5f,%2.5f\n",pix_var.x,potts);
    for (int iter = 0 ; iter < niters; iter++){
        cudaMemset(border, 0, npix*sizeof(bool));
        find_border_pixels<<<BlockPerGrid,ThreadPerBlock>>>(seg, border, npix,
                                                            width, height);
        for (int width_mod = 0 ; width_mod <2; width_mod++){
            for (int height_mod = 0; height_mod <2; height_mod++){
                update_missing_seg_subset<<<BlockPerGrid,ThreadPerBlock>>>(img, seg, \
                     border, missing, sp_params, pix_ivar, logdet_pix_var, potts,\
                     npix, nbatch, width, height, nftrs, width_mod, height_mod);
            }
        }
    }
    cudaMemset(border, 0, npix*sizeof(bool));
    find_border_pixels<<<BlockPerGrid,ThreadPerBlock>>>(seg,border,npix,width,height);
}


