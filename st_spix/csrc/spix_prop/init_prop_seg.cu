
#ifndef OUT_OF_BOUNDS_LABEL
#define OUT_OF_BOUNDS_LABEL -1
#endif

#ifndef BAD_TOPOLOGY_LABEL 
#define BAD_TOPOLOGY_LABEL -2
#endif

#ifndef NUM_OF_CHANNELS 
#define NUM_OF_CHANNELS 3
#endif


#ifndef USE_COUNTS
#define USE_COUNTS 1
#endif


#ifndef OUT_OF_BOUNDS_LABEL
#define OUT_OF_BOUNDS_LABEL -1
#endif

#define THREADS_PER_BLOCK 512


#include "init_prop_seg.h"
#ifndef MY_SP_SHARE_H
#define MY_SP_SHARE_H
#include "../bass/share/sp.h"
#endif

#include <stdio.h>
#ifndef WIN32
#include <unistd.h>
#endif


__host__ void init_prop_seg(float* img, int* seg,
                            int* missing, bool* border,
                            superpixel_params* sp_params, 
                            const int nPixels, const int nMissing,
                            int nbatch, int xdim, int ydim, int nftrs,
                            const float3 J_i, const float logdet_Sigma_i, 
                            float i_std, int s_std, int nInnerIters,
                            const int nSPs, int nSPs_buffer,
                            float beta_potts_term,
                            int* debug, bool debug_fill){

    int num_block_sub = ceil( double(nPixels) / double(THREADS_PER_BLOCK) ); 
    dim3 BlockPerGridSub(num_block_sub,nbatch);
    // int num_block2 = ceil( double(nPixels*4) / double(THREADS_PER_BLOCK) ); 
    // dim3 BlockPerGrid2(num_block2,nbatch);
    int num_block = ceil( double(nMissing) / double(THREADS_PER_BLOCK) ); 
    dim3 BlockPerGrid(num_block,nbatch);
    dim3 ThreadPerBlock(THREADS_PER_BLOCK,1);
    cudaMemset(border, 0, nbatch*nPixels*sizeof(bool));
    fprintf(stdout,"nMissing: %d\n",nMissing);

    // int single_border = 0 ;
    // cudaMemset(post_changes, 0, nPixels*sizeof(post_changes_helper));
    // int nInnerIters = 4;
    // float3 J_i;
    // J_i.x = 1.0;
    // J_i.y = 1.0;
    // J_i.z = 1.0;
    // float logdet_Sigma_i = 1.0;
    for (int iter = 0 ; iter < nInnerIters; iter++){

      //  -- find border pixels --
      cudaMemset(border, 0, nbatch*nPixels*sizeof(bool));
      find_prop_border_pixels<<<BlockPerGrid,ThreadPerBlock>>>      \
         (seg, missing, border, nMissing, nbatch, xdim, ydim);

      //  -- update segmentation --
      for (int xmod3 = 0 ; xmod3 < 2; xmod3++){
        for (int ymod3 = 0; ymod3 < 2; ymod3++){
          update_prop_seg_subset<<<BlockPerGridSub,ThreadPerBlock>>>(img, seg, \
               border, sp_params, J_i, logdet_Sigma_i,\
               i_std, s_std, nPixels, nSPs,\
               nbatch, xdim, ydim, nftrs,\
               xmod3, ymod3, beta_potts_term);
        }
      }

      // -- copy for debug --
      if (debug_fill){
        int* debug_iter = debug+iter*nbatch*nPixels;
        cudaMemcpy(debug_iter,seg,nbatch*nPixels*sizeof(int),cudaMemcpyDeviceToDevice);
      }

    }

    //  -- find border pixels --
    cudaMemset(border, 0, nbatch*nPixels*sizeof(bool));
    find_prop_border_pixels<<<BlockPerGrid,ThreadPerBlock>>>      \
      (seg, missing, border, nMissing, nbatch, xdim, ydim);
}


__global__  void update_prop_seg_subset(
    float* img, int* seg, bool* border,
    superpixel_params* sp_params, 
    const float3 J_i, const float logdet_Sigma_i,
    float i_std, int s_std, const int nPts, const int nSuperpixels,
    const int nbatch, const int xdim, const int ydim, const int nftrs,
    const int xmod3, const int ymod3, const float beta_potts_term){   

    // -- init --
    int label_check;
    int idx = threadIdx.x + blockIdx.x*blockDim.x;
    int seg_idx = idx; 
    if (seg_idx>=nPts)  return;
    int x = seg_idx % xdim;  
    if (x % 2 != xmod3) return;
    int y = seg_idx / xdim;   
    if (y % 2 != ymod3) return;
    if (border[seg_idx]==0) return;

    //float beta = 4;
    //printf("(%d, %d) - %d, %d, %d \n", x,y , idx_cache,threadIdx.x );
    // const bool x_greater_than_0 = (x>0);
    // const bool y_greater_than_0 = (y>0);
    // const bool x_smaller_than_xdim_minus_1 = x<(xdim-1);
    // const bool y_smaller_than_ydim_minus_1 = y<(ydim-1);
    // if ((!x_greater_than_0)||(!y_greater_than_0)||(!x_smaller_than_xdim_minus_1)||(!y_smaller_than_ydim_minus_1)) return;
   
    // int C = seg[seg_idx]; // center 
    // N = S = W = E = OUT_OF_BOUNDS_LABEL; // init to out-of-bounds 
    // -- init neighbors --
    bool nbrs[9];
    float beta = beta_potts_term;
    bool isNvalid = 0;
    bool isSvalid = 0;
    bool isEvalid = 0;
    bool isWvalid = 0; 
    //printf("Beta: %f", beta);

    // -- init for now --
    int count_diff_nbrs_N=0;
    int count_diff_nbrs_S=0;
    int count_diff_nbrs_E=0;
    int count_diff_nbrs_W=0;

    // -- init --
    float2 res_max;
    res_max.x = -999999;
    res_max.y = seg[seg_idx];

    // --> north, south, east, west <--
    int N = -1, S = -1, E = -1, W = -1;
    if (y>0){ N = __ldg(&seg[idx-xdim]); }// top
    if (x<(xdim-1)){ E = __ldg(&seg[idx+1]); } // right
    if (y<(ydim-1)){ S = __ldg(&seg[idx+xdim]); } // below
    if (x>0){ W = __ldg(&seg[idx-1]); } // left

    // --> diags [north (east, west), south (east, west)] <--
    int NE = -1, NW = -1, SE = -1, SW = -1;

    // -- read labels of neighbors --
    if ((y>0) and (x<(xdim-1))){ NE = __ldg(&seg[idx-xdim+1]); } // top-right
    if ((y>0) and (x>0)){  NW = __ldg(&seg[idx-xdim-1]); } // top-left
    if ((x<(xdim-1)) and (y<(ydim-1))){ SE = __ldg(&seg[idx+xdim+1]); } // btm-right
    if ((x>0) and (y<(ydim-1))){ SW = __ldg(&seg[idx+xdim-1]); } // btm-left

    // -- read neighor labels for potts term --
    // check 8 nbrs and save result if valid to change to the last place of array
    // return how many nbrs different for potts term calculation

    //N :
    set_nbrs(NW, N, NE,  W, E, SW, S, SE,N, nbrs);
    count_diff_nbrs_N = ischangbale_by_nbrs(nbrs);
    isNvalid = nbrs[8] or (res_max.y == -1);
    if(!isNvalid) return;
    
    //E:
    set_nbrs(NW, N, NE,  W, E, SW, S, SE,E, nbrs);
    count_diff_nbrs_E = ischangbale_by_nbrs(nbrs);
    isEvalid = nbrs[8] or (res_max.y == -1);
    if(!isEvalid) return;

    //S :
    set_nbrs(NW, N, NE,  W, E, SW, S, SE,S, nbrs);
    count_diff_nbrs_S = ischangbale_by_nbrs(nbrs);
    isSvalid = nbrs[8] or (res_max.y == -1);
    if(!isSvalid) return;

    //W :
    set_nbrs(NW, N, NE,  W, E, SW, S, SE,W, nbrs);
    count_diff_nbrs_W = ischangbale_by_nbrs(nbrs);
    isWvalid = nbrs[8] or (res_max.y == -1);
    if(!isWvalid) return;

    // -- index image --
    float* imgC = img + idx * 3;
   
    // -- compute posterior --
    bool valid = N >= 0;
    label_check = N;
    if (valid){
      res_max = cal_posterior_new(imgC,seg,x,y,sp_params,label_check,
                                     J_i,logdet_Sigma_i,i_std,s_std,
                                     count_diff_nbrs_N,beta,res_max);
      // res_max.y = N;
    }

    valid = S>=0;
    label_check = S;
    if(valid and (label_check!=N)){
      res_max = cal_posterior_new(imgC,seg,x,y,sp_params,label_check,
                                     J_i,logdet_Sigma_i,i_std,s_std,
                                     count_diff_nbrs_S,beta,res_max);
      // res_max.y = S;
    }

    valid = W >= 0;
    label_check = W;
    if(valid and (label_check!=S)&&(label_check!=N)) {
      res_max = cal_posterior_new(imgC,seg,x,y,sp_params,label_check,
                                     J_i,logdet_Sigma_i,i_std,s_std,
                                     count_diff_nbrs_W,beta,res_max);
      // res_max.y = W;
    }
    
    valid = E >= 0;
    label_check = E;
    if(valid and (label_check!=W)&&(label_check!=S)&&(label_check!=N)){
      res_max= cal_posterior_new(imgC,seg,x,y,sp_params,label_check,
                                    J_i,logdet_Sigma_i,i_std,s_std,
                                    count_diff_nbrs_E,beta,res_max);
      // res_max.y = E;
    }

    seg[seg_idx] = res_max.y;
    return;
}

__global__
void find_prop_border_pixels(const int* seg, const int* missing,
                             bool* border, const int nMissing,
                             const int nbatch, const int xdim, const int ydim){   

    // --> cuda indices <--
    int _idx = threadIdx.x + blockIdx.x * blockDim.x;  
    if (_idx>=nMissing) return; 

    // --> space coordinates <--
    // bool is_filled = filled[_idx];
    int idx = missing[_idx];
    int x = idx % xdim;
    int y = idx / xdim;

    // -- dont fill twice --
    // if (is_filled){
    //   border[idx] = 0;
    //   return;
    // }

    // --> north, south, east, west <--
    int N = -1, S = -1, E = -1, W = -1, C = -1;

    // -- check out of bounds --
    C = seg[idx]; // self
    if (y>0){ N = __ldg(&seg[idx-xdim]); } // above
    if (x>0){ W = __ldg(&seg[idx-1]); } // left
    if (y<(ydim-1)){ S = __ldg(&seg[idx+xdim]); } // below
    if (x<(xdim-1)){ E = __ldg(&seg[idx+1]); } // right
   
    // if the center is "-1" and any neighbor is valid, it is an edge
    bool valid = (N >= 0) or (W >= 0) or (S >= 0) or (E >= 0);
    // bool invalid = (N < 0) or (W < 0) or (S < 0) or (E < 0) or (C < 0);
    if (valid and (C<0)){
      border[idx]=1;  
      // filled[_idx]=1;
    }
    // border[idx]=1;
    return;        
}
