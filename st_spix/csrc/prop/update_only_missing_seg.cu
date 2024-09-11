
/*******************************************************

      This runs the posterior update using only the
      missing segmentations.

*******************************************************/

// -- cpp imports --
#include <stdio.h>
#include <cuda.h>
#include <cuda_runtime.h>
#include <cuda/std/type_traits>
#include <torch/types.h>
#include <cuda.h>
#include <cuda_runtime.h>
#include <torch/extension.h>


// update_only_missing_seg(img, seg, seg_potts_label, border, sp_params,
//                         J_i, logdet_Sigma_i, cal_cov, i_std, s_std,
//                         nInnerIters, npix, nspix, nspix_buffer,
//                         nbatch, dim_x, dim_y, nftrs,
//                         sp_options.beta_potts_term);




__global__
void update_missing_seg_nn(int* seg, float* centers, bool* border, 
                        const int nbatch, const int width, const int height,
                        const int npix, const int xmod3, const int ymod3){   

    // -- init --
    int label_check;
    int idx = threadIdx.x + blockIdx.x*blockDim.x;
    int seg_idx = idx; 
    if (seg_idx>=npix)  return;
    int x = seg_idx % width;  
    if (x % 2 != xmod3) return;
    int y = seg_idx / width;   
    if (y % 2 != ymod3) return;
    if (border[seg_idx]==0) return;

    // -- init neighbors --
    bool nbrs[9];
    float beta = beta_potts_term;
    bool isNvalid = 0;
    bool isSvalid = 0;
    bool isEvalid = 0;
    bool isWvalid = 0; 

    // -- init for now --
    int count_diff_nbrs_N=0;
    int count_diff_nbrs_S=0;
    int count_diff_nbrs_E=0;
    int count_diff_nbrs_W=0;

    // -- init --
    float2 res_max;
    res_max.x = -999999;
    res_max.y = seg[seg_idx];
    // int C = res_max.y;

    // --> north, south, east, west <--
    int N = -1, S = -1, E = -1, W = -1;
    if (x>0){ W = __ldg(&seg[idx-1]); } // left
    if (y>0){ N = __ldg(&seg[idx-width]); }// top
    if (x<(width-1)){ E = __ldg(&seg[idx+1]); } // right
    if (y<(height-1)){ S = __ldg(&seg[idx+width]); } // below

    // --> diags [north (east, west), south (east, west)] <--
    int NE = -1, NW = -1, SE = -1, SW = -1;

    // -- read labels of neighbors --
    if ((y>0) and (x<(width-1))){ NE = __ldg(&seg[idx-width+1]); } // top-right
    if ((y>0) and (x>0)){  NW = __ldg(&seg[idx-width-1]); } // top-left
    if ((x<(width-1)) and (y<(height-1))){ SE = __ldg(&seg[idx+width+1]); } // btm-right
    if ((x>0) and (y<(height-1))){ SW = __ldg(&seg[idx+width-1]); } // btm-left

    // -- read neighor labels for potts term --
    // check 8 nbrs and save result if valid to change to the last place of array
    // return how many nbrs different for potts term calculation

    //N :
    set_nbrs(NW, N, NE, W, E, SW, S, SE,N, nbrs);
    count_diff_nbrs_N = ischangbale_by_nbrs(nbrs);
    // isNvalid = nbrs[8] or (res_max.y == -1);
    // if(!isNvalid) return;
    
    //E:
    set_nbrs(NW, N, NE, W, E, SW, S, SE,E, nbrs);
    count_diff_nbrs_E = ischangbale_by_nbrs(nbrs);
    // isEvalid = nbrs[8] or (res_max.y == -1);
    // if(!isEvalid) return;

    //S :
    set_nbrs(NW, N, NE, W, E, SW, S, SE,S, nbrs);
    count_diff_nbrs_S = ischangbale_by_nbrs(nbrs);
    // isSvalid = nbrs[8] or (res_max.y == -1);
    // if(!isSvalid) return;

    //W :
    set_nbrs(NW, N, NE, W, E, SW, S, SE,W, nbrs);
    count_diff_nbrs_W = ischangbale_by_nbrs(nbrs);
    // isWvalid = nbrs[8] or (res_max.y == -1);
    // if(!isWvalid) return;

    // -- compute posterior --
    bool valid = N >= 0;
    label_check = N;
    if (valid){
      isotropic_space(res, x, y, centers+label_check*2);
    }

    valid = S>=0;
    label_check = S;
    if(valid && (label_check!=N)){
      isotropic_space(res, x, y, centers+label_check*2);
    }

    valid = W >= 0;
    label_check = W;
    if(valid && (label_check!=S)&&(label_check!=N)) {
      isotropic_space(res, x, y, centers+label_check*2);
    }
    
    valid = E >= 0;
    label_check = E;
    if(valid && (label_check!=W)&&(label_check!=S)&&(label_check!=N)){
      isotropic_space(res, x, y, centers+label_check*2);
    }

    seg[seg_idx] = res_max.y;
    return;
}


