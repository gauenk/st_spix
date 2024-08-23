
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
#include "init_prop_seg_space.h"
#ifndef MY_SP_SHARE_H
#define MY_SP_SHARE_H
#include "../bass/share/sp.h"
#endif
#include "../bass/core/Superpixels.h"
// #include "../share/utils.h"

#include <stdio.h>
#ifndef WIN32
#include <unistd.h>
#endif

__host__ void init_prop_seg_space(float* img, int* seg,
                                  int* missing, bool* border,
                                  superpixel_params* sp_params, 
                                  const int nPixels, const int nMissing,
                                  int nbatch, int xdim, int ydim, int nftrs,
                                  const float3 J_i, const float logdet_Sigma_i, 
                                  float i_std, int s_std, int nInnerIters,
                                  const int nSPs, int nSPs_buffer,
                                  float beta_potts_term,
                                  int* debug_spix, bool* debug_border, bool debug_fill){

    int num_block_sub = ceil( double(nPixels) / double(THREADS_PER_BLOCK) ); 
    dim3 BlockPerGridSub(num_block_sub,nbatch);
    // int num_block2 = ceil( double(nPixels*4) / double(THREADS_PER_BLOCK) ); 
    // dim3 BlockPerGrid2(num_block2,nbatch);
    int num_block = ceil( double(nMissing) / double(THREADS_PER_BLOCK) ); 
    dim3 BlockPerGrid(num_block,nbatch);
    dim3 ThreadPerBlock(THREADS_PER_BLOCK,1);
    cudaMemset(border, 0, nbatch*nPixels*sizeof(bool));
    // fprintf(stdout,"nMissing: %d\n",nMissing);
    gpuErrchk( cudaPeekAtLastError() );
    gpuErrchk( cudaDeviceSynchronize() );
    const int sizeofint = sizeof(int);
    int iter = 0; // for debug

    // -- init exit condition --
    bool any_update_cpu;
    bool* any_update_gpu;
    try {
      throw_on_cuda_error(cudaMalloc((void**)&any_update_gpu,sizeof(bool)));
    }
    catch (thrust::system_error& e) {
        std::cerr << "CUDA error after cudaMalloc: " << e.what() << std::endl;
    }
    any_update_cpu = true;

    // -- init debug variables --
    int prev_neg;
    int num_neg_cpu;
    int* num_neg_gpu;
    try {
      throw_on_cuda_error(cudaMalloc((void**)&num_neg_gpu,sizeofint));
    }
    catch (thrust::system_error& e) {
        std::cerr << "CUDA error after cudaMalloc: " << e.what() << std::endl;
    }
    num_neg_cpu = 1;
    prev_neg = 1;

    //  -- Step 1) find border pixels --
    cudaMemset(num_neg_gpu, 0, sizeof(int));
    cudaMemset(border, 0, nbatch*nPixels*sizeof(bool));
    find_prop_border_pixels<<<BlockPerGrid,ThreadPerBlock>>>            \
      (seg, missing, border, nMissing, nbatch, xdim, ydim, num_neg_gpu);
    gpuErrchk( cudaPeekAtLastError() );
    gpuErrchk( cudaDeviceSynchronize() );
    cudaMemcpy(&num_neg_cpu,num_neg_gpu,sizeof(int),cudaMemcpyDeviceToHost);
    // fprintf(stdout,"num negative spix: %d\n",num_neg_cpu);

    while (any_update_cpu == true){

      //  -- Step 2) update segmentation --
      cudaMemset(any_update_gpu, 0, sizeof(bool));
      for (int xmod3 = 0 ; xmod3 < 2; xmod3++){
        for (int ymod3 = 0; ymod3 < 2; ymod3++){
          update_prop_seg_subset_space<<<BlockPerGridSub,ThreadPerBlock>>>
            (img, seg, border, sp_params, nPixels, nSPs,
             nbatch, xdim, ydim, xmod3, ymod3, beta_potts_term, any_update_gpu);
          // gpuErrchk( cudaPeekAtLastError() );
          // gpuErrchk( cudaDeviceSynchronize() );
        }
      }
      cudaMemcpy(&any_update_cpu,any_update_gpu,sizeof(bool),cudaMemcpyDeviceToHost);

      // -- copy for debug --
      if (debug_fill and (iter < nInnerIters)){
        int* debug_spix_iter = debug_spix+iter*nbatch*nPixels;
        cudaMemcpy(debug_spix_iter,seg,
                   nbatch*nPixels*sizeof(int),cudaMemcpyDeviceToDevice);
        bool* debug_border_iter = debug_border+iter*nbatch*nPixels;
        cudaMemcpy(debug_border_iter,border,
                   nbatch*nPixels*sizeof(bool),cudaMemcpyDeviceToDevice);
        // gpuErrchk( cudaPeekAtLastError() );
        // gpuErrchk( cudaDeviceSynchronize() );
        // fprintf(stdout,"c\n");
        // cudaDeviceSynchronize();
      }
      iter++;

      // -- update previous --
      // if ((iter>0) and (num_neg_cpu == prev_neg)){
      //   fprintf(stdout,"An error of some type, the border won't shrink.\n");
      //   break;
      // }
      // prev_neg = num_neg_cpu;

    }

    // -- [Debug] find border pixels --
    if (debug_fill){
      // num_neg = 0;
      cudaMemset(num_neg_gpu, 0, sizeof(int));
      cudaMemset(border, 0, nbatch*nPixels*sizeof(bool));
      find_prop_border_pixels<<<BlockPerGrid,ThreadPerBlock>>>      \
        (seg, missing, border, nMissing, nbatch, xdim, ydim, num_neg_gpu);
      gpuErrchk( cudaPeekAtLastError() );
      gpuErrchk( cudaDeviceSynchronize() );
      cudaMemcpy(&num_neg_cpu,num_neg_gpu,sizeof(int),cudaMemcpyDeviceToHost);
      if (num_neg_cpu > 0){
        fprintf(stdout,"negative spix exist.\n");
      }
    }

    // -- free memory --
    cudaFree(num_neg_gpu);
    cudaFree(any_update_gpu);
    // gpuErrchk( cudaPeekAtLastError() );
    // gpuErrchk( cudaDeviceSynchronize() );

}


__global__  void update_prop_seg_subset_space(
    float* img, int* seg, bool* border,
    superpixel_params* sp_params, 
    const int npix, const int nSuperpixels,
    const int nbatch, const int xdim, const int ydim,
    const int xmod3, const int ymod3,
    const float beta_potts_term, bool* any_update){   

    // -- init --
    int label_check;
    int idx = threadIdx.x + blockIdx.x*blockDim.x;
    int seg_idx = idx; 
    if (seg_idx>=npix)  return;
    int x = seg_idx % xdim;  
    if (x % 2 != xmod3) return;
    int y = seg_idx / xdim;   
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
    int original_label = res_max.y;

    // --> north, south, east, west <--
    int N = -1, S = -1, E = -1, W = -1;
    if (y>0){ N = __ldg(&seg[idx-xdim]); }// top
    if (x>0){ W = __ldg(&seg[idx-1]); } // left
    if (x<(xdim-1)){ E = __ldg(&seg[idx+1]); } // right
    if (y<(ydim-1)){ S = __ldg(&seg[idx+xdim]); } // below
    if ((N < 0) and (S < 0) and (E < 0) and (W < 0)){ // don't update all invalid
      return;
    }

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
    set_nbrs(NW, N, NE, W, E, SW, S, SE,N, nbrs);
    count_diff_nbrs_N = ischangbale_by_nbrs(nbrs);
    //E:
    set_nbrs(NW, N, NE, W, E, SW, S, SE,E, nbrs);
    count_diff_nbrs_E = ischangbale_by_nbrs(nbrs);
    //S :
    set_nbrs(NW, N, NE, W, E, SW, S, SE,S, nbrs);
    count_diff_nbrs_S = ischangbale_by_nbrs(nbrs);
    //W :
    set_nbrs(NW, N, NE, W, E, SW, S, SE,W, nbrs);
    count_diff_nbrs_W = ischangbale_by_nbrs(nbrs);

    // -- index image --
    float* imgC = img + idx * 3;
   
    // -- compute posterior --
    bool valid = N >= 0;
    label_check = N;
    if (valid){
      res_max = calc_space(x,y,sp_params,label_check,res_max);
    }

    valid = S>=0;
    label_check = S;
    if(valid && (label_check!=N)){
      res_max = calc_space(x,y,sp_params,label_check,res_max);
    }

    valid = W >= 0;
    label_check = W;
    if(valid && (label_check!=S)&&(label_check!=N)) {
      res_max = calc_space(x,y,sp_params,label_check,res_max);
    }
    
    valid = E >= 0;
    label_check = E;
    if(valid && (label_check!=W)&&(label_check!=S)&&(label_check!=N)){
      res_max = calc_space(x,y,sp_params,label_check,res_max);
    }

    seg[seg_idx] = res_max.y;

    // -- update any update --
    if (res_max.y != original_label){
      border[seg_idx] = 0;
      *any_update = true;
    }

    // -- update boarder --
    bool updated_invalid = (original_label == -1) and (res_max.y > 0);
    if (updated_invalid and (N < 0) and (y>0)){
      border[idx-xdim] = 1;
    }
    if (updated_invalid and (W < 0) and (x>0)){
      border[idx-1] = 1;
    }
    if (updated_invalid and (E < 0) and (x<(xdim-1))){
      border[idx+1] = 1;
    }
    if (updated_invalid and (S < 0) and (y<(ydim-1))){
      border[idx+xdim] = 1;
    }

    bool valid_access = (y>0) and (x<(xdim-1));
    if (updated_invalid and (NE < 0) and valid_access){
      border[idx-xdim+1] = 1;
    }
    valid_access = (y>0) and (x>0);
    if (updated_invalid and (NW < 0) and valid_access){
      border[idx-xdim-1] = 1;
    }
    valid_access = (x<(xdim-1)) and (y<(ydim-1));
    if (updated_invalid and (SE < 0) and valid_access){
      border[idx+xdim+1] = 1;
    }
    valid_access = (x>0) and (y<(ydim-1));
    if (updated_invalid and (SW < 0) and valid_access){
      border[idx+xdim-1] = 1;
    }

    return;
}


