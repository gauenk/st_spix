
/*******************************************************

            This is just an initialization
      Fills missing pixels by spatial location alone.
            We want to get rid of any "-1"s

*******************************************************/

// -- cpp imports --
#include <stdio.h>
// #include <math.h>
#include <cuda.h>
#include <cuda_runtime.h>
// #include <cmath>


// #include <cuda/std/type_traits>
// #include <torch/types.h>
// #include <cuda.h>
// #include <cuda_runtime.h>
// #include <torch/extension.h>
// #include <vector>
// #include "base.h"
#include "pch.h"

// -- local import --
#include "seg_utils.h"
// #include "init_utils.h"
#include "fill_missing.h"

// -- define --
#define CHECK_CUDA(x) TORCH_CHECK(x.device().is_cuda(), #x " must be a CUDA tensor")
#define CHECK_CONTIGUOUS(x) TORCH_CHECK(x.is_contiguous(), #x " must be contiguous")
#define CHECK_INPUT(x) CHECK_CUDA(x); CHECK_CONTIGUOUS(x)
#define THREADS_PER_BLOCK 512


/**********************************************************

             -=-=-=-=- Main Function -=-=-=-=-=-

***********************************************************/

// // fill_missing(filled_spix_ptr, centers_ptr, missing_ptr, border,
//              nbatch, width, height, nspix, nmissing, break_iter);

__host__
void fill_missing(int* seg,  float* centers, int* missing, bool* border,
                  int nbatch, int width, int height,
                  int nspix, int nmissing, int break_iter){

    // -- init launch info --
    int npix = height*width;
    int num_block_sub = ceil( double(npix) / double(THREADS_PER_BLOCK) ); 
    dim3 BlockPerGridSub(num_block_sub,nbatch);
    int num_block = ceil( double(nmissing) / double(THREADS_PER_BLOCK) ); 
    dim3 BlockPerGrid(num_block,nbatch);
    dim3 ThreadPerBlock(THREADS_PER_BLOCK,1);

    // -- init border --
    cudaMemset(border, 0, nbatch*npix*sizeof(bool));
    const int sizeofint = sizeof(int);
    int iter = 0;

    // -- init num neg --
    int* num_neg_gpu = (int*)easy_allocate(1, sizeof(int));
    int prev_neg = 1;
    int num_neg_cpu = 1;

    while (num_neg_cpu > 0){

      // -- early break [ for viz ] --
      if ((break_iter>0) and (iter >= break_iter)){ break; }

      //  -- find border pixels --
      cudaMemset(num_neg_gpu, 0, sizeof(int));
      cudaMemset(border, 0, nbatch*npix*sizeof(bool));
      find_border_along_missing<<<BlockPerGrid,ThreadPerBlock>>>\
        (seg, missing, border, nmissing, nbatch, width, height, num_neg_gpu);
      cudaMemcpy(&num_neg_cpu,num_neg_gpu,sizeof(int),cudaMemcpyDeviceToHost);

      //  -- update segmentation --
      for (int xmod3 = 0 ; xmod3 < 2; xmod3++){
        for (int ymod3 = 0; ymod3 < 2; ymod3++){
          update_missing_seg_nn<<<BlockPerGridSub,ThreadPerBlock>>>(\
            seg, centers, border, nbatch, width, height, npix, xmod3, ymod3);
        }
      }

      // -- update previous --
      iter++;
      if ((iter>0) and (num_neg_cpu == prev_neg)){
        auto msg = "An error of some type, the border won't shrink: %d\n";
        fprintf(stdout,msg,num_neg_cpu);
        break;
      }
      prev_neg = num_neg_cpu;

    }

    //  -- find border pixels --
    cudaMemset(num_neg_gpu, 0, sizeof(int));
    cudaMemset(border, 0, nbatch*npix*sizeof(bool));
    find_border_along_missing<<<BlockPerGrid,ThreadPerBlock>>>            \
      (seg, missing, border, nmissing, nbatch, width, height, num_neg_gpu);
    // gpuErrchk( cudaPeekAtLastError() );
    // gpuErrchk( cudaDeviceSynchronize() );
    cudaMemcpy(&num_neg_cpu,num_neg_gpu,sizeof(int),cudaMemcpyDeviceToHost);
    if (num_neg_cpu > 0){
      fprintf(stdout,"negative spix exist.\n");
    }
    // fprintf(stdout,"d\n");
    // cudaDeviceSynchronize();

    // -- free memory --
    cudaFree(num_neg_gpu);

}


/**********************************************************

             -=-=-=-=- Helper Functions -=-=-=-=-=-

***********************************************************/

__device__ inline
float2 isotropic_space(float2 res, int label, int x, int y,
                       float* center_prop, int height, int width){
  // float sim = -100;
  float dx = (1.0f*x - center_prop[0])/(1.0f*width);
  float dy = (1.0f*y - center_prop[1])/(1.0f*height);
  float sim = -dx*dx - dy*dy;
  if (sim > res.x){
    res.x = sim;
    res.y = label;
  }
  return res;
}

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
    res_max.y = -1;

    // --> north, south, east, west <--
    int N = -1, S = -1, E = -1, W = -1;
    if (x>0){ W = __ldg(&seg[idx-1]); } // left
    if (y>0){ N = __ldg(&seg[idx-width]); }// top
    if (x<(width-1)){ E = __ldg(&seg[idx+1]); } // right
    if (y<(height-1)){ S = __ldg(&seg[idx+width]); } // below

    // --> diags [north (east, west), south (east, west)] <--
    // int NE = -1, NW = -1, SE = -1, SW = -1;

    // // -- read labels of neighbors --
    // if ((y>0) and (x<(width-1))){ NE = __ldg(&seg[idx-width+1]); } // top-right
    // if ((y>0) and (x>0)){  NW = __ldg(&seg[idx-width-1]); } // top-left
    // if ((x<(width-1)) and (y<(height-1))){SE = __ldg(&seg[idx+width+1]); } // btm-right
    // if ((x>0) and (y<(height-1))){ SW = __ldg(&seg[idx+width-1]); } // btm-left

    // -- read neighor labels for potts term --
    // check 8 nbrs and save result if valid to change to the last place of array
    // return how many nbrs different for potts term calculation

    //N :
    // set_nbrs(NW, N, NE, W, E, SW, S, SE,N, nbrs);
    // count_diff_nbrs_N = ischangbale_by_nbrs(nbrs);
    // // isNvalid = nbrs[8] or (res_max.y == -1);
    // // if(!isNvalid) return;
    
    // //E:
    // set_nbrs(NW, N, NE, W, E, SW, S, SE,E, nbrs);
    // count_diff_nbrs_E = ischangbale_by_nbrs(nbrs);
    // // isEvalid = nbrs[8] or (res_max.y == -1);
    // // if(!isEvalid) return;

    // //S :
    // set_nbrs(NW, N, NE, W, E, SW, S, SE,S, nbrs);
    // count_diff_nbrs_S = ischangbale_by_nbrs(nbrs);
    // // isSvalid = nbrs[8] or (res_max.y == -1);
    // // if(!isSvalid) return;

    //W :
    // set_nbrs(NW, N, NE, W, E, SW, S, SE,W, nbrs);
    // count_diff_nbrs_W = ischangbale_by_nbrs(nbrs);
    // isWvalid = nbrs[8] or (res_max.y == -1);
    // if(!isWvalid) return;

    // -- compute posterior --
    bool valid = N >= 0;
    label_check = N;
    if (valid){
      res_max = isotropic_space(res_max, label_check, x, y,
                                centers+label_check*2, height, width);
    }

    valid = S>=0;
    label_check = S;
    if(valid && (label_check!=N)){
      res_max = isotropic_space(res_max, label_check, x, y,
                                centers+label_check*2, height, width);
    }

    valid = W >= 0;
    label_check = W;
    if(valid && (label_check!=S)&&(label_check!=N)) {
      res_max = isotropic_space(res_max, label_check, x, y,
                                centers+label_check*2, height, width);
    }
    
    valid = E >= 0;
    label_check = E;
    if(valid && (label_check!=W)&&(label_check!=S)&&(label_check!=N)){
      res_max = isotropic_space(res_max, label_check, x, y,
                                centers+label_check*2, height,width);
    }

    seg[seg_idx] = res_max.y;
    return;
}



__global__
void find_border_along_missing(const int* seg, const int* missing,
                               bool* border, const int nmissing,
                               const int nbatch, const int width,
                               const int height, int* num_neg){   

    // --> cuda indices <--
    int _idx = threadIdx.x + blockIdx.x * blockDim.x;  
    if (_idx>=nmissing) return; 

    // --> space coordinates <--
    int idx = missing[_idx];
    int x = idx % width;
    int y = idx / width;

    // --> north, south, east, west <--
    int N = -1, S = -1, E = -1, W = -1, C = -1;

    // --> check out of bounds <--
    C = __ldg(&seg[idx]); // self
    if (y>0){ N = __ldg(&seg[idx-width]); } // above
    if (x>0){ W = __ldg(&seg[idx-1]); } // left
    if (y<(height-1)){ S = __ldg(&seg[idx+width]); } // below
    if (x<(width-1)){ E = __ldg(&seg[idx+1]); } // right
   
    // if the center is "-1" and any neighbor is valid, it is an edge
    bool valid = (N >= 0) or (W >= 0) or (S >= 0) or (E >= 0);
    // bool invalid = (N < 0) or (W < 0) or (S < 0) or (E < 0) or (C < 0);
    if (valid and (C<0)){ border[idx]=1; }
    // filled[_idx]=1;
    if (C<0){ atomicAdd(num_neg,1); }
    // border[idx]=1;
    return;        
}


/**********************************************************

             -=-=-=-=- Python API  -=-=-=-=-=-

***********************************************************/

torch::Tensor run_fill_missing(const torch::Tensor spix,
                               const torch::Tensor centers,
                               const torch::Tensor missing,
                               int nspix, int break_iter){

    // -- check --
    CHECK_INPUT(spix);
    CHECK_INPUT(centers);
    CHECK_INPUT(missing);

    // -- unpack --
    int nbatch = spix.size(0);
    int height = spix.size(1);
    int width = spix.size(2);
    int npix = height*width;
    int nmissing = missing.size(1);

    // -- allocate filled spix --
    auto options_i32 = torch::TensorOptions().dtype(torch::kInt32)
      .layout(torch::kStrided).device(spix.device());
    torch::Tensor filled_spix = spix.clone();
    int* filled_spix_ptr = filled_spix.data<int>();
    assert(nbatch==1);

    // -- allocate border --
    bool* border = (bool*)easy_allocate(nbatch*npix,sizeof(int));

    // -- run fill --
    float* centers_ptr = centers.data<float>();
    int* missing_ptr = missing.data<int>();
    if (nmissing>0){
      fill_missing(filled_spix_ptr, centers_ptr, missing_ptr, border,
                   nbatch, width, height, nspix, nmissing, break_iter);
    }
    cudaFree(border);

    return filled_spix;
}

void init_fill_missing(py::module &m){
  m.def("fill_missing", &run_fill_missing,"fill missing labels");
}

