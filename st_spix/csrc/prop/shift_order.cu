
/*********************************************************************************

Given:
 1. a boolean map of which superpixels contribute to which destination pixels, this 
 2. a list of superpixel sizes
This script selects the smallest superpixel to be "in front" of any overlapping regions

     Usage:

      tensor.shape = B,H,W,F
      spix.shape = B,H,W
      flow.shape = B,N_sp,2

      shift,contrib = shift_tensor(tensor,spix,flow)
      select = shift_order(contrib,sizes) // this script
      shift = shift_tensor_ordered(tensor,spix,flow,select)

*********************************************************************************/


// -- cpp imports --
#include <stdio.h>
#include "pch.h"

// -- define --
#define CHECK_CUDA(x) TORCH_CHECK(x.device().is_cuda(), #x " must be a CUDA tensor")
#define CHECK_CONTIGUOUS(x) TORCH_CHECK(x.is_contiguous(), #x " must be contiguous")
#define CHECK_INPUT(x) CHECK_CUDA(x); CHECK_CONTIGUOUS(x)
#define THREADS_PER_BLOCK 512


/**********************************************************

             -=-=-=-=- Main Function -=-=-=-=-=-

***********************************************************/


__global__
void shift_order_kernel(bool* contributors, int* sizes,
                        int* selected, int npix, int nspix,
                        int nbatch, int height, int width){

  // -- get pixel index --
  int pix_idx = threadIdx.x + blockIdx.x*blockDim.x;
  if (pix_idx>=npix) return;
  int batch_idx = blockIdx.y;
  pix_idx = pix_idx + npix*batch_idx;
  int* sizes_ptr = sizes + nspix*batch_idx;
  bool* contrib_ptr = contributors + nspix*pix_idx;
  int* selected_ptr = selected+pix_idx;

  // -- loop over all superpixels --
  int curr_val = -1;
  int argmin_val = -1;
  int min_val = 1000000;
  bool update;
  for(int si=0; si<nspix; si++){
    bool contrib = *(contrib_ptr+si);
    update = contrib ? (curr_val < min_val) : false;
    if (not update){ continue; }
    curr_val = *(sizes_ptr+si);
    argmin_val = update ? si : argmin_val;
    min_val = update ? curr_val : min_val;
  }
  if (argmin_val < 0){ return; }
  *selected_ptr = argmin_val;

}



/**********************************************************

             -=-=-=-=- Python API  -=-=-=-=-=-

***********************************************************/

torch::Tensor
run_shift_order(const torch::Tensor contributors,
                const torch::Tensor sizes){

    // -- check --
    CHECK_INPUT(contributors);
    CHECK_INPUT(sizes);

    // -- unpack --
    int nbatch = contributors.size(0);
    int height = contributors.size(1);
    int width = contributors.size(2);
    int npix = height*width;
    int nspix = sizes.size(1);

    // -- allocate filled spix --
    auto device = contributors.device();
    auto options_b = torch::TensorOptions().dtype(torch::kBool)
      .layout(torch::kStrided).device(device);
    auto options_i32 = torch::TensorOptions().dtype(torch::kInt32)
      .layout(torch::kStrided).device(device);
    auto options_f32 = torch::TensorOptions().dtype(torch::kFloat32)
      .layout(torch::kStrided).device(device);

    // -- allocate memory --
    torch::Tensor selected = -torch::ones({nbatch,height,width},options_i32);

    // -- init launch info --
    int nblocks_for_npix = ceil( double(npix) / double(THREADS_PER_BLOCK) ); 
    dim3 BlocksPixels(nblocks_for_npix,nbatch);
    dim3 NumThreads(THREADS_PER_BLOCK,1);

    // -- run kernel --
    shift_order_kernel<<<BlocksPixels,NumThreads>>>(contributors.data<bool>(),
                                                    sizes.data<int>(),
                                                    selected.data<int>(),
                                                    npix,nspix,nbatch,
                                                    height,width);

    return selected;
}

void init_shift_order(py::module &m){
  m.def("shift_order", &run_shift_order,"shift order");
}
