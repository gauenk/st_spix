
/*********************************************************************************

     - This code shifts labels according to the input optical flow
     and marks overlapping regions as `invalid' using that
     conveinent 'atomicAdd' features which reads atomicly from memory.

     - this atomic read makes invaliding the overlapping regions _very_ fast
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
void shift_labels_kernel(int* spix, int* flow, 
                         int* shifted_spix, int* counts,
                         int npix, int nspix, int nbatch,
                         int height, int width){

  // -- get pixel index --
  int pix_idx = threadIdx.x + blockIdx.x*blockDim.x;
  int h_idx = pix_idx / width;
  int w_idx = pix_idx % width;
  if (pix_idx>=npix) return;
  int batch_idx = blockIdx.y;
  pix_idx = pix_idx + npix*batch_idx;
  
  // -- superpixel at sources --
  int spix_label = *(spix+pix_idx);

  // -- flow at source --
  int flow_offset = 2*(nspix*batch_idx+spix_label);
  int flow_w = *(flow+flow_offset);
  int flow_h = *(flow+flow_offset+1);

  // -- dest index --
  int h_dest = h_idx+flow_h;
  int w_dest = w_idx+flow_w;

  // -- check boundary; skip if oob --
  bool valid_h = (0<=h_dest) and (h_dest<height);
  bool valid_w = (0<=w_dest) and (w_dest<width);
  bool valid = valid_h and valid_w;
  if (not valid){ return; }

  // -- write to destination --
  int dest_idx = h_dest * width + w_dest + npix*batch_idx;
  int* shifted_ptr = shifted_spix+dest_idx;
  int* counts_ptr = counts+dest_idx;

  // -- atomic read from mem --
  // int prev_max = atomicMin(counts_ptr,prev_spix != -1);
  int prev_max = atomicAdd(counts_ptr,1);
  int prev_spix = atomicMax(shifted_ptr,spix_label);

}


/**********************************************************

             -=-=-=-=- Python API  -=-=-=-=-=-

***********************************************************/

// torch::Tensor
std::tuple<torch::Tensor,torch::Tensor>
run_shift_labels(const torch::Tensor spix,
                 const torch::Tensor flow){

    // -- check --
    CHECK_INPUT(spix);
    CHECK_INPUT(flow);

    // -- unpack --
    int nbatch = spix.size(0);
    int height = spix.size(1);
    int width = spix.size(2);
    int npix = height*width;
    // int nspix = spix.max().item<int>()+1;
    int nspix = flow.size(1);
    // assert(nspix == nspix_);

    // -- allocate filled spix --
    auto options_b = torch::TensorOptions().dtype(torch::kBool)
      .layout(torch::kStrided).device(spix.device());
    auto options_i32 = torch::TensorOptions().dtype(torch::kInt32)
      .layout(torch::kStrided).device(spix.device());

    // -- allocate memory --
    torch::Tensor shifted_spix = -torch::ones({nbatch,height,width},options_i32);
    torch::Tensor counts = torch::zeros({nbatch,height,width},options_i32);

    // -- init launch info --
    int nblocks_for_npix = ceil( double(npix) / double(THREADS_PER_BLOCK) ); 
    dim3 BlocksPixels(nblocks_for_npix,nbatch);
    dim3 NumThreads(THREADS_PER_BLOCK,1);

    // -- run kernel --
    shift_labels_kernel<<<BlocksPixels,NumThreads>>>(spix.data<int>(),
                                                     flow.data<int>(),
                                                     shifted_spix.data<int>(),
                                                     counts.data<int>(),
                                                     npix,nspix,nbatch,height,width);
    // -- fill invalid --
    // auto condition = counts != 1;
    // torch::Tensor inds = torch::where(condition);
    // shifted_spix.index_put_({inds[0],inds[1],inds[2]}, -1);
    // // shifted_spix.index_put_({inds}, -1);

    return std::make_tuple(shifted_spix,counts);
}

void init_shift_labels(py::module &m){
  m.def("shift_labels", &run_shift_labels,"shift labels");
}


