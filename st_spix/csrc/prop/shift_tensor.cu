
/*********************************************************************************

     - This code shifts another tensor according to the "spix" and "flow"
     - ... this likely could have been done in native pytorch but this is
       much faster and we don't need grad...

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
void shift_tensor_kernel(int* spix, int* flow, 
                         float* in_tensor, float* out_tensor,
                         int* counts, bool* contributors, int npix, int nspix,
                         int nbatch, int height, int width, int nftrs){

  // -- get pixel index --
  int pix_idx = threadIdx.x + blockIdx.x*blockDim.x;
  int h_idx = pix_idx / width;
  int w_idx = pix_idx % width;
  if (pix_idx>=npix) return;
  int batch_idx = blockIdx.y;
  pix_idx = pix_idx + npix*batch_idx;
  
  // -- superpixel at sources --
  int spix_label = *(spix+pix_idx);
  if ((spix_label<0) or (spix_label>=nspix)){ return; }

  // -- flow at source --
  int flow_offset = 2*(nspix*batch_idx+spix_label);
  int flow_w = *(flow+flow_offset);
  int flow_h = *(flow+flow_offset+1);

  // -- dest index --
  int h_dest = h_idx+flow_h;
  int w_dest = w_idx+flow_w;

  // -- check boundary; skip if oob --
  bool valid_h = (0<=h_dest) and (h_dest<=(height-1));
  bool valid_w = (0<=w_dest) and (w_dest<=(width-1));
  bool valid = valid_h and valid_w;
  if (not valid){ return; }

  // -- write to destination --
  int dest_idx = h_dest * width + w_dest + npix*batch_idx;
  float* in_tensor_ptr = in_tensor+nftrs*pix_idx;
  float* out_tensor_ptr = out_tensor+nftrs*dest_idx;
  int* counts_ptr = counts+dest_idx;

  // -- atomic write --
  int prev_max = atomicAdd(counts_ptr,1);
  for (int fi = 0; fi < nftrs; fi++){
    atomicAdd(out_tensor_ptr+fi,*(in_tensor_ptr+fi));
  }

  // -- mark superpixel as contributing [for ordering] --
  bool* contrib_ptr = contributors+nspix*(dest_idx)+spix_label;
  *contrib_ptr = true;

}



/**********************************************************

             -=-=-=-=- Python API  -=-=-=-=-=-

***********************************************************/

std::tuple<torch::Tensor,torch::Tensor,torch::Tensor>
run_shift_tensor(const torch::Tensor in_tensor,
                 const torch::Tensor spix,
                 const torch::Tensor flow){

    // -- check --
    CHECK_INPUT(in_tensor);
    CHECK_INPUT(spix);
    CHECK_INPUT(flow);

    // -- unpack --
    int nbatch = spix.size(0);
    int height = spix.size(1);
    int width = spix.size(2);
    int nftrs = in_tensor.size(3);
    int npix = height*width;
    int nspix = spix.max().item<int>()+1;
    int nspix_ = flow.size(1);
    assert(nspix == nspix_);

    // -- allocate filled spix --
    auto options_b = torch::TensorOptions().dtype(torch::kBool)
      .layout(torch::kStrided).device(spix.device());
    auto options_i32 = torch::TensorOptions().dtype(torch::kInt32)
      .layout(torch::kStrided).device(spix.device());
    auto options_f32 = torch::TensorOptions().dtype(torch::kFloat32)
      .layout(torch::kStrided).device(spix.device());

    // -- allocate memory --
    torch::Tensor out_tensor = torch::zeros({nbatch,height,width,nftrs},options_f32);
    torch::Tensor counts = torch::zeros({nbatch,height,width},options_i32);
    torch::Tensor contributors = torch::zeros({nbatch,height,width,nspix},options_b);

    // -- init launch info --
    int nblocks_for_npix = ceil( double(npix) / double(THREADS_PER_BLOCK) ); 
    dim3 BlocksPixels(nblocks_for_npix,nbatch);
    dim3 NumThreads(THREADS_PER_BLOCK,1);

    // -- run kernel --
    shift_tensor_kernel<<<BlocksPixels,NumThreads>>>(spix.data<int>(),flow.data<int>(),
                                                     in_tensor.data<float>(),
                                                     out_tensor.data<float>(),
                                                     counts.data<int>(),
                                                     contributors.data<bool>(),
                                                     npix,nspix,nbatch,
                                                     height,width,nftrs);

    return std::make_tuple(out_tensor,counts,contributors);
}

void init_shift_tensor(py::module &m){
  m.def("shift_tensor", &run_shift_tensor,"shift tensor");
}


