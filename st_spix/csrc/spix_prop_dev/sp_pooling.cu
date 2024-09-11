

#include <stdio.h>
#include <math.h>
#include <cuda.h>
#include <cuda_runtime.h>
#include <cmath>

#include <cuda/std/type_traits>
#include <torch/types.h>
#include <cuda.h>
#include <cuda_runtime.h>
#include <torch/extension.h>
#include <vector>

#include <thrust/host_vector.h>
#include <thrust/device_vector.h>
#include <thrust/copy.h>
#include <thrust/fill.h>
#include <thrust/reduce.h>

// -- project imports --
#include "sp_pooling.h"
#include "../bass/core/Superpixels.h"

// -- define --
#define CHECK_CUDA(x) TORCH_CHECK(x.device().is_cuda(), #x " must be a CUDA tensor")
#define CHECK_CONTIGUOUS(x) TORCH_CHECK(x.is_contiguous(), #x " must be contiguous")
#define CHECK_INPUT(x) CHECK_CUDA(x); CHECK_CONTIGUOUS(x)
#define THREADS_PER_BLOCK 512

/********************************************


                Forward


********************************************/
// __global__
// void run_sp_downcount(int* seg, float* downcount, const int npix){
  
//   // -- get pixel index --
//   int pix_idx = threadIdx.x + blockIdx.x*blockDim.x;
//   if (pix_idx>=npix) return;

//   // -- get segmentation index --
//   int seg_idx = seg[pix_idx];
//   if (seg_idx < 0){ return; }

//   // -- add to downsampled --
//   float* dsC = downcount + seg_idx;
//   atomicAdd(dsC,static_cast<float>(1));
// }


__global__
void run_sp_downsample(float* img, int* seg,
                       float* downsampled, float* downcount,
                       const int npix, const int nftrs){
  
  // -- get pixel index --
  int pix_idx = threadIdx.x + blockIdx.x*blockDim.x;
  if (pix_idx>=npix) return;

  // -- get segmentation index --
  int seg_idx = seg[pix_idx];
  if (seg_idx < 0){ return; }

  // -- add to downsampled --
  float* imgF = img + pix_idx * nftrs;
  float* dsF = downsampled + seg_idx * nftrs;
  float* dsC = downcount + seg_idx;
  for (int fidx = 0; fidx < nftrs; fidx++){
    atomicAdd(dsF+fidx,*(imgF+fidx));
  }
  atomicAdd(dsC,static_cast<float>(1));
}

__global__
void run_sp_pooling(float* pooled, int* seg, float* downsampled,
                    const int npix, const int nftrs){

  // -- get pixel index --
  int pix_idx = threadIdx.x + blockIdx.x*blockDim.x;
  if (pix_idx>=npix) return;

  // -- get segmentation index --
  int seg_idx = seg[pix_idx];
  if (seg_idx < 0){ return; }

  // -- write to pooled --
  float* dsF = downsampled + seg_idx * nftrs;
  // int dsC = *(downcount + seg_idx);
  float* poolF = pooled + pix_idx * nftrs;
  for (int fidx = 0; fidx < nftrs; fidx++){
    // *(poolF+fidx) = (*(dsF+fidx))/dsC;
    *(poolF+fidx) = *(dsF+fidx);
  }

}

std::tuple<torch::Tensor,torch::Tensor,torch::Tensor>
sp_pooling_fwd(const torch::Tensor img, const torch::Tensor seg, int nspix){


  // -- check --
  CHECK_INPUT(img);
  CHECK_INPUT(seg);

  // -- unpack --
  int nbatch = img.size(0);
  int height = img.size(1);
  int width = img.size(2);
  int nftrs = img.size(3);
  int npix = height*width;
  assert(nbatch == 1);


  // -- pointers --
  float* img_ptr = img.data<float>();
  int* seg_ptr = seg.data<int>();

  // -- alloc options --
  auto options_f32 = torch::TensorOptions().dtype(torch::kFloat32)
    .layout(torch::kStrided).device(img.device());
  auto options_i32 = torch::TensorOptions().dtype(torch::kFloat32)
    .layout(torch::kStrided).device(img.device());

  // -- init pooled --
  torch::Tensor pooled = torch::zeros({nbatch, height, width, nftrs}, options_f32);
  float* pooled_ptr = pooled.data<float>();

  // -- init downsampled & counts --
  torch::Tensor downsampled = torch::zeros({nbatch, nspix, nftrs}, options_f32);
  float* downsampled_ptr = downsampled.data<float>();
  torch::Tensor counts = torch::zeros({nbatch, nspix}, options_f32);
  float* counts_ptr = counts.data<float>();

  // -- launch pooling --
  int num_block = ceil( double(npix) / double(THREADS_PER_BLOCK) ); 
  dim3 BlockPerGrid(num_block);
  dim3 ThreadPerBlock(THREADS_PER_BLOCK,1);
  run_sp_downsample<<<BlockPerGrid,ThreadPerBlock>>>
    (img_ptr, seg_ptr, downsampled_ptr, counts_ptr, npix, nftrs);
  downsampled /= (counts.unsqueeze(2) + 1e-10); // normalize in-place
  run_sp_pooling<<<BlockPerGrid,ThreadPerBlock>>>
    (pooled_ptr, seg_ptr, downsampled_ptr, npix, nftrs);

  return std::make_tuple(pooled,downsampled,counts);
}


/********************************************


           Upscale Pooled Features


********************************************/

torch::Tensor
downsampled_to_pooled(const torch::Tensor downsampled,
                    const torch::Tensor seg, int nspix){

  // -- check --
  CHECK_INPUT(downsampled);
  CHECK_INPUT(seg);
  // assert(seg.max() <= nspix);

  // -- unpack --
  int nbatch = seg.size(0);
  int height = seg.size(1);
  int width = seg.size(2);
  int nftrs = downsampled.size(2);
  int npix = height*width;
  assert(nbatch == 1);

  // -- pointers --
  float* downsampled_ptr = downsampled.data<float>();
  int* seg_ptr = seg.data<int>();

  // -- alloc options --
  auto options_f32 = torch::TensorOptions().dtype(torch::kFloat32)
    .layout(torch::kStrided).device(seg.device());

  // -- init pooled --
  torch::Tensor pooled = torch::zeros({nbatch, height, width, nftrs}, options_f32);
  float* pooled_ptr = pooled.data<float>();

  // -- launch pooling --
  int num_block = ceil( double(npix) / double(THREADS_PER_BLOCK) ); 
  dim3 BlockPerGrid(num_block);
  dim3 ThreadPerBlock(THREADS_PER_BLOCK,1);
  run_sp_pooling<<<BlockPerGrid,ThreadPerBlock>>>
    (pooled_ptr, seg_ptr, downsampled_ptr, npix, nftrs);

  return pooled;
}

/********************************************


                Backward


********************************************/

// __global__
// void scatter_bwd(float* img_grad,
//                  float* pooled_grad,
//                  float* downsampled_grad,
//                  float* counts,  int* seg,
//                  const int npix, const int nftrs){
  
//   // -- get pixel index --
//   int pix_idx = threadIdx.x + blockIdx.x*blockDim.x;
//   if (pix_idx>=npix) return;

//   // -- get segmentation index --
//   int seg_idx = seg[pix_idx];
//   if (seg_idx < 0){ return; }

//   // -- add to downsampled --
//   float* imgF = img_grad + pix_idx * nftrs;
//   float* poolF = pooled_grad + pix_idx * nftrs;
//   float* dsF = downsampled_grad + seg_idx * nftrs;
//   float count = *(counts + seg_idx);
//   for (int fidx = 0; fidx < nftrs; fidx++){
//     atomicAdd(imgF+fidx,*(dsF+fidx)/count);
//     atomicAdd(imgF+fidx,*(poolF+fidx)/count);
//   }

// }


std::tuple<torch::Tensor>
sp_pooling_bwd(const torch::Tensor pooled_grad,
               const torch::Tensor downsampled_grad,
               const torch::Tensor counts,
               const torch::Tensor seg,
               int nbatch, int height, int width, int nftrs, int nspix){

  // -- check --
  CHECK_INPUT(pooled_grad);
  CHECK_INPUT(downsampled_grad);
  CHECK_INPUT(counts);
  CHECK_INPUT(seg);

  // -- unpack --
  int npix = height*width;
  auto device = seg.device();
  assert(nbatch == 1);

  // -- allocate --
  auto options_f32 = torch::TensorOptions().dtype(torch::kFloat32)
    .layout(torch::kStrided).device(device);
  torch::Tensor img_grad = torch::zeros({nbatch, height, width, nftrs}, options_f32);
  float* img_grad_ptr = img_grad.data<float>();

  // // -- init downsampled & counts --
  // torch::Tensor downsampled = torch::zeros({nbatch, nspix, nftrs}, options_f32);
  // float* downsampled_ptr = downsampled.data<float>();
  // torch::Tensor counts = torch::zeros({nbatch, nspix, 1}, options_f32);
  // float* counts_ptr = counts.data<float>();

  // // -- launch pooling --
  // int num_block = ceil( double(npix) / double(THREADS_PER_BLOCK) ); 
  // dim3 BlockPerGrid(num_block);
  // dim3 ThreadPerBlock(THREADS_PER_BLOCK,1);
  // scatter_bwd<<<BlockPerGrid,ThreadPerBlock>>>
  //   (img_ptr, seg_ptr, downsampled_ptr, counts_ptr, npix, nftrs);


  // downsampled /= (counts + 1e-10); // normalize in-place
  // run_sp_pooling<<<BlockPerGrid,ThreadPerBlock>>>
  //   (pooled_ptr, seg_ptr, downsampled_ptr, npix, nftrs);


  return std::make_tuple(img_grad);
}


void init_sp_pooling(py::module &m){
  m.def("sp_pooling_fwd", &sp_pooling_fwd,"superpixel pooling fwd");
  m.def("downsampled_to_pooled", &downsampled_to_pooled,
        "upscale from downsampled features");
  // m.def("sp_pooling_bwd", &sp_pooling_bwd,"superpixel pooling bwd");
}


