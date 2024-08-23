

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

__global__
void run_sp_downsample(float* img, int* seg,
                       float* down_sampled, float* down_count,
                       const int npix, const int nftrs){
  
  // -- get pixel index --
  int pix_idx = threadIdx.x + blockIdx.x*blockDim.x;
  if (pix_idx>=npix) return;

  // -- get segmentation index --
  int seg_idx = seg[pix_idx];
  if (seg_idx < 0){ return; }

  // -- add to down_sampled --
  float* imgF = img + pix_idx * nftrs;
  float* dsF = down_sampled + seg_idx * nftrs;
  float* dsC = down_count + seg_idx;
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
  // int dsC = *(down_count + seg_idx);
  float* poolF = pooled + pix_idx * nftrs;
  for (int fidx = 0; fidx < nftrs; fidx++){
    // *(poolF+fidx) = (*(dsF+fidx))/dsC;
    *(poolF+fidx) = *(dsF+fidx);
  }

}

std::tuple<torch::Tensor,torch::Tensor>
sp_pooling(const torch::Tensor img, const torch::Tensor seg, int nspix){


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
  torch::Tensor counts = torch::zeros({nbatch, nspix, 1}, options_f32);
  float* counts_ptr = counts.data<float>();

  // -- launch pooling --
  int num_block = ceil( double(npix) / double(THREADS_PER_BLOCK) ); 
  dim3 BlockPerGrid(num_block);
  dim3 ThreadPerBlock(THREADS_PER_BLOCK,1);
  run_sp_downsample<<<BlockPerGrid,ThreadPerBlock>>>
    (img_ptr, seg_ptr, downsampled_ptr, counts_ptr, npix, nftrs);
  downsampled /= (counts + 1e-10); // normalize in-place
  run_sp_pooling<<<BlockPerGrid,ThreadPerBlock>>>
    (pooled_ptr, seg_ptr, downsampled_ptr, npix, nftrs);

  return std::make_tuple(pooled,downsampled);
}




void init_sp_pooling(py::module &m){
  m.def("sp_pooling", &sp_pooling,"superpixel pooling");
}


