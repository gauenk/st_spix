
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
#include "share.cu"

// -- define --
#define CHECK_CUDA(x) TORCH_CHECK(x.device().is_cuda(), #x " must be a CUDA tensor")
#define CHECK_CONTIGUOUS(x) TORCH_CHECK(x.is_contiguous(), #x " must be contiguous")
#define CHECK_INPUT(x) CHECK_CUDA(x); CHECK_CONTIGUOUS(x)
#define THREADS_PER_BLOCK 512

__global__
void scatter_spix_forward_kernel(float* img, float* flow,
                                 float* scatter, float* counts,
                                 int height, int width, int npix, int nftrs){

    // -- rasterize --
    int cuda_ix = threadIdx.x + blockIdx.x * blockDim.x;  
    if (cuda_ix>=npix) return; 

    // -- read source location --
    int hi = cuda_ix/width;
    int wi = cuda_ix%width;
    float* img_ptr = img + cuda_ix * nftrs;
    float* flow_ptr = flow + cuda_ix * 2;

    // -- read flow --
    float dw = flow_ptr[0];
    float dh = flow_ptr[1];

    // -- (top,left) of dest location --
    float hi_f = hi + dh;
    float wi_f = wi + dw;

    // -- bilinear write --
    float eps = 1e-4;
    bilin2d_interpolate(img_ptr, scatter, counts,
                        hi_f, wi_f, height, width, nftrs, eps);

}

std::tuple<torch::Tensor,torch::Tensor>
scatter_spix_forward(const torch::Tensor spix,
                     const torch::Tensor flow){

    // -- check --
    CHECK_INPUT(spix);
    CHECK_INPUT(flow);

    // -- unpack --
    int nbatch = spix.size(0);
    int height = spix.size(1);
    int width = spix.size(2);
    int npix = height*width;
    int nftrs = 1;
    auto options_f32 = torch::TensorOptions()
      .dtype(torch::kFloat32).layout(torch::kStrided).device(spix.device());

    // -- init viz --
    torch::Tensor scatter_spix = torch::zeros({nbatch, height, width}, options_f32);
    torch::Tensor count_spix = torch::zeros({nbatch, height, width}, options_f32);

    // -- dispatch info --
    int num_blocks0 = ceil( double(npix) / double(THREADS_PER_BLOCK) ); 
    dim3 nthreads0(THREADS_PER_BLOCK);
    dim3 nblocks0(num_blocks0,nbatch);

    // -- launch --
    scatter_spix_forward_kernel<<<nblocks0,nthreads0>>>(spix.data<float>(),
                                                        flow.data<float>(),
                                                        scatter_spix.data<float>(),
                                                        count_spix.data<float>(),
                                                        height,width,npix,nftrs);

    // -- return --
    return std::make_tuple(scatter_spix,count_spix);
}


void init_scatter_spix(py::module &m){
  m.def("scatter_spix_forward", &scatter_spix_forward,"scatter_spix");
}
