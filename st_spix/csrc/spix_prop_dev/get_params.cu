

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

// -- local import --
#ifndef MY_PROP_SP_STRUCT
#define MY_PROP_SP_STRUCT
#include "../bass/share/refine.h"
#include "../bass/core/Superpixels.h"
// #include "../bass/share/my_sp_struct.h"
#endif
// #include "../bass/core/Superpixels.h"
#include "init_prop_seg.h"

// -- define --
#define CHECK_CUDA(x) TORCH_CHECK(x.device().is_cuda(), #x " must be a CUDA tensor")
#define CHECK_CONTIGUOUS(x) TORCH_CHECK(x.is_contiguous(), #x " must be contiguous")
#define CHECK_INPUT(x) CHECK_CUDA(x); CHECK_CONTIGUOUS(x)
#define THREADS_PER_BLOCK 512
// void throw_on_cuda_error(cudaError_t code)
// {
//   if(code != cudaSuccess){
//     throw thrust::system_error(code, thrust::cuda_category());
//   }
// }



__global__
void get_copy_spix_params(float* means, float* cov, int* counts,
                          superpixel_params* sp_params, int* ids, int K){

    // -- filling superpixel params into image --
    int ix = threadIdx.x + blockIdx.x * blockDim.x;  
    if (ix>=K) return; 

    // -- offset memory access --
    float* cov_ix = cov + ix * 4;
    float* means_ix = means + ix * 5;
    int* counts_ix = counts + ix;

    // -- read spix --
    int sp_index = ids[ix];
    auto params_ix = sp_params[sp_index];
      
    // -- fill params --
    cov_ix[0]  = params_ix.sigma_s.x;
    cov_ix[1]  = params_ix.sigma_s.y;
    cov_ix[2]  = params_ix.sigma_s.y;
    cov_ix[3]  = params_ix.sigma_s.z;
    // cov_ix[3]  = params_ix.logdet_Sigma_s;
    means_ix[0] = params_ix.mu_i.x;
    means_ix[1] = params_ix.mu_i.y;
    means_ix[2] = params_ix.mu_i.z;
    means_ix[3] = params_ix.mu_s.x;
    means_ix[4] = params_ix.mu_s.y;
    counts_ix[0] = params_ix.count;
}


std::tuple<torch::Tensor,torch::Tensor,torch::Tensor,torch::Tensor>
get_params_forward(const torch::Tensor imgs,
                   const torch::Tensor in_spix,
                   int nPixels_in_square_side, float i_std,
                   float alpha, float beta, int in_K){
  
    // -- check --
    CHECK_INPUT(imgs);
    CHECK_INPUT(in_spix);

    // -- unpack --
    int nbatch = imgs.size(0);
    int height = imgs.size(1);
    int width = imgs.size(2);
    int nftrs = imgs.size(3);
    int nPix = height*width;
    auto options_i32 =torch::TensorOptions().dtype(torch::kInt32)
      .layout(torch::kStrided).device(imgs.device());
    auto options_f32 =torch::TensorOptions().dtype(torch::kFloat32)
      .layout(torch::kStrided).device(imgs.device());
    auto options_b =torch::TensorOptions().dtype(torch::kBool)
      .layout(torch::kStrided).device(imgs.device());
    int* seg_gpu = in_spix.data<int>();

    // -- init superpixels --
    superpixel_options spoptions = get_sp_options(nPixels_in_square_side,
                                                  i_std, beta, alpha);
    Superpixels sp = Superpixels(nbatch, width, height, nftrs, spoptions, in_K, seg_gpu);

    // -- load single image --
    sp.load_gpu_img((float*)(imgs.data<uint8_t>()));

    // -- use current image to compute params, skipping invalid --
    sp.run_update_param(); // perhaps compute this explicitly?


    /*****************************************************

                    Copy Covariance 

    *****************************************************/

    // -- init covariance --
    auto unique_ids = std::get<0>(at::_unique(in_spix));
    int K = unique_ids.sizes()[0];
    torch::Tensor means = torch::zeros({nbatch, K, 5}, options_f32);
    torch::Tensor cov = torch::zeros({nbatch, K, 4}, options_f32);
    torch::Tensor counts = torch::zeros({nbatch, K}, options_i32);

    // -- dispatch info --
    int num_blocks0 = ceil( double(K) / double(THREADS_PER_BLOCK) ); 
    dim3 nthreads0(THREADS_PER_BLOCK);
    dim3 nblocks0(num_blocks0);

    // -- launch --
    get_copy_spix_params<<<nblocks0,nthreads0>>>(means.data<float>(),
                                                 cov.data<float>(),
                                                 counts.data<int>(),
                                                 sp.get_cuda_sp_params(),
                                                 unique_ids.data<int>(),K);

    return std::make_tuple(means,cov,counts,unique_ids);
}


void
get_params_backward(torch::Tensor d_imgs, const torch::Tensor d_means,
                    const torch::Tensor d_cov, const torch::Tensor in_spix, int in_K){

    // -- check --
    CHECK_INPUT(d_imgs);
    CHECK_INPUT(d_means);
    CHECK_INPUT(d_cov);
    CHECK_INPUT(in_spix);

    // -- unpack --
    int nbatch = d_imgs.size(0);
    int height = d_imgs.size(1);
    int width = d_imgs.size(2);
    int nftrs = d_imgs.size(3);
    int nPix = height*width;
    auto options_i32 =torch::TensorOptions().dtype(torch::kInt32)
      .layout(torch::kStrided).device(d_imgs.device());
    auto options_f32 =torch::TensorOptions().dtype(torch::kFloat32)
      .layout(torch::kStrided).device(d_imgs.device());
    auto options_b =torch::TensorOptions().dtype(torch::kBool)
      .layout(torch::kStrided).device(d_imgs.device());

}


void init_get_params(py::module &m){
  m.def("get_params_forward", &get_params_forward,
        "get the parameters forward");
  m.def("get_params_backward", &get_params_backward,
        "get the parameters forward");
}
