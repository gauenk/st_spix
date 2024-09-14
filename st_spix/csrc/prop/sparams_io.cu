
#include <stdio.h>
#include <math.h>
#include <cuda.h>
#include <cuda_runtime.h>
#include <cmath>

#ifndef MY_SP_STRUCT
#define MY_SP_STRUCT
#include "../bass/share/my_sp_struct.h"
#endif
#define THREADS_PER_BLOCK 512
#include "sparams_io.h"
#include "init_utils.h"

__host__
PySuperpixelParams get_params_as_tensors(superpixel_params* sp_params,
                                         int* ids, int num){

  // -- allocate helpers --
  torch::Device device(torch::kCUDA, 0); 
  auto options_i32 = torch::TensorOptions().dtype(torch::kInt32)
    .layout(torch::kStrided).device(device);
  auto options_f32 = torch::TensorOptions().dtype(torch::kFloat32)
    .layout(torch::kStrided).device(device);

  // -- allocate superpixel params --
  PySuperpixelParams sp_params_py;
  sp_params_py.mu_i = torch::zeros({num,3},options_f32);
  sp_params_py.mu_s = torch::zeros({num,2},options_f32);
  sp_params_py.sigma_s = torch::zeros({num,3},options_f32);
  sp_params_py.logdet_Sigma_s = torch::zeros({num},options_f32);
  sp_params_py.counts = torch::zeros({num},options_i32);
  sp_params_py.prior_counts = torch::zeros({num},options_i32);
  sp_params_py.ids = torch::from_blob(ids,{num},options_i32);

  // -- fill the tensors with sp_params --
  params_to_tensors(sp_params_py,sp_params,num);
  return sp_params_py;
  
}

__host__
superpixel_params* get_tensors_as_params(PySuperpixelParams params,
                                         int sp_size, int npix, int nspix,
                                         int nspix_buffer){

  // -- allocate helpers --
  torch::Device device(torch::kCUDA, 0); 
  auto options_i32 = torch::TensorOptions().dtype(torch::kInt32)
    .layout(torch::kStrided).device(device);
  auto options_f32 = torch::TensorOptions().dtype(torch::kFloat32)
    .layout(torch::kStrided).device(device);


  // -- allocate superpixel params --
  const int sparam_size = sizeof(superpixel_params);
  superpixel_params* sp_params=(superpixel_params*)easy_allocate(nspix_buffer,
                                                                 sparam_size);
  init_sp_params(sp_params,sp_size,nspix,nspix_buffer,npix);

  // -- check legal accessing --
  int num = params.ids.size(0);
  assert(num <= nspix_buffer); // buffer must be larger (and very probably is)

  // -- fill the tensors with sp_params --
  tensors_to_params(params,sp_params);
  return sp_params;
  
}



__host__
void params_to_tensors(PySuperpixelParams sp_params_py,
                       superpixel_params* sp_params, int num){
  
  // -- unpack python pointers --
  auto mu_i = sp_params_py.mu_i.data<float>();
  auto mu_s = sp_params_py.mu_s.data<float>();
  auto sigma_s = sp_params_py.sigma_s.data<float>();
  auto logdet_Sigma_s = sp_params_py.logdet_Sigma_s.data<float>();
  auto counts = sp_params_py.counts.data<int>();
  auto prior_counts = sp_params_py.prior_counts.data<int>();
  auto ids = sp_params_py.ids.data<int>();

  // -- read from [sp_params] into [mu_i,mu_s,...] --
  int num_blocks = ceil( double(num) / double(THREADS_PER_BLOCK) ); 
  dim3 nblocks(num_blocks);
  dim3 nthreads(THREADS_PER_BLOCK);
  read_params<<<nblocks,nthreads>>>(mu_i, mu_s, sigma_s, logdet_Sigma_s,
              counts, prior_counts, ids, sp_params, num);
  
}

__global__
void read_params(float* mu_i, float* mu_s, float* cov, float* logdet_Sigma_s,
                 int* counts, int* prior_counts, int* ids,
                 superpixel_params* sp_params, int nspix){

    // -- filling superpixel params into image --
    int ix = threadIdx.x + blockIdx.x * blockDim.x;  
    if (ix>=nspix) return; 

    // -- offset memory access --
    float* mu_i_ix = mu_i + ix * 3;
    float* mu_s_ix = mu_s + ix * 2;
    float* cov_ix = cov + ix * 3;
    float* logsigma_ix = logdet_Sigma_s + ix;
    int* counts_ix = counts + ix;
    int* prior_counts_ix = prior_counts + ix;

    // -- read spix --
    int sp_index = ids[ix];
    if (sp_index < 0){ return; }
    auto params_ix = sp_params[sp_index];
      
    // -- fill params --
    cov_ix[0] = params_ix.sigma_s.x;
    cov_ix[1] = params_ix.sigma_s.y;
    cov_ix[2] = params_ix.sigma_s.z;

    logsigma_ix[0] = params_ix.logdet_Sigma_s;

    mu_i_ix[0] = params_ix.mu_i.x;
    mu_i_ix[1] = params_ix.mu_i.y;
    mu_i_ix[2] = params_ix.mu_i.z;

    mu_s_ix[0] = params_ix.mu_s.x;
    mu_s_ix[1] = params_ix.mu_s.y;

    counts_ix[0] = params_ix.count;
    prior_counts_ix[0] = params_ix.prior_count;
}

__host__
void tensors_to_params(PySuperpixelParams sp_params_py,
                       superpixel_params* sp_params){
  
  // -- unpack python pointers --
  auto mu_i = sp_params_py.mu_i.data<float>();
  auto mu_s = sp_params_py.mu_s.data<float>();
  auto sigma_s = sp_params_py.sigma_s.data<float>();
  auto logdet_Sigma_s = sp_params_py.logdet_Sigma_s.data<float>();
  auto counts = sp_params_py.counts.data<int>();
  auto prior_counts = sp_params_py.prior_counts.data<int>();
  auto ids = sp_params_py.ids.data<int>();
  int num = sp_params_py.ids.size(0);

  // -- write from [mu_i,mu_s,...] to [sp_params] --
  int num_blocks = ceil( double(num) / double(THREADS_PER_BLOCK) ); 
  dim3 nblocks(num_blocks);
  dim3 nthreads(THREADS_PER_BLOCK);
  write_params<<<nblocks,nthreads>>>(mu_i, mu_s, sigma_s, logdet_Sigma_s,
              counts, prior_counts, ids, sp_params, num);
  
}



__global__
void write_params(float* mu_i, float* mu_s, float* cov, float* logdet_Sigma_s,
                 int* counts, int* prior_counts, int* ids,
                 superpixel_params* sp_params, int nspix){

    // -- filling superpixel params into image --
    int ix = threadIdx.x + blockIdx.x * blockDim.x;  
    if (ix>=nspix) return; 

    // -- offset memory access --
    float* mu_i_ix = mu_i + ix * 3;
    float* mu_s_ix = mu_s + ix * 2;
    float* cov_ix = cov + ix * 3;
    float* logsigma_ix = logdet_Sigma_s + ix;
    int* counts_ix = counts + ix;
    int* prior_counts_ix = prior_counts + ix;

    // -- read spix --
    int sp_index = ids[ix];
    if (sp_index < 0){ return; }
    auto params_ix = sp_params[sp_index];
      

    // -- fill params --
    double3 sigma_s_tmp;
    sigma_s_tmp.x = cov_ix[0];
    sigma_s_tmp.y = cov_ix[1];
    sigma_s_tmp.z = cov_ix[2];
    params_ix.sigma_s = sigma_s_tmp;
    params_ix.logdet_Sigma_s = logsigma_ix[0];

    float3 mu_i_tmp;
    mu_i_tmp.x = mu_i_ix[0];
    mu_i_tmp.y = mu_i_ix[1];
    mu_i_tmp.z = mu_i_ix[2];
    params_ix.mu_i = mu_i_tmp;
    
    double2 mu_s_tmp;
    mu_s_tmp.x = mu_s_ix[0];
    mu_s_tmp.y = mu_s_ix[1];
    params_ix.mu_s = mu_s_tmp;

    params_ix.count = counts_ix[0];
    params_ix.prior_count = prior_counts_ix[0];

}



/***************************************************
----------------------------------------------------


               Extra Info Here


----------------------------------------------------
***************************************************/


__global__
void copy_spix_to_params_parents(float* means, float* cov,
                                 int* counts, int* spix_parents, 
                                 superpixel_params* sp_params, int* ids, int nspix){

    // -- filling superpixel params into image --
    int ix = threadIdx.x + blockIdx.x * blockDim.x;  
    if (ix>=nspix) return; 

    // -- offset memory access --
    float* means_ix = means + ix * 5;
    float* cov_ix = cov + ix * 4;
    int* counts_ix = counts + ix;
    int* spix_ix = spix_parents + ix;

    // -- read spix --
    int sp_index = ids[ix];
    if (sp_index < 0){ return; }
    auto params_ix = sp_params[sp_index];
      
    // -- fill params --
    cov_ix[0] = params_ix.sigma_s.x;
    cov_ix[1] = params_ix.sigma_s.y;
    cov_ix[2] = params_ix.sigma_s.z;
    cov_ix[3] = params_ix.logdet_Sigma_s;
    means_ix[0] = params_ix.mu_i.x;
    means_ix[1] = params_ix.mu_i.y;
    means_ix[2] = params_ix.mu_i.z;
    means_ix[3] = params_ix.mu_s.x;
    means_ix[4] = params_ix.mu_s.y;
    counts_ix[0] = params_ix.count;
    // spix_ix[0] = params_ix.parent_spix;
}


__global__
void copy_spix_to_params_icov2cov(float* means, float* cov, int* counts,
                                  superpixel_params* sp_params, int* ids, int nspix){

    // -- filling superpixel params into image --
    int ix = threadIdx.x + blockIdx.x * blockDim.x;  
    if (ix>=nspix) return; 

    // -- offset memory access --
    float* means_ix = means + ix * 5;
    float* cov_ix = cov + ix * 4;
    int* counts_ix = counts + ix;

    // -- read spix --
    int sp_index = ids[ix];
    if (sp_index < 0){ return; }
    auto params_ix = sp_params[sp_index];
      
    // -- fill params --
    cov_ix[0] = params_ix.sigma_s.x;
    cov_ix[1] = params_ix.sigma_s.y;
    cov_ix[2] = params_ix.sigma_s.z;
    cov_ix[3] = params_ix.logdet_Sigma_s;
    means_ix[0] = params_ix.mu_i.x;
    means_ix[1] = params_ix.mu_i.y;
    means_ix[2] = params_ix.mu_i.z;
    means_ix[3] = params_ix.mu_s.x;
    means_ix[4] = params_ix.mu_s.y;
    counts_ix[0] = params_ix.count;

    // -- invert cov --
    // .x => sx   .z => sy
    // .y => rho * sx * sy
    double inv_detC = exp(cov_ix[3]);
    cov_ix[0] = inv_detC * cov[2];
    cov_ix[1] = -inv_detC * cov[1];
    cov_ix[2] = inv_detC * cov[0];

}



/*********************************************************

        Allocate Memory (Vid,Seg) -> (Params)

/*********************************************************/

// superpixels_params* allocate_superpixels(int nspix_buffer){
//   const int sofsparams = sizeof(superpixel_params);
//   superpixel_params* params;
//   try {
//     throw_on_cuda_error(cudaMalloc((void**)&sp_params, nspix_buffer * sofsparams));
//   }
//   catch (thrust::system_error& e) {
//     std::cerr << "CUDA error after cudaMalloc: " << e.what() << std::endl;
//     cudaSetDevice(0);
//   }
//   return params;
// }

// superpixels_GPU_helper_sm* allocate_helper(int nspix_buffer){
//   const int sofsphelper_sm = sizeof(superpixel_GPU_helper_sm);
//   superpixel_GPU_helper_sm* helper;
//   try {
//     throw_on_cuda_error(cudaMalloc((void**)&helper,nspix_buffer*sofsphelper_sm));
//   }
//   catch (thrust::system_error& e) {
//     std::cerr << "CUDA error after cudaMalloc: " << e.what() << std::endl;
//     cudaSetDevice(0);
//   }
//   return helper;
// }

// __host__ void update_param(const float* image_gpu_double, const int* seg_gpu,
//                            superpixel_params* sp_params,
//                            superpixel_GPU_helper* sp_gpu_helper,
//                            const int nPixels, const int nSps,
//                            const int nSps_buffer, const int nbatch,
//                            const int xdim, const int ydim, const int nftrs,
//                            const int prior_sigma_s, const int prior_count){

// // -- helper params --
// superpixels_params* allocate_superpixels(int nspix_buffer){



// const int sofsparams = sizeof(superpixel_params);
//     const int sofsphelper = sizeof(superpixel_GPU_helper);
//     const int sofsphelper_sm = sizeof(superpixel_GPU_helper_sm);
//     const int sofpost_changes = sizeof(post_changes_helper);

//     sp_params_cpu = (superpixel_params*)malloc(nSPs_buffer * sofsparams);

//     try {
//         throw_on_cuda_error(cudaMalloc((void**)&sp_params, nSPs_buffer * sofsparams));
//         throw_on_cuda_error(cudaMalloc((void**)&sp_gpu_helper,nSPs_buffer*sofsphelper));
//         throw_on_cuda_error(cudaMalloc((void**)&sp_gpu_helper_sm,\
//                                        nSPs_buffer*sofsphelper_sm));
//         throw_on_cuda_error(cudaMalloc((void**)&post_changes, nPixels * sofpost_changes));
//     }
//     catch (thrust::system_error& e) {
//         std::cerr << "CUDA error after cudaMalloc: " << e.what() << std::endl;
//         cudaSetDevice(0);
//     }

