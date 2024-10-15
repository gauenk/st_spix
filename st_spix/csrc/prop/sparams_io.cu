
// #include <stdio.h>
// #include <math.h>
// #include <cuda.h>
// #include <cuda_runtime.h>
// #include <cmath>

#ifndef MY_SP_STRUCT
#define MY_SP_STRUCT
#include "../bass/share/my_sp_struct.h"
#endif
#define THREADS_PER_BLOCK 512
#include "pch.h"
#include "init_utils.h"
#include "sparams_io.h"
#include "init_sparams.h"

/***********************************************************************

                       Full Superpixels

***********************************************************************/

__host__
PySuperpixelParams get_params_as_tensors(spix_params* sp_params,int* ids, int num){

  // -- allocate helpers --
  torch::Device device(torch::kCUDA, 0); 
  auto options_i32 = torch::TensorOptions().dtype(torch::kInt32)
    .layout(torch::kStrided).device(device);
  auto options_f32 = torch::TensorOptions().dtype(torch::kFloat32)
    .layout(torch::kStrided).device(device);

  // -- allocate superpixel params --
  PySuperpixelParams sp_params_py;
  // -- appearance --
  sp_params_py.mu_app = torch::zeros({num,3},options_f32);
  sp_params_py.sigma_app = torch::zeros({num,3},options_f32);
  sp_params_py.logdet_sigma_app = torch::zeros({num},options_f32);
  sp_params_py.prior_mu_app = torch::zeros({num,3},options_f32);
  sp_params_py.prior_sigma_app = torch::zeros({num,3},options_f32);
  sp_params_py.prior_mu_app_count = torch::zeros({num},options_i32);
  sp_params_py.prior_sigma_app_count = torch::zeros({num},options_i32);

  // -- shape --
  sp_params_py.mu_shape = torch::zeros({num,2},options_f32);
  sp_params_py.sigma_shape = torch::zeros({num,3},options_f32);
  sp_params_py.logdet_sigma_shape = torch::zeros({num},options_f32);
  sp_params_py.prior_mu_shape = torch::zeros({num,2},options_f32);
  sp_params_py.prior_sigma_shape = torch::zeros({num,3},options_f32);
  sp_params_py.prior_mu_shape_count = torch::zeros({num},options_i32);
  sp_params_py.prior_sigma_shape_count = torch::zeros({num},options_i32);

  // -- helpers --
  sp_params_py.counts = torch::zeros({num},options_i32);
  sp_params_py.prior_counts = torch::zeros({num},options_i32);
  sp_params_py.ids = torch::from_blob(ids,{num},options_i32);

  // -- fill the tensors with sp_params --
  params_to_tensors(sp_params_py,sp_params,num);
  return sp_params_py;
  
}

__host__ spix_params* get_tensors_as_params(PySuperpixelParams params,
                                            int sp_size, int npix, int nspix,
                                            int nspix_buffer){
  // -- allocate helpers --
  torch::Device device(torch::kCUDA, 0); 
  auto options_i32 = torch::TensorOptions().dtype(torch::kInt32)
    .layout(torch::kStrided).device(device);
  auto options_f32 = torch::TensorOptions().dtype(torch::kFloat32)
    .layout(torch::kStrided).device(device);

  // -- allocate superpixel params --
  const int sparam_size = sizeof(spix_params);
  spix_params* sp_params=(spix_params*)easy_allocate(nspix_buffer,sparam_size);
  // init_sp_params(sp_params,sp_size,nspix,nspix_buffer,npix);

  // -- check legal accessing --
  int num = params.ids.size(0);
  assert(num <= nspix_buffer); // buffer must be larger (and very probably is)

  // -- fill the tensors with sp_params --
  tensors_to_params(params,sp_params);
  return sp_params;
  
}

__host__ void params_to_tensors(PySuperpixelParams sp_params_py,
                                spix_params* sp_params, int num){
  
  /****************************************************

                    Unpack Pointers

  *****************************************************/

  // -- appearance --
  auto mu_app = sp_params_py.mu_app.data<float>();
  auto sigma_app = sp_params_py.sigma_app.data<float>();
  auto logdet_sigma_app = sp_params_py.logdet_sigma_app.data<float>();
  auto prior_mu_app = sp_params_py.prior_mu_app.data<float>();
  auto prior_sigma_app = sp_params_py.prior_sigma_app.data<float>();
  auto prior_mu_app_count = sp_params_py.prior_mu_app_count.data<int>();
  auto prior_sigma_app_count = sp_params_py.prior_sigma_app_count.data<int>();

  // -- shape --
  auto mu_shape = sp_params_py.mu_shape.data<float>();
  auto sigma_shape = sp_params_py.sigma_shape.data<float>();
  auto logdet_sigma_shape = sp_params_py.logdet_sigma_shape.data<float>();
  auto prior_mu_shape = sp_params_py.prior_mu_shape.data<float>();
  auto prior_sigma_shape = sp_params_py.prior_sigma_shape.data<float>();
  auto prior_mu_shape_count = sp_params_py.prior_mu_shape_count.data<int>();
  auto prior_sigma_shape_count = sp_params_py.prior_sigma_shape_count.data<int>();

  // -- misc --
  auto counts = sp_params_py.counts.data<int>();
  auto prior_counts = sp_params_py.prior_counts.data<int>();
  auto ids = sp_params_py.ids.data<int>();
  int max_num = sp_params_py.ids.size(0);
  assert(num <= max_num);

  // -- read from [sp_params] into [mu_app,mu_shape,...] --
  int num_blocks = ceil( double(num) / double(THREADS_PER_BLOCK) ); 
  dim3 nblocks(num_blocks);
  dim3 nthreads(THREADS_PER_BLOCK);
  read_params<<<nblocks,nthreads>>>(mu_app, sigma_app, logdet_sigma_app,
                                    prior_mu_app, prior_sigma_app,
                                    prior_mu_app_count, prior_sigma_app_count,
                                    mu_shape, sigma_shape, logdet_sigma_shape,
                                    prior_mu_shape, prior_sigma_shape,
                                    prior_mu_shape_count, prior_sigma_shape_count,
                                    counts, prior_counts, ids, sp_params, num);
  
}


__global__ 
void read_params(float* mu_app, float* sigma_app, float* logdet_sigma_app,
                 float* prior_mu_app, float* prior_sigma_app,
                 int* prior_mu_app_count, int* prior_sigma_app_count,
                 float* mu_shape, float* sigma_shape, float* logdet_sigma_shape,
                 float* prior_mu_shape, float* prior_sigma_shape,
                 int* prior_mu_shape_count, int* prior_sigma_shape_count,
                 int* counts, int* prior_counts, int* ids,
                 spix_params* sp_params, int spix){
    // -- filling superpixel params into image --
    int ix = threadIdx.x + blockIdx.x * blockDim.x;
    if (ix >= spix) return;  // 'nspix' should be 'spix', as passed by the caller

    // -- offest memory access [appearance] --
    float* mu_app_ix = mu_app + ix * 3;
    float* sigma_app_ix = sigma_app + ix * 3;  // Handle sigma_app
    float* logdet_sigma_app_ix = logdet_sigma_app + ix;  // Handle logdet_sigma_app
    float* prior_mu_app_ix = prior_mu_app + ix * 3;  // Handle prior_mu_app
    float* prior_sigma_app_ix = prior_sigma_app + ix * 3;  // Handle prior_sigma_app
    int* prior_mu_app_count_ix = prior_mu_app_count + ix; 
    int* prior_sigma_app_count_ix = prior_sigma_app_count + ix;

    // -- offest memory access [shape] --
    float* mu_shape_ix = mu_shape + ix * 2;
    float* sigma_shape_ix = sigma_shape + ix * 3;  // Handle sigma_shape
    float* logsigma_shape_ix = logdet_sigma_shape + ix;  // Already in the code
    float* prior_mu_shape_ix = prior_mu_shape + ix * 2;  // Handle prior_mu_shape
    float* prior_sigma_shape_ix = prior_sigma_shape + ix * 3;  // Handle prior_sigma_shape
    int* prior_mu_shape_count_ix = prior_mu_shape_count + ix; 
    int* prior_sigma_shape_count_ix = prior_sigma_shape_count + ix; 

    // -- misc --
    int* counts_ix = counts + ix;
    int* prior_counts_ix = prior_counts + ix;  
                                                                                          
    // -- read spix --
    int sp_index = ids[ix];
    if (sp_index < 0){ return; }
    auto params_ix = sp_params[sp_index];
    
    /*****************************************************

                    Fill the Params                             

    *****************************************************/

    // -- appearance [est] --
    mu_app_ix[0] = params_ix.mu_app.x;
    mu_app_ix[1] = params_ix.mu_app.y;
    mu_app_ix[2] = params_ix.mu_app.z;
    sigma_app_ix[0] = params_ix.sigma_app.x;  // Fill sigma_app 
    sigma_app_ix[1] = params_ix.sigma_app.y;
    sigma_app_ix[2] = params_ix.sigma_app.z;
    logdet_sigma_app_ix[0] = params_ix.logdet_sigma_app;  // Fill logdet_sigma_app
    // -- appearance [prior] --
    prior_mu_app_ix[0] = params_ix.prior_mu_app.x;  // Fill prior_mu_app
    prior_mu_app_ix[1] = params_ix.prior_mu_app.y;
    prior_mu_app_ix[2] = params_ix.prior_mu_app.z;
    prior_mu_app_count_ix[0] = params_ix.prior_mu_app_count;
    prior_sigma_app_ix[0] = params_ix.prior_sigma_app.x;  // Fill prior_sigma_app
    prior_sigma_app_ix[1] = params_ix.prior_sigma_app.y;
    prior_sigma_app_ix[2] = params_ix.prior_sigma_app.z;
    prior_sigma_app_count_ix[0] = params_ix.prior_sigma_app_count;

    // -- shape [est] --
    mu_shape_ix[0] = params_ix.mu_shape.x;
    mu_shape_ix[1] = params_ix.mu_shape.y;
    sigma_shape_ix[0] = params_ix.sigma_shape.x;  // Fill sigma_shape
    sigma_shape_ix[1] = params_ix.sigma_shape.y;
    sigma_shape_ix[2] = params_ix.sigma_shape.z;
    logsigma_shape_ix[0] = params_ix.logdet_sigma_shape;  // Already in the code
    // -- shape [prior] --
    prior_mu_shape_ix[0] = params_ix.prior_mu_shape.x;  // Fill prior_mu_shape
    prior_mu_shape_ix[1] = params_ix.prior_mu_shape.y;                                    
    prior_sigma_shape_ix[0] = params_ix.prior_sigma_shape.x;  // Fill prior_sigma_shape
    prior_sigma_shape_ix[1] = params_ix.prior_sigma_shape.y;
    prior_mu_shape_count_ix[0] = params_ix.prior_mu_shape_count;
    prior_sigma_shape_count_ix[0] = params_ix.prior_sigma_shape_count;

    // -- misc --
    counts_ix[0] = params_ix.count;
    prior_counts_ix[0] = params_ix.prior_count;


}

__host__ void tensors_to_params(PySuperpixelParams sp_params_py, spix_params* sp_params){
  
  // -- unpack python pointers --
  auto mu_app = sp_params_py.mu_app.data<float>();
  auto sigma_app = sp_params_py.sigma_app.data<float>();
  auto logdet_sigma_app = sp_params_py.logdet_sigma_app.data<float>();
  auto prior_mu_app = sp_params_py.prior_mu_app.data<float>();
  auto prior_sigma_app = sp_params_py.prior_sigma_app.data<float>();
  auto prior_mu_app_count = sp_params_py.prior_mu_app_count.data<int>();
  auto prior_sigma_app_count = sp_params_py.prior_sigma_app_count.data<int>();
  auto mu_shape = sp_params_py.mu_shape.data<float>();
  auto sigma_shape = sp_params_py.sigma_shape.data<float>();
  auto logdet_sigma_shape = sp_params_py.logdet_sigma_shape.data<float>();
  auto prior_mu_shape = sp_params_py.prior_mu_shape.data<float>();
  auto prior_sigma_shape = sp_params_py.prior_sigma_shape.data<float>();
  auto prior_mu_shape_count = sp_params_py.prior_mu_shape_count.data<int>();
  auto prior_sigma_shape_count = sp_params_py.prior_sigma_shape_count.data<int>();

  auto counts = sp_params_py.counts.data<int>();
  auto prior_counts = sp_params_py.prior_counts.data<int>();
  auto ids = sp_params_py.ids.data<int>();
  int num = sp_params_py.ids.size(0);

  // -- write from [mu_app,mu_shape,...] to [sp_params] --
  int num_blocks = ceil( double(num) / double(THREADS_PER_BLOCK) ); 
  dim3 nblocks(num_blocks);
  dim3 nthreads(THREADS_PER_BLOCK);
  write_params<<<nblocks,nthreads>>>(mu_app,  sigma_app,  logdet_sigma_app,
                                     prior_mu_app,  prior_sigma_app,
                                     prior_mu_app_count,  prior_sigma_app_count,
                                     mu_shape,  sigma_shape,  logdet_sigma_shape,
                                     prior_mu_shape,  prior_sigma_shape,
                                     prior_mu_shape_count,  prior_sigma_shape_count,
                                     counts,  prior_counts,  ids, sp_params, num);

}


__global__
void write_params(float* mu_app, float* sigma_app, float* logdet_sigma_app,
                  float* prior_mu_app, float* prior_sigma_app,
                  int* prior_mu_app_count, int* prior_sigma_app_count,
                  float* mu_shape, float* sigma_shape, float* logdet_sigma_shape,
                  float* prior_mu_shape, float* prior_sigma_shape,
                  int* prior_mu_shape_count, int* prior_sigma_shape_count,
                  int* counts, int* prior_counts, int* ids,
                  spix_params* sp_params, int nspix) {

    // -- filling superpixel params into image --
    int ix = threadIdx.x + blockIdx.x * blockDim.x;
    if (ix >= nspix) return;

    // -- offset memory access for appearance --
    float* mu_app_ix = mu_app + ix * 3;
    float* sigma_app_ix = sigma_app + ix * 3;
    float* logdet_sigma_app_ix = logdet_sigma_app + ix;
    float* prior_mu_app_ix = prior_mu_app + ix * 3;
    float* prior_sigma_app_ix = prior_sigma_app + ix * 3;
    int* prior_mu_app_count_ix = prior_mu_app_count + ix;
    int* prior_sigma_app_count_ix = prior_sigma_app_count + ix;

    // -- offset memory access for shape --
    float* mu_shape_ix = mu_shape + ix * 2;
    float* sigma_shape_ix = sigma_shape + ix * 3;
    float* logdet_sigma_shape_ix = logdet_sigma_shape + ix;
    float* prior_mu_shape_ix = prior_mu_shape + ix * 2;
    float* prior_sigma_shape_ix = prior_sigma_shape + ix * 3;
    int* prior_mu_shape_count_ix = prior_mu_shape_count + ix;
    int* prior_sigma_shape_count_ix = prior_sigma_shape_count + ix;

    // -- misc --
    int* counts_ix = counts + ix;
    int* prior_counts_ix = prior_counts + ix;

    // -- read spix --
    // int sp_index = ids[ix];
    int sp_index = ix;
    if (sp_index < 0) return;
    auto params_ix = sp_params[sp_index];

    // -- write params from spix_params into the tensors --

    // -- appearance [est] --
    sp_params[sp_index].mu_app.x = mu_app_ix[0];
    sp_params[sp_index].mu_app.y = mu_app_ix[1];
    sp_params[sp_index].mu_app.z = mu_app_ix[2];
    sp_params[sp_index].sigma_app.x = sigma_app_ix[0];
    sp_params[sp_index].sigma_app.y = sigma_app_ix[1];
    sp_params[sp_index].sigma_app.z = sigma_app_ix[2];
    sp_params[sp_index].logdet_sigma_app = logdet_sigma_app_ix[0];

    // -- appearance [prior] --
    sp_params[sp_index].prior_mu_app.x = prior_mu_app_ix[0];
    sp_params[sp_index].prior_mu_app.y = prior_mu_app_ix[1];
    sp_params[sp_index].prior_mu_app.z = prior_mu_app_ix[2];
    sp_params[sp_index].prior_sigma_app.x = prior_sigma_app_ix[0];
    sp_params[sp_index].prior_sigma_app.y = prior_sigma_app_ix[1];
    sp_params[sp_index].prior_sigma_app.z = prior_sigma_app_ix[2];
    sp_params[sp_index].prior_mu_app_count = prior_mu_app_count_ix[0];
    sp_params[sp_index].prior_sigma_app_count = prior_sigma_app_count_ix[0];

    // -- shape [est] --
    sp_params[sp_index].mu_shape.x = mu_shape_ix[0];
    sp_params[sp_index].mu_shape.y = mu_shape_ix[1];
    sp_params[sp_index].sigma_shape.x = sigma_shape_ix[0];
    sp_params[sp_index].sigma_shape.y = sigma_shape_ix[1];
    sp_params[sp_index].sigma_shape.z = sigma_shape_ix[2];
    sp_params[sp_index].logdet_sigma_shape = logdet_sigma_shape_ix[0];

    // -- shape [prior] --
    sp_params[sp_index].prior_mu_shape.x = prior_mu_shape_ix[0];
    sp_params[sp_index].prior_mu_shape.y = prior_mu_shape_ix[1];
    sp_params[sp_index].prior_sigma_shape.x = prior_sigma_shape_ix[0];
    sp_params[sp_index].prior_sigma_shape.y = prior_sigma_shape_ix[1];
    sp_params[sp_index].prior_mu_shape_count = prior_mu_shape_count_ix[0];
    sp_params[sp_index].prior_sigma_shape_count = prior_sigma_shape_count_ix[0];

    // -- misc --
    sp_params[sp_index].count = counts_ix[0];
    sp_params[sp_index].prior_count = prior_counts_ix[0];
}
