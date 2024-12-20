/************************************************************

     "Read" means we go from spix_params* to PySuperpixelParams
     "Write" means we go from PySuperpixelParams to spix_params* 

*************************************************************/

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
PySuperpixelParams get_output_params(spix_params* sp_params,
                                     PySuperpixelParams prior_params,
                                     int* ids,int num_ids, int max_num){

  // -- init params --
  PySuperpixelParams sp_params_py = init_tensor_params(max_num);

  // -- fill with prior [as much as possible] --
  fill_params_from_params(sp_params_py,prior_params);

  // -- fill the tensors with sp_params --
  params_to_tensors(sp_params_py,sp_params,ids,num_ids);
  return sp_params_py;
  

}


__host__
PySuperpixelParams get_params_as_tensors(spix_params* sp_params,
                                         int* ids, int num_ids, int max_num){

  // -- init params --
  PySuperpixelParams sp_params_py = init_tensor_params(max_num);

  // -- fill the tensors with sp_params --
  params_to_tensors(sp_params_py,sp_params,ids,num_ids);
  return sp_params_py;
  
}

__host__
void fill_params_from_params(PySuperpixelParams dest_params,
                             PySuperpixelParams src_params){


  // -- check --
  int size = src_params.ids.size(0);
  int size_d = dest_params.ids.size(0);
  assert(size_d >= size);
  
  // -- appearance --
  dest_params.mu_app.narrow(0, 0, size).copy_(src_params.mu_app); 
  dest_params.sigma_app.narrow(0, 0, size).copy_(src_params.sigma_app); 
  dest_params.logdet_sigma_app.\
    narrow(0, 0, size).copy_(src_params.logdet_sigma_app);
  dest_params.prior_mu_app.narrow(0, 0, size).copy_(src_params.prior_mu_app);
  dest_params.prior_sigma_app.\
    narrow(0, 0, size).copy_(src_params.prior_sigma_app);
  dest_params.prior_mu_app_count.\
    narrow(0, 0, size).copy_(src_params.prior_mu_app_count);
  dest_params.prior_sigma_app_count\
    .narrow(0, 0, size).copy_(src_params.prior_sigma_app_count);

  // -- shape --
  dest_params.mu_shape.narrow(0, 0, size).copy_(src_params.mu_shape); 
  dest_params.sigma_shape.narrow(0, 0, size).copy_(src_params.sigma_shape); 
  dest_params.logdet_sigma_shape.\
    narrow(0, 0, size).copy_(src_params.logdet_sigma_shape);
  dest_params.prior_mu_shape.narrow(0, 0, size).copy_(src_params.prior_mu_shape);
  dest_params.prior_sigma_shape\
    .narrow(0, 0, size).copy_(src_params.prior_sigma_shape);
  dest_params.prior_mu_shape_count.\
    narrow(0, 0, size).copy_(src_params.prior_mu_shape_count);
  dest_params.prior_sigma_shape_count\
    .narrow(0, 0, size).copy_(src_params.prior_sigma_shape_count);

  // -- helpers --
  dest_params.counts.narrow(0, 0, size).copy_(src_params.counts); 
  dest_params.prior_counts.narrow(0, 0, size).copy_(src_params.prior_counts); 
  dest_params.ids.narrow(0, 0, size).copy_(src_params.ids); 

}


__host__
PySuperpixelParams init_tensor_params(int size){

  // -- allocate helpers --
  torch::Device device(torch::kCUDA, 0); 
  auto options_i32 = torch::TensorOptions().dtype(torch::kInt32)
    .layout(torch::kStrided).device(device);
  auto options_f32 = torch::TensorOptions().dtype(torch::kFloat32)
    .layout(torch::kStrided).device(device);
  auto options_f64 = torch::TensorOptions().dtype(torch::kFloat64)
    .layout(torch::kStrided).device(device);


  // -- allocate superpixel params --
  PySuperpixelParams sp_params_py;

  // -- appearance --
  sp_params_py.mu_app = torch::zeros({size,3},options_f32);
  sp_params_py.sigma_app = torch::zeros({size,3},options_f32);
  sp_params_py.logdet_sigma_app = torch::zeros({size},options_f32);
  sp_params_py.prior_mu_app = torch::zeros({size,3},options_f32);
  sp_params_py.prior_sigma_app = torch::zeros({size,3},options_f32);
  sp_params_py.prior_mu_app_count = torch::zeros({size},options_i32);
  sp_params_py.prior_sigma_app_count = torch::zeros({size},options_i32);

  // -- shape --
  sp_params_py.mu_shape = torch::zeros({size,2},options_f64);
  sp_params_py.sigma_shape = torch::zeros({size,3},options_f64);
  sp_params_py.logdet_sigma_shape = torch::zeros({size},options_f32);
  sp_params_py.prior_mu_shape = torch::zeros({size,2},options_f64);
  sp_params_py.prior_sigma_shape = torch::zeros({size,3},options_f64);
  sp_params_py.prior_mu_shape_count = torch::zeros({size},options_i32);
  sp_params_py.prior_sigma_shape_count = torch::zeros({size},options_i32);

  // -- helpers --
  sp_params_py.counts = torch::zeros({size},options_i32);
  sp_params_py.prior_counts = torch::zeros({size},options_f32);
  sp_params_py.ids = torch::zeros({size},options_i32); // i think ".ids" should be deleted
  // sp_params_py.ids = torch::from_blob(ids,{size},options_i32); // I think this is a bad idea

  return sp_params_py;

}


__host__ spix_params* get_tensors_as_params(PySuperpixelParams params,
                                            int sp_size, int npix,
                                            int nspix, int nspix_buffer){
  // -- allocate helpers --
  torch::Device device(torch::kCUDA, 0); 
  auto options_i32 = torch::TensorOptions().dtype(torch::kInt32)
    .layout(torch::kStrided).device(device);
  auto options_f32 = torch::TensorOptions().dtype(torch::kFloat32)
    .layout(torch::kStrided).device(device);

  // -- allocate superpixel params --
  const int sparam_size = sizeof(spix_params);
  spix_params* sp_params=(spix_params*)easy_allocate(nspix_buffer,sparam_size);
  cudaMemset(sp_params,0,nspix_buffer*sparam_size);
  // init_sp_params(sp_params,sp_size,nspix,nspix_buffer,npix);

  // -- check legal accessing --
  // int num = params.ids.size(0);
  // assert(num <= nspix_buffer); // buffer must be larger (and very probably is)

  // -- fill the tensors with sp_params --
  tensors_to_params(params,sp_params);
  return sp_params;
  
}

__host__ void params_to_tensors(PySuperpixelParams sp_params_py,
                                spix_params* sp_params, int* ids, int num){
  
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
  auto mu_shape = sp_params_py.mu_shape.data<double>();
  auto sigma_shape = sp_params_py.sigma_shape.data<double>();
  auto logdet_sigma_shape = sp_params_py.logdet_sigma_shape.data<float>();
  auto prior_mu_shape = sp_params_py.prior_mu_shape.data<double>();
  auto prior_sigma_shape = sp_params_py.prior_sigma_shape.data<double>();
  auto prior_mu_shape_count = sp_params_py.prior_mu_shape_count.data<int>();
  auto prior_sigma_shape_count = sp_params_py.prior_sigma_shape_count.data<int>();

  // -- misc --
  auto counts = sp_params_py.counts.data<int>();
  auto prior_counts = sp_params_py.prior_counts.data<float>();
  // auto ids = sp_params_py.ids.data<int>();
  // int max_num = sp_params_py.ids.size(0);
  // assert(num <= max_num);

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
                                    counts, prior_counts, sp_params, ids, num);
  
}


__global__ 
void read_params(float* mu_app, float* sigma_app, float* logdet_sigma_app,
                 float* prior_mu_app, float* prior_sigma_app,
                 int* prior_mu_app_count, int* prior_sigma_app_count,
                 double* mu_shape, double* sigma_shape, float* logdet_sigma_shape,
                 double* prior_mu_shape, double* prior_sigma_shape,
                 int* prior_mu_shape_count, int* prior_sigma_shape_count,
                 int* counts, float* prior_counts, 
                 spix_params* sp_params, int* ids, int num_spix){

    // -- filling superpixel params into image --
    int _ix = threadIdx.x + blockIdx.x * blockDim.x;
    if (_ix >= num_spix) return;  // 'nspix' should be 'spix', as passed by the caller
    // if (ix >= max_spix) return;  // 'nspix' should be 'spix', as passed by the caller
    int ix = ids[_ix];
    int sp_index = ids[_ix];
    if (sp_index < 0){ return; }

    // -- offest memory access [appearance] --
    float* mu_app_ix = mu_app + ix * 3;
    float* sigma_app_ix = sigma_app + ix * 3;  // Handle sigma_app
    float* logdet_sigma_app_ix = logdet_sigma_app + ix;  // Handle logdet_sigma_app
    float* prior_mu_app_ix = prior_mu_app + ix * 3;  // Handle prior_mu_app
    float* prior_sigma_app_ix = prior_sigma_app + ix * 3;  // Handle prior_sigma_app
    int* prior_mu_app_count_ix = prior_mu_app_count + ix; 
    int* prior_sigma_app_count_ix = prior_sigma_app_count + ix;

    // -- offest memory access [shape] --
    double* mu_shape_ix = mu_shape + ix * 2;
    double* sigma_shape_ix = sigma_shape + ix * 3;  // Handle sigma_shape
    float* logsigma_shape_ix = logdet_sigma_shape + ix;  // Already in the code
    double* prior_mu_shape_ix = prior_mu_shape + ix * 2;  // Handle prior_mu_shape
    double* prior_sigma_shape_ix = prior_sigma_shape + ix * 3; // Handle prior_sigma_shape
    int* prior_mu_shape_count_ix = prior_mu_shape_count + ix; 
    int* prior_sigma_shape_count_ix = prior_sigma_shape_count + ix; 

    // -- misc --
    int* counts_ix = counts + ix;
    float* prior_counts_ix = prior_counts + ix;  
                                                                                          
    // -- read spix --
    auto params_ix = sp_params[sp_index];
    
    /*****************************************************

                    Fill the Params                             

    *****************************************************/

    // -- appearance [est] --
    mu_app_ix[0] = params_ix.mu_app.x;
    mu_app_ix[1] = params_ix.mu_app.y;
    mu_app_ix[2] = params_ix.mu_app.z;
    // sigma_app_ix[0] = params_ix.sigma_app.x;  // Fill sigma_app 
    // sigma_app_ix[1] = params_ix.sigma_app.y;
    // sigma_app_ix[2] = params_ix.sigma_app.z;
    // logdet_sigma_app_ix[0] = params_ix.logdet_sigma_app;  // Fill logdet_sigma_app
    // -- appearance [prior] --
    prior_mu_app_ix[0] = params_ix.prior_mu_app.x;  // Fill prior_mu_app
    prior_mu_app_ix[1] = params_ix.prior_mu_app.y;
    prior_mu_app_ix[2] = params_ix.prior_mu_app.z;
    prior_mu_app_count_ix[0] = params_ix.prior_mu_app_count;
    // prior_sigma_app_ix[0] = params_ix.prior_sigma_app.x;  // Fill prior_sigma_app
    // prior_sigma_app_ix[1] = params_ix.prior_sigma_app.y;
    // prior_sigma_app_ix[2] = params_ix.prior_sigma_app.z;
    // prior_sigma_app_count_ix[0] = params_ix.prior_sigma_app_count;

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
    prior_sigma_shape_ix[2] = params_ix.prior_sigma_shape.z;
    prior_mu_shape_count_ix[0] = params_ix.prior_mu_shape_count;
    prior_sigma_shape_count_ix[0] = params_ix.prior_sigma_shape_count;

    // -- misc --
    counts_ix[0] = params_ix.count;
    prior_counts_ix[0] = params_ix.prior_count;


}

__host__ void tensors_to_params(PySuperpixelParams sp_params_py,
                                spix_params* sp_params){
  
  // -- unpack python pointers --
  auto mu_app = sp_params_py.mu_app.data<float>();
  auto sigma_app = sp_params_py.sigma_app.data<float>();
  auto logdet_sigma_app = sp_params_py.logdet_sigma_app.data<float>();
  auto prior_mu_app = sp_params_py.prior_mu_app.data<float>();
  auto prior_sigma_app = sp_params_py.prior_sigma_app.data<float>();
  auto prior_mu_app_count = sp_params_py.prior_mu_app_count.data<int>();
  auto prior_sigma_app_count = sp_params_py.prior_sigma_app_count.data<int>();
  auto mu_shape = sp_params_py.mu_shape.data<double>();
  auto sigma_shape = sp_params_py.sigma_shape.data<double>();
  auto logdet_sigma_shape = sp_params_py.logdet_sigma_shape.data<float>();
  auto prior_mu_shape = sp_params_py.prior_mu_shape.data<double>();
  auto prior_sigma_shape = sp_params_py.prior_sigma_shape.data<double>();
  auto prior_mu_shape_count = sp_params_py.prior_mu_shape_count.data<int>();
  auto prior_sigma_shape_count = sp_params_py.prior_sigma_shape_count.data<int>();

  auto counts = sp_params_py.counts.data<int>();
  auto prior_counts = sp_params_py.prior_counts.data<float>();
  // auto ids = sp_params_py.ids.data<int>();
  int max_num = sp_params_py.ids.size(0); // keep this for now
  // remove "ids" when we are more sure we won't want it

  // -- write from [mu_app,mu_shape,...] to [sp_params] --
  int num_blocks = ceil( double(max_num) / double(THREADS_PER_BLOCK) ); 
  dim3 nblocks(num_blocks);
  dim3 nthreads(THREADS_PER_BLOCK);
  write_params<<<nblocks,nthreads>>>(mu_app,  sigma_app,  logdet_sigma_app,
                                     prior_mu_app,  prior_sigma_app,
                                     prior_mu_app_count,  prior_sigma_app_count,
                                     mu_shape,  sigma_shape,  logdet_sigma_shape,
                                     prior_mu_shape,  prior_sigma_shape,
                                     prior_mu_shape_count,  prior_sigma_shape_count,
                                     counts,  prior_counts, sp_params, max_num);

}


__host__
void write_prior_counts(PySuperpixelParams src_params,spix_params* dest_params,
                        int* ids, int nactive){
  float* prior_counts = src_params.prior_counts.data<float>();
  int num_blocks = ceil( double(nactive) / double(THREADS_PER_BLOCK) ); 
  dim3 nblocks(num_blocks);
  dim3 nthreads(THREADS_PER_BLOCK);
  write_prior_counts_kernel<<<nblocks,nthreads>>>(prior_counts, dest_params,
                                                  ids, nactive);
}

__global__
void write_prior_counts_kernel(float* prior_counts, spix_params* sp_params,
                               int* ids, int nactive) {
    // -- filling superpixel params into image --
    int ix = threadIdx.x + blockIdx.x * blockDim.x;
    // if (ix >= nspix) return;
    if (ix >= nactive) return;
    int spix_id = ids[ix];
    // printf("[%d]: %d,%f\n",ix,prior_counts[ix],sp_params[ix].prior_count);
    sp_params[spix_id].prior_count = prior_counts[spix_id];
}


__global__
void write_params(float* mu_app, float* sigma_app, float* logdet_sigma_app,
                  float* prior_mu_app, float* prior_sigma_app,
                  int* prior_mu_app_count, int* prior_sigma_app_count,
                  double* mu_shape, double* sigma_shape, float* logdet_sigma_shape,
                  double* prior_mu_shape, double* prior_sigma_shape,
                  int* prior_mu_shape_count, int* prior_sigma_shape_count,
                  int* counts, float* prior_counts, spix_params* sp_params, int nspix) {

   /**********************************************************************

           Fills sp_params with information from PySuperpixelParams

           Option 1.) Always fill all of them.

           The length of any tensor in "PySuperpixelParams"
           is the maximum size of the superpixel

    **********************************************************************/

    // -- filling superpixel params into image --
    int ix = threadIdx.x + blockIdx.x * blockDim.x;
    if (ix >= nspix) return;

    // -- offset memory access for appearance --
    float* mu_app_ix = mu_app + ix * 3;
    // float* sigma_app_ix = sigma_app + ix * 3;
    // float* logdet_sigma_app_ix = logdet_sigma_app + ix;
    float* prior_mu_app_ix = prior_mu_app + ix * 3;
    // float* prior_sigma_app_ix = prior_sigma_app + ix * 3;
    int* prior_mu_app_count_ix = prior_mu_app_count + ix;
    // int* prior_sigma_app_count_ix = prior_sigma_app_count + ix;

    // -- offset memory access for shape --
    double* mu_shape_ix = mu_shape + ix * 2;
    double* sigma_shape_ix = sigma_shape + ix * 3;
    float* logdet_sigma_shape_ix = logdet_sigma_shape + ix;
    double* prior_mu_shape_ix = prior_mu_shape + ix * 2;
    double* prior_sigma_shape_ix = prior_sigma_shape + ix * 3;
    int* prior_mu_shape_count_ix = prior_mu_shape_count + ix;
    int* prior_sigma_shape_count_ix = prior_sigma_shape_count + ix;

    // -- misc --
    int* counts_ix = counts + ix;
    float* prior_counts_ix = prior_counts + ix;

    // -- read spix --
    // int sp_index = ids[ix];
    int sp_index = ix;
    if (sp_index < 0) return;

    // -- write params from spix_params into the tensors --

    // -- appearance [est] --
    sp_params[sp_index].mu_app.x = mu_app_ix[0];
    sp_params[sp_index].mu_app.y = mu_app_ix[1];
    sp_params[sp_index].mu_app.z = mu_app_ix[2];
    // sp_params[sp_index].sigma_app.x = sigma_app_ix[0];
    // sp_params[sp_index].sigma_app.y = sigma_app_ix[1];
    // sp_params[sp_index].sigma_app.z = sigma_app_ix[2];
    // sp_params[sp_index].logdet_sigma_app = logdet_sigma_app_ix[0];

    // -- appearance [prior] --
    sp_params[sp_index].prior_mu_app.x = prior_mu_app_ix[0];
    sp_params[sp_index].prior_mu_app.y = prior_mu_app_ix[1];
    sp_params[sp_index].prior_mu_app.z = prior_mu_app_ix[2];
    // sp_params[sp_index].prior_sigma_app.x = prior_sigma_app_ix[0];
    // sp_params[sp_index].prior_sigma_app.y = prior_sigma_app_ix[1];
    // sp_params[sp_index].prior_sigma_app.z = prior_sigma_app_ix[2];
    sp_params[sp_index].prior_mu_app_count = prior_mu_app_count_ix[0];
    // sp_params[sp_index].prior_sigma_app_count = prior_sigma_app_count_ix[0];

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
    // sp_params[sp_index].prior_sigma_shape.x = prior_sigma_shape_ix[0];
    // sp_params[sp_index].prior_sigma_shape.y = prior_sigma_shape_ix[1];
    // sp_params[sp_index].prior_sigma_shape.z = prior_sigma_shape_ix[2];
    sp_params[sp_index].prior_mu_shape_count = prior_mu_shape_count_ix[0];
    sp_params[sp_index].prior_sigma_shape_count = prior_sigma_shape_count_ix[0];

    // -- shape ... --
    double cov_xx = prior_sigma_shape_ix[0];
    double cov_xy = prior_sigma_shape_ix[1];
    double cov_yy = prior_sigma_shape_ix[2];
    // float det = cov_xx * cov_yy - cov_xy*cov_xy;
    // if (det < 0){ det = 0.00001; }
    // prior_sigma_shape_ptr[0] = icov_yy/det;
    // prior_sigma_shape_ptr[1] = -icov_xy/det;
    // prior_sigma_shape_ptr[2] = icov_xx/det;
    // sp_params[sp_index].prior_sigma_shape.x = cov_yy/det;
    // sp_params[sp_index].prior_sigma_shape.y = -cov_xy/det;
    // sp_params[sp_index].prior_sigma_shape.z = cov_xx/det;
    sp_params[sp_index].prior_sigma_shape.x = cov_xx;
    sp_params[sp_index].prior_sigma_shape.y = cov_xy;
    sp_params[sp_index].prior_sigma_shape.z = cov_yy;
    // printf("prior_sigma: %lf %lf %lf\n",
    //        sp_params[sp_index].prior_sigma_shape.x,
    //        sp_params[sp_index].prior_sigma_shape.y,
    //        sp_params[sp_index].prior_sigma_shape.z);

    // -- misc --
    sp_params[sp_index].count = counts_ix[0];
    sp_params[sp_index].prior_count = prior_counts_ix[0];
    sp_params[sp_index].prop = true;
    sp_params[sp_index].valid = 0; // this is set later

    // sp_params[sp_index].prop = true;
}



/********************************************************************

              Ensure the New Superpixels "LABEL" Are Compact

   For example: [0,1,2] + [3,5,6] -> [0,1,2] + [3,4,5]


   -- ensure the new spix are added without including empty spix --
   i don't think its serious to keep these,
   but conceptually it may be confusing when analyzing the output

   the only danger would be not-initialized PySuperpixelParams...
    so maybe this danger is real...



********************************************************************/

__global__ 
void compact_new_spix(int* spix, int* compression_map, int* prop_ids,
                      int num_new, int prev_nspix, int npix){

  // -- indexing pixel indices --
  int ix = threadIdx.x + blockIdx.x * blockDim.x;
  if (ix >= npix) return;

  // -- read current id --
  int spix_id = spix[ix];
  if (spix_id < prev_nspix){ return; } // numbering starts @ 0; so "=prev_nspix" is new

  // -- update to compact index if "new" --
  // int shift_ix = spix_id - prev_nspix;
  for (int jx=0; jx < num_new; jx++){
    if (spix_id == prop_ids[jx]){
      int new_spix_id = jx+prev_nspix;
      spix[ix] = jx+prev_nspix; // update to "index" within "prop_ids" offset by prev max
      compression_map[jx] = spix_id; // for updating spix_params
      break;
    }
  }

}

__global__ 
void fill_new_params_from_old(spix_params* params, spix_params*  new_params,
                              int* compression_map, int num_new){

  // -- indexing new superpixel labels --
  int dest_ix = threadIdx.x + blockIdx.x * blockDim.x;
  if (dest_ix >= num_new) return;
  int src_ix = compression_map[dest_ix];
  // new_params[dest_ix] = params[src_ix];

  new_params[dest_ix].mu_app = params[src_ix].mu_app;
  // new_params[dest_ix].sigma_app = params[src_ix].sigma_app;
  new_params[dest_ix].prior_mu_app = params[src_ix].prior_mu_app;
  // new_params[dest_ix].prior_sigma_app = params[src_ix].prior_sigma_app;
  new_params[dest_ix].prior_mu_app_count = params[src_ix].prior_mu_app_count;
  // new_params[dest_ix].prior_sigma_app_count = params[src_ix].prior_sigma_app_count;
  new_params[dest_ix].mu_shape = params[src_ix].mu_shape;
  new_params[dest_ix].sigma_shape = params[src_ix].sigma_shape;
  new_params[dest_ix].prior_mu_shape = params[src_ix].prior_mu_shape;
  new_params[dest_ix].prior_sigma_shape = params[src_ix].prior_sigma_shape;
  new_params[dest_ix].prior_mu_shape_count = params[src_ix].prior_mu_shape_count;
  new_params[dest_ix].prior_sigma_shape_count = params[src_ix].prior_sigma_shape_count;
  // new_params[dest_ix].logdet_sigma_app = params[src_ix].logdet_sigma_app;
  new_params[dest_ix].logdet_sigma_shape = params[src_ix].logdet_sigma_shape;
  new_params[dest_ix].logdet_prior_sigma_shape = params[src_ix].logdet_prior_sigma_shape;
  new_params[dest_ix].prior_lprob = params[src_ix].prior_lprob;
  new_params[dest_ix].count = params[src_ix].count;
  new_params[dest_ix].prior_count = params[src_ix].prior_count;
  new_params[dest_ix].valid = params[src_ix].valid;

}

__global__ 
void fill_old_params_from_new(spix_params* params, spix_params*  new_params,
                              int prev_max, int num_new){

  // -- indexing new superpixel labels --
  int ix = threadIdx.x + blockIdx.x * blockDim.x;
  if (ix >= num_new) return;
  params[prev_max+ix].mu_app = new_params[ix].mu_app;
  // params[prev_max+ix].sigma_app = new_params[ix].sigma_app;
  params[prev_max+ix].prior_mu_app = new_params[ix].prior_mu_app;
  // params[prev_max+ix].prior_sigma_app = new_params[ix].prior_sigma_app;
  params[prev_max+ix].prior_mu_app_count = new_params[ix].prior_mu_app_count;
  // params[prev_max+ix].prior_sigma_app_count = new_params[ix].prior_sigma_app_count;
  params[prev_max+ix].mu_shape = new_params[ix].mu_shape;
  params[prev_max+ix].sigma_shape = new_params[ix].sigma_shape;
  params[prev_max+ix].prior_mu_shape = new_params[ix].prior_mu_shape;
  params[prev_max+ix].prior_sigma_shape = new_params[ix].prior_sigma_shape;
  params[prev_max+ix].prior_mu_shape_count = new_params[ix].prior_mu_shape_count;
  params[prev_max+ix].prior_sigma_shape_count = new_params[ix].prior_sigma_shape_count;
  // params[prev_max+ix].logdet_sigma_app = new_params[ix].logdet_sigma_app;
  params[prev_max+ix].logdet_sigma_shape = new_params[ix].logdet_sigma_shape;
  params[prev_max+ix].logdet_prior_sigma_shape = new_params[ix].logdet_prior_sigma_shape;
  params[prev_max+ix].prior_lprob = new_params[ix].prior_lprob;
  params[prev_max+ix].count = new_params[ix].count;
  params[prev_max+ix].prior_count = new_params[ix].prior_count;
  params[prev_max+ix].valid = new_params[ix].valid;


}

int compactify_new_superpixels(torch::Tensor spix,spix_params* sp_params,
                               int prev_nspix,int max_spix,int npix){

  // -- get new ids --
  auto unique_ids = std::get<0>(at::_unique(spix));
  auto ids = unique_ids.data<int>();
  int num_ids = unique_ids.sizes()[0];
  auto mask = unique_ids >= prev_nspix;
  auto prop_ids = unique_ids.masked_select(mask); // uncompressed and alive

  // -- update maximum number of superpixels --
  int num_new = prop_ids.size(0);
  int compact_nspix = prev_nspix + num_new;
  // fprintf(stdout,"num_new: %d\n",num_new);
  if (num_new == 0) {return compact_nspix;}

  // -- allocate spix for storing --
  spix_params* new_params=(spix_params*)easy_allocate(num_new,sizeof(spix_params));
  int* compression_map=(int*)easy_allocate(num_new,sizeof(int));

  // -- update spix labels and params to reflect compacted labeling --
  int* spix_ptr = spix.data<int>();
  int* prop_ids_ptr = prop_ids.data<int>();
  int num_blocks = ceil( double(npix) / double(THREADS_PER_BLOCK) ); 
  dim3 nblocks(num_blocks);
  dim3 nthreads(THREADS_PER_BLOCK);
  compact_new_spix<<<nblocks,nthreads>>>(spix_ptr,compression_map,
                                         prop_ids_ptr,num_new,prev_nspix,npix);

  // -- shift params into correct location --
  int num_blocks1 = ceil( double(num_new) / double(THREADS_PER_BLOCK) ); 
  dim3 nblocks1(num_blocks1);
  dim3 nthreads1(THREADS_PER_BLOCK);
  fill_new_params_from_old<<<nblocks1,nthreads1>>>(sp_params,new_params,
                                                  compression_map,num_new);
  fill_old_params_from_new<<<nblocks1,nthreads1>>>(sp_params,new_params,
                                                  prev_nspix,num_new);

  // -- free parameters --
  cudaFree(compression_map);
  cudaFree(new_params);

  return compact_nspix;
}


/*********************************************************************





             -=-=-=-=- Python API  -=-=-=-=-=-

    Update the priors using the spix, img, and PySuperpixelParams







**********************************************************************/


// __global__
// void update_prior_kernel(float* mu_app, float* prior_mu_app,
//                          float* mu_shape, float* prior_mu_shape,
//                          float* sigma_shape, float* prior_sigma_shape,
//                          int* ids, int nspix, int prev_max_spix);


__global__
void update_prior_kernel(float* mu_app, float* prior_mu_app,
                         double* mu_shape, double* prior_mu_shape,
                         double* sigma_shape, double* prior_sigma_shape,
                         int* ids, int nspix, int prev_nspix, bool invert) {

    // -- filling superpixel params into image --
    int ix = threadIdx.x + blockIdx.x * blockDim.x;
    if (ix >= nspix) return;
    int sp_index = ids[ix];
    if (sp_index < 0){ return; } // not needed; remove me
    // printf("[update_prior_kernel.a]: %d\n",sp_index);
    if (sp_index < prev_nspix){ return; }
    // printf("[update_prior_kernel]: %d\n",sp_index);

    // -- mean appearance --
    float* prior_mu_app_ptr = prior_mu_app + 3*sp_index;
    float* mu_app_ptr = mu_app + 3*sp_index;
    prior_mu_app_ptr[0] = mu_app_ptr[0];
    prior_mu_app_ptr[1] = mu_app_ptr[1];
    prior_mu_app_ptr[2] = mu_app_ptr[2];

    // -- mean shape --
    double* prior_mu_shape_ptr = prior_mu_shape + 2*sp_index;
    double* mu_shape_ptr = mu_shape + 2*sp_index;
    prior_mu_shape_ptr[0] = mu_shape_ptr[0];
    prior_mu_shape_ptr[1] = mu_shape_ptr[1];

    // -- cov shape --
    double* prior_sigma_shape_ptr = prior_sigma_shape + 3*sp_index;
    double* sigma_shape_ptr = sigma_shape + 3*sp_index;
    double icov_xx = sigma_shape_ptr[0];
    double icov_xy = sigma_shape_ptr[1];
    double icov_yy = sigma_shape_ptr[2];
    double det = icov_xx * icov_yy - icov_xy*icov_xy;
    if (det < 0){ det = 0.000001; }

    if (invert){
      prior_sigma_shape_ptr[0] = icov_yy/det;
      prior_sigma_shape_ptr[1] = -icov_xy/det;
      prior_sigma_shape_ptr[2] = icov_xx/det;
    }else{
      prior_sigma_shape_ptr[0] = icov_xx;
      prior_sigma_shape_ptr[1] = icov_xy;
      prior_sigma_shape_ptr[2] = icov_yy;
    }
    // prior_sigma_shape_ptr[0] = sigma_shape_ptr[0];
    // prior_sigma_shape_ptr[1] = sigma_shape_ptr[1];
    // prior_sigma_shape_ptr[2] = sigma_shape_ptr[2];

}


void run_update_prior(const torch::Tensor spix,PySuperpixelParams params,
                      int prev_nspix, bool invert){

    // -- check --
    CHECK_INPUT(spix);
    CHECK_INPUT(params.mu_app);
    CHECK_INPUT(params.mu_shape);
    CHECK_INPUT(params.sigma_shape);
    CHECK_INPUT(params.logdet_sigma_shape);
    CHECK_INPUT(params.counts);
    CHECK_INPUT(params.prior_counts);

    // -- get spixel parameters as tensors --
    // int max_spix = spix.max().item<int>();
    auto unique_ids = std::get<0>(at::_unique(spix));
    auto ids = unique_ids.data<int>();
    int nspix = unique_ids.sizes()[0];
    int _nspix = spix.max().item<int>()+1;
    // int _min = spix.min().item<int>();
    // printf("nspix,_nspix,_min: %d,%d,%d\n",nspix,_nspix,_min);
    assert(nspix <= _nspix);

    // -- unpack --
    float* mu_app = params.mu_app.data<float>();
    float* prior_mu_app = params.prior_mu_app.data<float>();
    double* mu_shape = params.mu_shape.data<double>();
    double* prior_mu_shape = params.prior_mu_shape.data<double>();
    double* sigma_shape = params.sigma_shape.data<double>();
    double* prior_sigma_shape = params.prior_sigma_shape.data<double>();

    // -- launch copy kernel --
    int num_blocks = ceil( double(nspix) / double(THREADS_PER_BLOCK) ); 
    dim3 nblocks(num_blocks);
    dim3 nthreads(THREADS_PER_BLOCK);
    update_prior_kernel<<<nblocks,nthreads>>>(mu_app, prior_mu_app, mu_shape,
                                              prior_mu_shape, sigma_shape,
                                              prior_sigma_shape, ids,
                                              nspix, prev_nspix, invert);

}

void init_sparams_io(py::module &m){
  m.def("run_update_prior", &run_update_prior,"copy mode estimates to prior");
}


