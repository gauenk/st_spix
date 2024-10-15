
#include "cuda.h"
#include "cuda_runtime.h"
#include "device_launch_parameters.h"

#ifndef MY_SP_STRUCT
#define MY_SP_STRUCT
#include "../bass/share/my_sp_struct.h"
#endif

__host__ PySuperpixelParams get_params_as_tensors(spix_params* sp_params,
                                                  int* ids_ptr, int num);
__host__
spix_params* get_tensors_as_params(PySuperpixelParams params,
                                   int sp_size, int npix, int nspix,
                                   int nspix_buffer);

__host__
void params_to_tensors(PySuperpixelParams sp_params_py,
                       spix_params* sp_params, int num);
__host__
void tensors_to_params(PySuperpixelParams sp_params_py,spix_params* sp_params);

__global__
void read_params(float* mu_app, float* sigma_app, float* logdet_sigma_app,
                 float* prior_mu_app, float* prior_sigma_app,
                 int* prior_mu_app_count, int* prior_sigma_app_count,
                 float* mu_shape, float* sigma_shape, float* logdet_sigma_shape,
                 float* prior_mu_shape, float* prior_sigma_shape,
                 int* prior_mu_shape_count, int* prior_sigma_shape_count,
                 int* counts, int* prior_counts, int* ids,
                 spix_params* sp_params, int spix);
__global__
void write_params(float* mu_app, float* sigma_app, float* logdet_sigma_app,
                  float* prior_mu_app, float* prior_sigma_app,
                  int* prior_mu_app_count, int* prior_sigma_app_count,
                  float* mu_shape, float* sigma_shape, float* logdet_sigma_shape,
                  float* prior_mu_shape, float* prior_sigma_shape,
                  int* prior_mu_shape_count, int* prior_sigma_shape_count,
                  int* counts, int* prior_counts, int* ids,
                  spix_params* sp_params, int nspix);


