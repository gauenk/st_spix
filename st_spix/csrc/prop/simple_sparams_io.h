
#include "cuda.h"
#include "cuda_runtime.h"
#include "device_launch_parameters.h"

#ifndef MY_SP_STRUCT
#define MY_SP_STRUCT
#include "../bass/share/my_sp_struct.h"
#endif

__host__ PySuperpixelParams get_params_as_tensors_s(superpixel_params* sp_params,
                                                  int* ids_ptr, int num);
__host__ superpixel_params* get_tensors_as_params_s(PySuperpixelParams params,
                                                  int sp_size, int npix, int nspix,
                                                  int nspix_buffer);


/* __host__ void get_params_as_tensors(superpixel_params* sp_params, int num); */

__host__
void params_to_tensors_s(PySuperpixelParams sp_params_py,
                         superpixel_params* sp_params, int num);
__host__
void tensors_to_params_s(PySuperpixelParams sp_params_py,superpixel_params* sp_params);

__global__
void read_params_s(float* mu_i, float* mu_s, float* cov, float* log_Sigma_s,
                   int* counts, int* prior_counts, int* ids,
                   superpixel_params* sp_params, int nspix);
__global__
void write_params_s(float* mu_i, float* mu_s, float* cov, float* log_Sigma_s,
                    int* counts, int* prior_counts, int* ids,
                    superpixel_params* sp_params, int nspix);

__global__ void copy_spix_to_params_parents_s(float* means, float* cov,
                                              int* counts, int* spix_parents,
                                              superpixel_params* sp_params,
                                              int* ids, int nspix);
__global__ void copy_spix_to_params_icov2cov_s(float* means, float* cov, int* counts,
                                               superpixel_params* sp_params,
                                               int* ids, int nspix);

