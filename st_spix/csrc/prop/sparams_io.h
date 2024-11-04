
#include "cuda.h"
#include "cuda_runtime.h"
#include "device_launch_parameters.h"

#ifndef MY_SP_STRUCT
#define MY_SP_STRUCT
#include "../bass/share/my_sp_struct.h"
#endif

PySuperpixelParams get_params_as_tensors(spix_params* sp_params, int* ids,
                                         int num, int max_num);
/* __host__ */
/* PySuperpixelParams get_params_as_tensors(spix_params* sp_params, */
/*                                          int* ids, int num); */
__host__
spix_params* get_tensors_as_params(PySuperpixelParams params, int sp_size,
                                   int npix, int nspix,int nspix_buffer);

__host__
void fill_params_from_params(PySuperpixelParams dest_params,
                             PySuperpixelParams src_params);

__host__
void params_to_tensors(PySuperpixelParams sp_params_py,
                       spix_params* sp_params, int* ids, int num);
__host__
void tensors_to_params(PySuperpixelParams sp_params_py,
                       spix_params* sp_params);

__host__
PySuperpixelParams get_output_params(spix_params* sp_params,
                                     PySuperpixelParams prior_params,
                                     int* ids,int num, int max_num);

__global__
void read_params(float* mu_app, float* sigma_app, float* logdet_sigma_app,
                 float* prior_mu_app, float* prior_sigma_app,
                 int* prior_mu_app_count, int* prior_sigma_app_count,
                 float* mu_shape, double* sigma_shape, float* logdet_sigma_shape,
                 float* prior_mu_shape, double* prior_sigma_shape,
                 int* prior_mu_shape_count, int* prior_sigma_shape_count,
                 int* counts, float* prior_counts, spix_params* sp_params,
                 int* ids, int spix);
__global__
void write_params(float* mu_app, float* sigma_app, float* logdet_sigma_app,
                  float* prior_mu_app, float* prior_sigma_app,
                  int* prior_mu_app_count, int* prior_sigma_app_count,
                  float* mu_shape, double* sigma_shape, float* logdet_sigma_shape,
                  float* prior_mu_shape, double* prior_sigma_shape,
                  int* prior_mu_shape_count, int* prior_sigma_shape_count,
                  int* counts, float* prior_counts, spix_params* sp_params, int nspix);

void run_update_prior(const torch::Tensor spix,PySuperpixelParams params,
                      int max_spix, bool invert);

__global__
void update_prior_kernel(float* mu_app, float* prior_mu_app,
                         float* mu_shape, float* prior_mu_shape,
                         double* sigma_shape, double* prior_sigma_shape,
                         int* ids, int nspix, int prev_max_spix, bool invert);
/* __host__ */
/* void write_prior_counts(PySuperpixelParams src_params,spix_params* dest_params); */


__host__
void write_prior_counts(PySuperpixelParams src_params,spix_params* dest_params,
                        int* ids, int nactive);

__global__
void write_prior_counts_kernel(float* prior_counts, spix_params* sp_params,
                               int* ids, int nactive);


// -- compact --
__global__
void compact_new_spix(int* spix, int* compression_map, int* prop_ids,
                      int num_new, int prev_max, int npix);
__global__
void fill_new_params_from_old(spix_params* params, spix_params*  new_params,
                              int* compression_map, int num_new);
__global__
void fill_old_params_from_new(spix_params* params, spix_params*  new_params,
                              int prev_max, int num_new);
int compactify_new_superpixels(torch::Tensor spix,spix_params* sp_params,
                               int prev_max_spix,int max_spix,int npix);



__host__ PySuperpixelParams init_tensor_params(int size);

