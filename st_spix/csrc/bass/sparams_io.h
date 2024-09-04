
#include "cuda.h"
#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#include "core/Superpixels.h"


__global__ void copy_spix_to_params(float* means, float* cov, int* counts,
                                    superpixel_params* sp_params, int* ids, int nspix);
__global__ void copy_spix_to_params_parents(float* means, float* cov,
                                            int* counts, int* spix_parents,
                                            superpixel_params* sp_params,
                                            int* ids, int nspix);
__global__ void copy_spix_to_params_icov2cov(float* means, float* cov, int* counts,
                                             superpixel_params* sp_params,
                                             int* ids, int nspix);
__global__ void copy_params_to_spix(float* means, float* cov, int* counts,
                                 superpixel_params* sp_params, int* ids, int nspix);

