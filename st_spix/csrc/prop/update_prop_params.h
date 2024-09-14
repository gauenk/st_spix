#include "cuda.h"
#include "cuda_runtime.h"
#include "device_launch_parameters.h"

#ifndef MY_SP_STRUCT
#define MY_SP_STRUCT
#include "../bass/share/my_sp_struct.h"
#endif

__host__ void update_prop_params(const float* img,const int* seg,
                                 superpixel_params* sp_params,
                                 superpixel_GPU_helper* sp_helper,
                                 superpixel_params* prior_params, int * prior_map,
                                 const int npix, const int nspix,
                                 const int nspix_buffer, const int nbatch,
                                 const int xdim, const int ydim, const int nftrs);

__global__ void clear_fields(superpixel_params* sp_params,
                             superpixel_GPU_helper* sp_helper,
                             const int nsuperpixel, const int nsuperpixel_buffer,
                             const int nftrs);

__global__ void sum_by_label(const float* img, const int* seg,
                             superpixel_params* sp_params,
                             superpixel_GPU_helper* sp_helper,
                             const int npix, const int nbatch,
                             const int xdim, const int nftrs);

__global__ void calculate_mu_and_sigma(superpixel_params* sp_params,
                                       superpixel_GPU_helper* sp_helper,
                                       superpixel_params* prior_params,
                                       int* prior_map, const int nspix,
                                       const int nspix_buffer);


