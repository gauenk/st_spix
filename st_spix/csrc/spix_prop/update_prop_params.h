#include "cuda.h"
#include "cuda_runtime.h"
#include "device_launch_parameters.h"

#ifndef MY_SP_STRUCT
#define MY_SP_STRUCT
#include "../share/my_sp_struct.h"
#endif

__global__ void clear_fields(superpixel_params* sp_params,
                             superpixel_GPU_helper* sp_gpu_helper,
                             const int nsuperpixel, const int nsuperpixel_buffer,
                             const int nftrs);

__global__ void sum_by_label(const float* image_gpu_double, const int* seg_gpu,
                             superpixel_params* sp_params,
                             superpixel_GPU_helper* sp_gpu_helper,
                             const int nPixels, const int nbatch,
                             const int xdim, const int nftrs);

__global__ void calculate_mu_and_sigma(superpixel_params* sp_params,
                                       superpixel_GPU_helper* sp_gpu_helper,
                                       const int nsuperpixel,
                                       const int nsuperpixel_buffer,
                                       const int prior_sigma_s,
                                       const int prior_count,
                                       const int nftrs);

__global__ void calculate_mu_spaceonly(superpixel_params* sp_params,
                                       superpixel_GPU_helper* sp_gpu_helper,
                                       const int nsuperpixel,
                                       const int nsuperpixel_buffer,
                                       const int prior_sigma_s,
                                       const int prior_count,
                                       const int nftrs);

__host__ void update_param(const float* image_gpu_double,const int* seg_gpu,
                           superpixel_params* sp_params,
                           superpixel_GPU_helper* sp_gpu_helper,
                           const int nPixels, const int nSps,
                           const int nSps_buffer, const int nbatch,
                           const int xdim, const int ydim, const int nftrs,
                           const int prior_sigma_s, const int prior_count);

__global__ void clear_fields_spaceonly(superpixel_params* sp_params,
                                       superpixel_GPU_helper* sp_gpu_helper,
                                       const int nsuperpixel,
                                       const int nsuperpixel_buffer,
                                       const int nftrs);

__host__ void update_param_spaceonly(const float* image_gpu_double, const int* seg_gpu,
                                     superpixel_params* sp_params,
                                     superpixel_GPU_helper* sp_gpu_helper,
                                     const int nPixels, const int nSps,
                                     const int nSps_buffer, const int nbatch,
                                     const int xdim, const int ydim, const int nftrs,
                                     const int prior_sigma_s, const int prior_count);
/* __global__ void reduce0(const float* image_gpu_double, const int* seg_gpu, */
/*                         superpixel_params* sp_params, */
/*                         superpixel_GPU_helper* sp_gpu_helper, */
/*                         const int nPixels, const int nbatch, const int xdim); */
