#include "cuda.h"
#include "cuda_runtime.h"
#include "device_launch_parameters.h"

#ifndef MY_SP_STRUCT
#define MY_SP_STRUCT
#include "../bass/share/my_sp_struct.h"
#endif

__device__ inline void
calc_transition(float& xfer0, float& xfer1,
                int spix0, int spix1, int spix_curr, float* pix,
                superpixel_params* sp_params,
                superpixel_params* sp_params_prev,
                superpixel_GPU_helper* sp_gpu_helper);

__device__ inline void
xfer_case0(float& xfer0, float& xfer1,
           int spix_curr, int spix_prop, float* pix,
           superpixel_params* sp_params,
           superpixel_params* sp_params_prev,
           superpixel_GPU_helper* sp_gpu_helper);

__device__ inline void
xfer_case0_means(float& xfer0, float& xfer1,
                 int spix_curr, int spix_prop, float* pix,
                 superpixel_params* sp_params,
                 superpixel_params* sp_params_prev,
                 superpixel_GPU_helper* sp_gpu_helper);

__device__ inline void
get_updated_means(float3& mu, float* pix,
                       superpixel_GPU_helper* sp_gpu_helper,
                       int spix, int sign);

__device__ inline void
transition_mean_probs(float& probs, float3 mu_i_prop, float3 mu_i_curr);




