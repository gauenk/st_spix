/*************************************************

        This script initializes spix_params
            WITH and WITHOUT propogation

**************************************************/

#include "cuda.h"
#include "cuda_runtime.h"
#include "device_launch_parameters.h"

#ifndef MY_SP_STRUCT
#define MY_SP_STRUCT
#include "../bass/share/my_sp_struct.h"
#endif


/*************************************************

                WITHOUT prior

**************************************************/

__host__ void init_sp_params(spix_params* sp_params, float prior_sigma_app,
                             float* img, int* spix, spix_helper* sp_helper,
                             int npix, int nspix, int nspix_buffer,
                             int nbatch, int width, int nftrs);
__global__ void init_sp_params_kernel(spix_params* sp_params,float prior_sigma_app,
                                      const int nspix, int nspix_buffer, int npix);
__host__
void init_sp_params_from_past(spix_params* curr_params,spix_params* prev_params,
                              float4 rescale, int nspix,int nspix_buffer,int npix);
__global__
void init_sp_params_from_past_kernel(spix_params* curr_params,
                                     spix_params* prev_params,
                                     float4 rescale, int nspix,
                                     int nspix_buffer, int npix);
