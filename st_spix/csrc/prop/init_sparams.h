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

__host__ void init_sp_params(spix_params* sp_params, const int sp_size,
                             const int nspix, int nspix_buffer, int npix);
__global__ void init_sp_params_kernel(spix_params* sp_params, const int sp_size,
                                      const int nspix, int nspix_buffer, int npix);
