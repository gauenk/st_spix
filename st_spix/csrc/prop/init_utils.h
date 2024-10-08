/*************************************************

          This script helps allocate
          and initialize memory for
          supporting information

**************************************************/

#include "cuda.h"
#include "cuda_runtime.h"
#include "device_launch_parameters.h"

#ifndef MY_SP_STRUCT
#define MY_SP_STRUCT
#include "../bass/share/my_sp_struct.h"
#endif

/*************************************************

               Allocation

**************************************************/

void throw_on_cuda_error(cudaError_t code);
void* easy_allocate(int size, int esize);


/*************************************************

                Initialize Values

**************************************************/

__host__ void init_prior_counts(superpixel_params* sp_params,
                                int* prior_counts, int* prior_map, int nprior);
__global__ void init_prior_counts_kernel(superpixel_params* sp_params,
                                         int* prior_counts, int* prior_map, int nprior);
