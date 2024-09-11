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
#include "../share/my_sp_struct.h"
#endif

/*************************************************

               Allocation

**************************************************/

bool* allocate_border(int size){
    bool* border;
    try {
      throw_on_cuda_error(cudaMalloc((void**)&border,size*sizeof(bool)));
      // throw_on_cuda_error(malloc((void*)num_neg_cpu,sizeofint));
    }
    catch (thrust::system_error& e) {
        std::cerr << "CUDA error after cudaMalloc: " << e.what() << std::endl;
    }
    return border;
}

superpixel_params* allocate_sp_params(int size){
    superpixel_params* sp_params;
    try {
      throw_on_cuda_error(cudaMalloc((void**)&sp_params,size*sizeof(superpixel_params)));
    }
    catch (thrust::system_error& e) {
        std::cerr << "CUDA error after cudaMalloc: " << e.what() << std::endl;
    }
    return sp_params;
}

superpixel_GPU_helper* allocate_sp_helper(int size){
  superpixel_GPU_helper* sp_helper;
  try {
    throw_on_cuda_error(cudaMalloc((void**)&sp_helper,size*sizeof(superpixel_GPU_helper)));
  }
  catch (thrust::system_error& e) {
    std::cerr << "CUDA error after cudaMalloc: " << e.what() << std::endl;
  }
  return sp_helper;
}

superpixel_GPU_helper_sm* allocate_sm_helper(int size){
  superpixel_GPU_helper_sm* sm_helper;
  const int elem_size = sizeof(superpixel_GPU_helper_sm);
  try {
    throw_on_cuda_error(cudaMalloc((void**)&sm_helper,size*elem_size));
  }
  catch (thrust::system_error& e) {
    std::cerr << "CUDA error after cudaMalloc: " << e.what() << std::endl;
  }
  return sm_helper;
}




/*************************************************

                Initialize Values

**************************************************/

__host__ void init_sp_params(superpixel_params* sp_params, const int sp_size,
                             const int nspix, int nspix_buffer, int npix);
__global__ void init_sp_params_kernel(superpixel_params* sp_params, const int sp_size,
                                      const int nspix, int nspix_buffer, int npix);

