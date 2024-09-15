/*************************************************

          This script helps allocate
          and initialize memory for
          supporting information

**************************************************/

#include "init_utils.h"
#include <thrust/system_error.h>
#include <thrust/system/cuda/error.h>
#define THREADS_PER_BLOCK 512

/*************************************************

               Allocation

**************************************************/

void throw_on_cuda_error_prop(cudaError_t code){ // new name since two .so objects (ugh)
  if(code != cudaSuccess){
    throw thrust::system_error(code, thrust::cuda_category());
  }
}

void* easy_allocate(int size, int esize){
  void* mem;
  try {
    throw_on_cuda_error_prop(cudaMalloc((void**)&mem,size*esize));
  }
  catch (thrust::system_error& e) {
    std::cerr << "CUDA error after cudaMalloc: " << e.what() << std::endl;
  }
  return mem;
}

/*************************************************

                Initialize Values

**************************************************/

__host__ void init_sp_params(superpixel_params* sp_params, const int sp_size,
                             const int nspix, int nspix_buffer, int npix){
  int num_block = ceil( double(nspix_buffer)/double(THREADS_PER_BLOCK) ); //Roy- TO Change
  dim3 ThreadPerBlock(THREADS_PER_BLOCK,1);
  dim3 BlockPerGrid(num_block,1);
  init_sp_params_kernel<<<BlockPerGrid,ThreadPerBlock>>>(sp_params,sp_size,nspix,
                                                         nspix_buffer, npix);
}

__global__ void init_sp_params_kernel(superpixel_params* sp_params, const int sp_size,
                                      const int nspix, int nspix_buffer, int npix)
{
  // the label
  int k = threadIdx.x + blockIdx.x * blockDim.x;  
  if (k>=nspix_buffer) return;
  double sp_size_square = double(sp_size) * double(sp_size); 

  // calculate the inverse of covariance
  double3 sigma_s_local;
  sigma_s_local.x = 1.0/sp_size_square;
  sigma_s_local.y = 0.0;
  sigma_s_local.z = 1.0/sp_size_square;
  sp_params[k].sigma_s = sigma_s_local;
  sp_params[k].prior_count = npix/nspix;
  if(k>=nspix) {
    sp_params[k].count = 0;
    float3 mu_i;
    mu_i.x = -999;
    mu_i.y = -999;
    mu_i.z = -999;
    sp_params[k].mu_i = mu_i;
    double2 mu_s;
    mu_s.x = -999;
    mu_s.y = -999;
    sp_params[k].mu_s = mu_s;
    sp_params[k].valid = 0;
  }
  else {
    sp_params[k].valid = 1;
  }

  // calculate the log of the determinant of covariance
  sp_params[k].logdet_Sigma_s = log(sp_size_square * sp_size_square);  

}