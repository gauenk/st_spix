/*************************************************

          Initialize Simple Superpixels

**************************************************/

#include "pch.h"
#include "simple_init_sparams.h"
#include <thrust/system_error.h>
#include <thrust/system/cuda/error.h>

/**********************************************************

      Initialize Empty (but Valid) "Simple" Superpixels

***********************************************************/

__host__ void init_sp_params_s(superpixel_params* sp_params, const int sp_size,
                               const int nspix, int nspix_buffer, int npix){
  int num_block = ceil( double(nspix_buffer)/double(THREADS_PER_BLOCK) ); //Roy- TO Change
  dim3 ThreadPerBlock(THREADS_PER_BLOCK,1);
  dim3 BlockPerGrid(num_block,1);
  init_sp_params_kernel_s<<<BlockPerGrid,ThreadPerBlock>>>(sp_params,sp_size,nspix,
                                                         nspix_buffer, npix);
}

__global__ void init_sp_params_kernel_s(superpixel_params* sp_params, const int sp_size,
                                      const int nspix, int nspix_buffer, int npix){
  // the label
  int k = threadIdx.x + blockIdx.x * blockDim.x;  
  if (k>=nspix_buffer) return;
  double sp_size_square = double(sp_size) * double(sp_size); 

  // calculate the inverse of covariance
  double3 sigma_s_local;
  sigma_s_local.x = sp_size;
  sigma_s_local.y = 0.0;
  sigma_s_local.z = sp_size;
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
