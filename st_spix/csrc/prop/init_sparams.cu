/*************************************************

        This script initializes spix_params
            WITH and WITHOUT propogation

**************************************************/

#include "pch.h"
#include "init_sparams.h"
#include "update_params.h"
#include <thrust/system_error.h>
#include <thrust/system/cuda/error.h>


/*************************************************

      Initialize Empty (but Valid) Superpixels

**************************************************/

__host__ void init_sp_params(spix_params* sp_params,
                             float* img, int* spix, spix_helper* sp_helper,
                             int npix, int nspix, int nspix_buffer,
                             int nbatch, int width, int nftrs){

  // -- fill sp_params with summary statistics --
  update_params_summ(img, spix, sp_params, sp_helper,
                     npix, nspix_buffer, nbatch, width, nftrs);
  int num_block = ceil( double(nspix_buffer)/double(THREADS_PER_BLOCK) );
  dim3 ThreadPerBlock(THREADS_PER_BLOCK,1);
  dim3 BlockPerGrid(num_block,1);
  init_sp_params_kernel<<<BlockPerGrid,ThreadPerBlock>>>(sp_params, nspix,
                                                         nspix_buffer, npix);
}

__global__ void init_sp_params_kernel(spix_params* sp_params,
                                      const int nspix, int nspix_buffer, int npix){
  // the label
  int k = threadIdx.x + blockIdx.x * blockDim.x;  
  if (k>=nspix_buffer) return;

  /****************************************************

           Shift Summary Statistics to Prior

  *****************************************************/

  if(k<nspix) {

    // -- activate! --
    sp_params[k].valid = 1;

    // -- appearance --
    sp_params[k].prior_mu_app = sp_params[k].mu_app;
    sp_params[k].prior_sigma_app = sp_params[k].sigma_app;
    sp_params[k].prior_mu_app_count = 1;
    sp_params[k].prior_sigma_app_count = sp_params[k].count;
    sp_params[k].mu_app.x = 0;
    sp_params[k].mu_app.y = 0;
    sp_params[k].mu_app.z = 0;
    sp_params[k].sigma_app.x = 0;
    sp_params[k].sigma_app.y = 0;
    sp_params[k].sigma_app.z = 0;

    // -- shape --
    sp_params[k].prior_mu_shape = sp_params[k].mu_shape;
    sp_params[k].prior_sigma_shape = sp_params[k].sigma_shape;
    sp_params[k].prior_mu_shape_count = 1;
    sp_params[k].prior_sigma_shape_count = sp_params[k].count;
    sp_params[k].logdet_prior_sigma_shape = sp_params[k].logdet_sigma_shape;
    sp_params[k].mu_shape.x = 0;
    sp_params[k].mu_shape.y = 0;
    sp_params[k].sigma_shape.x = 0;
    sp_params[k].sigma_shape.y = 0;
    sp_params[k].sigma_shape.z = 0;
    sp_params[k].logdet_sigma_shape = 0;
  }else{
    sp_params[k].count = 0;
    float3 mu_app;
    mu_app.x = -999;
    mu_app.y = -999;
    mu_app.z = -999;
    sp_params[k].mu_app = mu_app;
    double2 mu_shape;
    mu_shape.x = -999;
    mu_shape.y = -999;
    sp_params[k].mu_shape = mu_shape;
    sp_params[k].valid = 0;
  }

}
