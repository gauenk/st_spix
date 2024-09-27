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
                             float prior_sigma_app,
                             float* img, int* spix, spix_helper* sp_helper,
                             int npix, int nspix, int nspix_buffer,
                             int nbatch, int width, int nftrs){

  // -- fill sp_params with summary statistics --
  update_params_summ(img, spix, sp_params, sp_helper,
                     npix, nspix_buffer, nbatch, width, nftrs);
  int num_block = ceil( double(nspix_buffer)/double(THREADS_PER_BLOCK) );
  dim3 ThreadPerBlock(THREADS_PER_BLOCK,1);
  dim3 BlockPerGrid(num_block,1);
  init_sp_params_kernel<<<BlockPerGrid,ThreadPerBlock>>>(sp_params, prior_sigma_app,
                                                         nspix, nspix_buffer, npix);
}

__global__ void init_sp_params_kernel(spix_params* sp_params,float prior_sigma_app,
                                      const int nspix, int nspix_buffer, int npix){
  // the label
  int k = threadIdx.x + blockIdx.x * blockDim.x;  
  if (k>=nspix_buffer) return;

  /****************************************************

           Shift Summary Statistics to Prior

  *****************************************************/

  int count = npix/(1.*nspix);
  // int count = max(sp_params[k].count,1);
  if(k<nspix) {

    // -- activate! --
    sp_params[k].valid = 1;
    sp_params[k].prior_count = npix/nspix;

    // -- appearance --
    sp_params[k].prior_mu_app = sp_params[k].mu_app;
    sp_params[k].prior_sigma_app.x = prior_sigma_app;
    sp_params[k].prior_sigma_app.y = prior_sigma_app;
    sp_params[k].prior_sigma_app.z = prior_sigma_app;
    sp_params[k].prior_mu_app_count = 1;
    sp_params[k].prior_sigma_app_count = count;
    sp_params[k].mu_app.x = 0;
    sp_params[k].mu_app.y = 0;
    sp_params[k].mu_app.z = 0;
    sp_params[k].sigma_app.x = 0;
    sp_params[k].sigma_app.y = 0;
    sp_params[k].sigma_app.z = 0;

    // -- shape --
    sp_params[k].prior_mu_shape = sp_params[k].mu_shape;
    // sp_params[k].prior_sigma_shape = sp_params[k].sigma_shape;
    sp_params[k].prior_sigma_shape.x = count*count;
    sp_params[k].prior_sigma_shape.z = count*count;
    sp_params[k].prior_sigma_shape.y = 0;
    sp_params[k].prior_mu_shape_count = 1;
    sp_params[k].prior_sigma_shape_count = count;
    sp_params[k].logdet_prior_sigma_shape = 4*log(max(count,1));
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

    // -- fixed for debugging --
    // sp_params[k].prior_sigma_shape.x = count*count;
    // sp_params[k].prior_sigma_shape.z = count*count;
    // sp_params[k].prior_sigma_shape.y = 0;
    // sp_params[k].prior_sigma_shape_count = count;
    // sp_params[k].logdet_prior_sigma_shape = 4*log(max(count,1));

  }

}



/************************************************************


    Initialize Superpixels using Spix from Previous Frame


*************************************************************/

__host__
void init_sp_params_from_past(spix_params* curr_params,spix_params* prev_params,
                              float4 rescale, int nspix,int nspix_buffer,int npix){
  int num_block = ceil( double(nspix_buffer)/double(THREADS_PER_BLOCK) );
  dim3 ThreadPerBlock(THREADS_PER_BLOCK,1);
  dim3 BlockPerGrid(num_block,1);
  init_sp_params_from_past_kernel<<<BlockPerGrid,ThreadPerBlock>>>(curr_params,
                                                                   prev_params,
                                                                   rescale, nspix,
                                                                   nspix_buffer, npix);
}

__global__
void init_sp_params_from_past_kernel(spix_params* curr_params,
                                     spix_params* prev_params,
                                     float4 rescale, int nspix,
                                     int nspix_buffer, int npix){
  // -- ... --
  int k = threadIdx.x + blockIdx.x * blockDim.x;  
  if (k>=nspix_buffer) return;

  int count = npix/(1.*nspix);
  if(k<nspix) {

    // -- activate! --
    curr_params[k].valid = 1;

    // -- unpack for reading --
    float rescale_mu_app = rescale.x;
    float rescale_sigma_app = rescale.y;
    float rescale_mu_shape = rescale.z;
    float rescale_sigma_shape = rescale.w;

    // -- appearance --
    // int count = prev_params[k].count;
    // curr_params[k].prior_mu_app = prev_params[k].mu_app;
    // curr_params[k].prior_sigma_app.x = 0.002025;
    // curr_params[k].prior_sigma_app.y = 0.002025;
    // curr_params[k].prior_sigma_app.z = 0.002025;
    // curr_params[k].prior_mu_app_count = 1;
    // curr_params[k].prior_sigma_app_count = 1;//count;
    // curr_params[k].mu_app.x = 0;
    // curr_params[k].mu_app.y = 0;
    // curr_params[k].mu_app.z = 0;
    // curr_params[k].sigma_app.x = 0;
    // curr_params[k].sigma_app.y = 0;
    // curr_params[k].sigma_app.z = 0;

    curr_params[k].prior_mu_app = prev_params[k].mu_app;
    curr_params[k].prior_sigma_app.x = prev_params[k].sigma_app.x;
    curr_params[k].prior_sigma_app.y = prev_params[k].sigma_app.y;
    curr_params[k].prior_sigma_app.z = prev_params[k].sigma_app.z;
    // curr_params[k].prior_mu_app_count = max(rescale_mu_app * count,1.0);
    // curr_params[k].prior_sigma_app_count = max(rescale_sigma_app * count,1.0);
    curr_params[k].prior_mu_app_count = 1;
    curr_params[k].prior_sigma_app_count = count;
    curr_params[k].mu_app.x = 0;
    curr_params[k].mu_app.y = 0;
    curr_params[k].mu_app.z = 0;
    curr_params[k].sigma_app.x = 0;
    curr_params[k].sigma_app.y = 0;
    curr_params[k].sigma_app.z = 0;

    // -- shape --
    curr_params[k].prior_mu_shape = prev_params[k].mu_shape;
    double logdet_shape = prev_params[k].logdet_sigma_shape;
    double det = exp(logdet_shape);

    // curr_params[k].prior_sigma_shape.x = prev_params[k].sigma_shape.z*det;
    // curr_params[k].prior_sigma_shape.y = -prev_params[k].sigma_shape.y*det;
    // curr_params[k].prior_sigma_shape.z = prev_params[k].sigma_shape.x*det;
    // curr_params[k].prior_mu_shape_count = max(rescale_mu_shape * count,1.0);
    // curr_params[k].prior_sigma_shape_count = max(rescale_sigma_shape * count,1.0);
    // curr_params[k].logdet_prior_sigma_shape = logdet_shape;

    curr_params[k].prior_sigma_shape.x = count*count;
    curr_params[k].prior_sigma_shape.z = count*count;
    curr_params[k].prior_sigma_shape.y = 0;
    curr_params[k].prior_mu_shape_count = 1;//prev_params[k].count;
    curr_params[k].prior_sigma_shape_count = count;
    curr_params[k].logdet_prior_sigma_shape = 4*log(max(count,1));

    curr_params[k].mu_shape.x = 0;
    curr_params[k].mu_shape.y = 0;
    curr_params[k].sigma_shape.x = 0;
    curr_params[k].sigma_shape.y = 0;
    curr_params[k].sigma_shape.z = 0;
    curr_params[k].logdet_sigma_shape = 0;

  }else{
    curr_params[k].count = 0;
    float3 mu_app;
    mu_app.x = -999;
    mu_app.y = -999;
    mu_app.z = -999;
    curr_params[k].mu_app = mu_app;
    double2 mu_shape;
    mu_shape.x = -999;
    mu_shape.y = -999;
    curr_params[k].mu_shape = mu_shape;
    curr_params[k].valid = 0;
  }


}
