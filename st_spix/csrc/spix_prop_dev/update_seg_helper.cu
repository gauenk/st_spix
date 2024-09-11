// #ifndef OUT_OF_BOUNDS_LABEL
// #define OUT_OF_BOUNDS_LABEL -1
// #endif

// #ifndef BAD_TOPOLOGY_LABEL 
// #define BAD_TOPOLOGY_LABEL -2
// #endif

// #ifndef NUM_OF_CHANNELS 
// #define NUM_OF_CHANNELS 3
// #endif


// #ifndef USE_COUNTS
// #define USE_COUNTS 1
// #endif


// #ifndef OUT_OF_BOUNDS_LABEL
// #define OUT_OF_BOUNDS_LABEL -1
// #endif

// #define THREADS_PER_BLOCK 512


// #include "cuda.h"
// #include "cuda_runtime.h"
// #include "device_launch_parameters.h"

// #include <assert.h>
// #include "update_prop_seg.h"
// #ifndef MY_SP_SHARE_H
// #define MY_SP_SHARE_H
// #include "../bass/share/sp.h"
// #endif

// #include "update_seg_helper.h"
// #include <cmath>
// #include <stdio.h>
// // #include <tuple>
// #ifndef WIN32
// #include <unistd.h>
// #endif



/***********************************************************************


                    Update and Read Cov


***********************************************************************/

__device__ inline
void get_updated_cov(float3& mu_i, float* pix,
                       superpixel_GPU_helper* sp_gpu_helper, int spix, int sign){
  int count = sp_gpu_helper[spix].mu_i_sum.x + sign;
  mu_i.x = (sp_gpu_helper[spix].mu_i_sum.x + sign*pix[0])/count;
  mu_i.y = (sp_gpu_helper[spix].mu_i_sum.y + sign*pix[1])/count;
  mu_i.z = (sp_gpu_helper[spix].mu_i_sum.z + sign*pix[2])/count;
}


__device__ inline 
void transition_cov_probs(float& probs, float3 cov_prop, float3 cov_curr){
  probs = pow(cov_prop.x - cov_curr.x,2);
  probs += pow(cov_prop.y - cov_curr.y,2);
  probs += pow(cov_prop.z - cov_curr.z,2);
  probs = -1/2 * probs; // todo; add prior weight
}

__device__ inline void
xfer_case0_cov(float& xfer0, float& xfer1,
               int spix_curr, int spix_prop, float* pix,
               superpixel_params* sp_params,
               superpixel_params* sp_params_prev,
               superpixel_GPU_helper* sp_gpu_helper){

  // -- compute 1st mean term -- p(\theta_c) / p(\theta_c^b)
  int sign = -1;
  float4 cov_prior,cov_curr;
  // float3 mu_prior = sp_params_prev[spix_curr].mu_i;
  // float3 mu_curr = sp_params[spix_curr].mu_i;
  float4 cov_prop;
  get_updated_cov(cov_prop, pix, sp_gpu_helper, spix_curr, sign);
  float xfer0_cov_c,xfer1_cov_c;
  transition_cov_probs(xfer0_cov_c, mu_curr, mu_prior);
  transition_cov_probs(xfer1_cov_c, mu_prop, mu_prior);

  // -- compute 2nd mean term --
  // get_updated_cov();
  sign = 1;
  mu_prior = sp_params_prev[spix_prop].mu_i;
  mu_curr = sp_params[spix_prop].mu_i;
  get_updated_cov(mu_prop, pix, sp_gpu_helper, spix_prop, sign);
  float xfer0_cov_p, xfer1_cov_p;
  transition_cov_probs(xfer0_cov_p, mu_curr, mu_prior);
  transition_cov_probs(xfer1_cov_p, mu_prop, mu_prior);

  // -- total mean probs --
  xfer0 = xfer0_cov_p + xfer0_cov_c; // NOT MLE updated theta
  xfer1 = xfer1_cov_p + xfer1_cov_c; // MLE updated theta

}

/***********************************************************************


                    Update and Read Means


***********************************************************************/


__device__ inline 
void transition_mean_probs(float& probs, float3 mu_i_prop, float3 mu_i_curr){
  probs = pow(mu_i_prop.x - mu_i_curr.x,2);
  probs += pow(mu_i_prop.y - mu_i_curr.y,2);
  probs += pow(mu_i_prop.z - mu_i_curr.z,2);
  probs = -1/2 * probs; // todo; add prior weight
}

__device__ inline
void get_updated_means(float3& mu_i, float* pix,
                       superpixel_GPU_helper* sp_gpu_helper, int spix, int sign){
  int count = sp_gpu_helper[spix].mu_i_sum.x + sign;
  mu_i.x = (sp_gpu_helper[spix].mu_i_sum.x + sign*pix[0])/count;
  mu_i.y = (sp_gpu_helper[spix].mu_i_sum.y + sign*pix[1])/count;
  mu_i.z = (sp_gpu_helper[spix].mu_i_sum.z + sign*pix[2])/count;
}

__device__ inline void
xfer_case0_means(float& xfer0, float& xfer1,
                 int spix_curr, int spix_prop, float* pix,
                 superpixel_params* sp_params,
                 superpixel_params* sp_params_prev,
                 superpixel_GPU_helper* sp_gpu_helper){

  // -- compute 1st mean term -- p(\theta_c) / p(\theta_c^b)
  int sign = -1;
  float3 mu_prior = sp_params_prev[spix_curr].mu_i;
  float3 mu_curr = sp_params[spix_curr].mu_i;
  float3 mu_prop;
  get_updated_means(mu_prop, pix, sp_gpu_helper, spix_curr, sign);
  float xfer0_means_c,xfer1_means_c;
  transition_mean_probs(xfer0_means_c, mu_curr, mu_prior);
  transition_mean_probs(xfer1_means_c, mu_prop, mu_prior);

  // -- compute 2nd mean term --
  // get_updated_cov();
  sign = 1;
  mu_prior = sp_params_prev[spix_prop].mu_i;
  mu_curr = sp_params[spix_prop].mu_i;
  get_updated_means(mu_prop, pix, sp_gpu_helper, spix_prop, sign);
  float xfer0_means_p, xfer1_means_p;
  transition_mean_probs(xfer0_means_p, mu_curr, mu_prior);
  transition_mean_probs(xfer1_means_p, mu_prop, mu_prior);

  // -- total mean probs --
  xfer0 = xfer0_means_p + xfer0_means_c; // NOT MLE updated theta
  xfer1 = xfer1_means_p + xfer1_means_c; // MLE updated theta

}


/***********************************************************************


                Calculate Transition Probs after MLE


***********************************************************************/

__device__ inline void
xfer_case0(float& xfer0, float& xfer1,
           int spix_curr, int spix_prop, float* pix,
           superpixel_params* sp_params,
           superpixel_params* sp_params_prev,
           superpixel_GPU_helper* sp_gpu_helper){



  // means
  float xfer0_means,xfer1_means;
  xfer0_means = xfer1_means = 0;
  xfer_case0_means(xfer0_means,xfer1_means,
                   spix_curr, spix_prop, pix,
                   sp_params, sp_params_prev,
                   sp_gpu_helper);

  // covariance
  float xfer0_cov,xfer1_cov;
  xfer0_cov = xfer1_cov = 0;
  xfer_case0_cov(xfer0_cov,xfer1_cov,
                 spix_curr, spix_prop, pix,
                 sp_params, sp_params_prev,
                 sp_gpu_helper);

  // total
  xfer0 = xfer0_means + xfer0_cov;
  xfer1 = xfer1_means + xfer1_cov;
}

__device__ inline void
calc_transition(float& xfer0,float& xfer1,
                int spix0, int spix1, int spix_curr,
                float* pix, superpixel_params* sp_params,
                superpixel_params* sp_params_prev,
                superpixel_GPU_helper* sp_gpu_helper){

  // float xfer0,xfer1;
  if (spix0 == spix_curr){
    xfer_case0(xfer0,xfer1,spix0,spix1,pix,
               sp_params,sp_params_prev,
               sp_gpu_helper);
  }else if (spix1 == spix_curr){
    xfer_case0(xfer1,xfer0,spix0,spix1,pix,
               sp_params,sp_params_prev,
               sp_gpu_helper);
  }else{
    // xfer0 = 0;
    // xfer0 = 0;
    // xfer0,xfer1 = xfer_case1(spix0,spix1);
    // xfer0 = cal_transition_prob_mismatch(spix0,spix1);
    // xfer1 = cal_transition_prob_mismatch(spix1,spix0);
  }

  // return std::make_tuple(xfer0,xfer1);
}

