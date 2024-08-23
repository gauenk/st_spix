#include "cuda.h"
#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#include "math_constants.h"

#ifndef MY_SP_STRUCT
#define MY_SP_STRUCT
#include "../bass/share/my_sp_struct.h"
#endif

/***********************************************************************


                    Update and Read Cov


***********************************************************************/

__device__ inline
double4 compute_inv_cov(double C00, double C01, double C11,
                        int mu_x, int mu_y, float s_prior,
                        int count, int total_count){

  if (count > 3){
    //update cumulative count and covariance
    C00 = C00 - mu_x * mu_x * count;
    C01 = C01 - mu_x * mu_y * count;
    C11 = C11 - mu_y * mu_y * count;
  }

  C00 = (s_prior + C00) / (total_count - 3.0);
  C01 = C01 / (total_count - 3);
  C11 = (s_prior + C11) / (total_count - 3.0);

  double detC = C00 * C11 - C01 * C01;
  if (detC <= 0){
      C00 = C00 + 0.00001;
      C11 = C11 + 0.00001;
      detC = C00*C11-C01*C01;
      if(detC <=0) detC = 0.0001;//hack
  }

  //Take the inverse of sigma_space to get J_space
  double4 inv_cov;
  inv_cov.x = C11 / detC;
  inv_cov.y = -C01 / detC;
  inv_cov.z = C00 / detC;
  inv_cov.w = log(detC);
  return inv_cov;
}

__device__ inline
double4 get_updated_cov(int x, int y,
                        superpixel_params* sp_params,
                        superpixel_GPU_helper* sp_gpu_helper,
                        int spix, int sign){
  float a_prior = sp_params[spix].prior_count;
  float s_prior = a_prior * a_prior;
  int count = sp_params[spix].count + sign;
  int mu_x = sp_gpu_helper[spix].mu_s_sum.x + sign*x;
  int mu_y = sp_gpu_helper[spix].mu_s_sum.y + sign*y;
  double C00 = sp_gpu_helper[spix].sigma_s_sum.x + sign*x*x;
  double C01 =  sp_gpu_helper[spix].sigma_s_sum.y + sign*x*y;
  double C11 = sp_gpu_helper[spix].sigma_s_sum.z + sign*y*y;
  double total_count = (double) count + a_prior;
  double4 cov = compute_inv_cov(C00, C01, C11, mu_x, mu_y, s_prior, count, total_count);
  return cov;
}

__device__ inline
double4 read_cov(superpixel_params* sp_params, int spix){
  double4 cov;
  cov.x = sp_params[spix].sigma_s.x;
  cov.y = sp_params[spix].sigma_s.y;
  cov.z = sp_params[spix].sigma_s.z;
  cov.w = sp_params[spix].logdet_Sigma_s;
  return cov;
}

__device__ inline
void transition_cov_probs(float& probs, double4 cov, double4 cov_prior, float df){
  /* probs = pow(cov.x - cov_prior.x,2); */
  /* probs += pow(cov.y - cov_prior.y,2); */
  /* probs += pow(cov.z - cov_prior.z,2); */
  /* probs = -1/2 * probs; // todo; add prior weight */
  // Terms independent of [df and cov] can be deleted.
  float mmtr = -1./2 * (cov_prior.x / cov.x + cov_prior.y / cov.y - 2*cov_prior.z/cov.z);
  float logdet = (df/2.) * logf(cov_prior.w) - (df+3)/2. * logf(cov.w);
  float gamma_p =  -1/2.*CUDART_LNPI_F - lgammaf(df/2.) - lgammaf((df-1)/2.);
  float Z =  - df * CUDART_LN2_F - gamma_p;
  probs = mmtr + logdet + Z;
}

__device__ inline
void xfer_case0_cov(float& xferA, float& xferC,
                    int spix_C, int spix_A,
                    int width_ix, int height_ix,
                    superpixel_params* sp_params,
                    superpixel_params* sp_params_prev,
                    superpixel_GPU_helper* sp_gpu_helper){

  // -- compute 1st mean term -- p(\theta_c) / p(\theta_c^b)
  int sign = -1;
  float df = 1;
  double4 cov_prior = read_cov(sp_params_prev,spix_C);
  double4 cov_curr = read_cov(sp_params,spix_C);
  double4 cov_prop = get_updated_cov(width_ix, height_ix,
                                     sp_params, sp_gpu_helper, spix_C, sign);
  float xferA_cov_c,xferC_cov_c;
  transition_cov_probs(xferA_cov_c, cov_curr, cov_prior, df);
  transition_cov_probs(xferC_cov_c, cov_prop, cov_prior, df);

  // -- compute 2nd mean term --
  sign = 1;
  cov_prior = read_cov(sp_params_prev,spix_A);
  cov_curr = read_cov(sp_params,spix_A);
  cov_prop = get_updated_cov(width_ix, height_ix,
                             sp_params, sp_gpu_helper, spix_A, sign);
  float xferA_cov_p, xferC_cov_p;
  transition_cov_probs(xferA_cov_p, cov_curr, cov_prior, df);
  transition_cov_probs(xferC_cov_p, cov_prop, cov_prior, df);

  // -- total mean probs --
  xferA = xferA_cov_p + xferA_cov_c; // NOT MLE updated theta
  xferC = xferC_cov_p + xferC_cov_c; // MLE updated theta
  /* xferA = 0; */
  /* xferC = 0; */

}

__device__ inline
void xfer_case1_cov(float& xferA, float& xferB,
                    int spix_A, int spix_B,
                    int width_ix, int height_ix,
                    superpixel_params* sp_params,
                    superpixel_params* sp_params_prev,
                    superpixel_GPU_helper* sp_gpu_helper){

  // -- compute 1st mean term -- p(\theta_c) / p(\theta_c^b)
  int sign = -1;
  float df = 1;
  double4 cov_prior = read_cov(sp_params_prev,spix_A);
  double4 cov_0 = read_cov(sp_params,spix_A);
  double4 cov_1 = get_updated_cov(width_ix, height_ix,
                                  sp_params, sp_gpu_helper, spix_B, sign);
  float xferA_cov_c,xferB_cov_c;
  transition_cov_probs(xferA_cov_c, cov_0, cov_prior, df);
  transition_cov_probs(xferB_cov_c, cov_1, cov_prior, df);

  // -- compute 2nd mean term --
  sign = 1;
  cov_prior = read_cov(sp_params_prev,spix_B);
  cov_0 = read_cov(sp_params,spix_B);
  cov_1 = get_updated_cov(width_ix, height_ix,
                          sp_params, sp_gpu_helper, spix_B, sign);
  float xferA_cov_p, xferB_cov_p;
  transition_cov_probs(xferA_cov_p, cov_0, cov_prior, df);
  transition_cov_probs(xferB_cov_p, cov_1, cov_prior, df);

  // -- total mean probs --
  xferA = xferA_cov_p + xferA_cov_c; // NOT MLE updated theta
  xferB = xferB_cov_p + xferB_cov_c; // MLE updated theta
  /* xferA = 0; */
  /* xferB = 0; */

}


/***********************************************************************


                    Update and Read Means


***********************************************************************/

__device__ inline
void transition_mean_probs(float& probs, float3 mu_i_prop, float3 mu_i_curr){
  float tmp = 0;
  tmp = pow(mu_i_prop.x - mu_i_curr.x,2);
  tmp += pow(mu_i_prop.y - mu_i_curr.y,2);
  tmp += pow(mu_i_prop.z - mu_i_curr.z,2);
  probs += -1/2 * tmp; // todo; add prior weight
}

__device__ inline
float3 get_updated_means(float* pix,
                         superpixel_GPU_helper* sp_gpu_helper, int spix, int sign){
  int count = sp_gpu_helper[spix].mu_i_sum.x + sign;
  float3 mu_i;
  mu_i.x = (sp_gpu_helper[spix].mu_i_sum.x + sign*pix[0])/count;
  mu_i.y = (sp_gpu_helper[spix].mu_i_sum.y + sign*pix[1])/count;
  mu_i.z = (sp_gpu_helper[spix].mu_i_sum.z + sign*pix[2])/count;
}

__device__ inline void
xfer_updated_means(float& xfer, int spix, int sign, float* pix,
                   superpixel_params* sp_params_prev,
                   superpixel_GPU_helper* sp_gpu_helper){
  float3 mu_prior = sp_params_prev[spix].mu_i;
  float3 mu = get_updated_means(pix, sp_gpu_helper, spix, sign);
  transition_mean_probs(xfer, mu, mu_prior);
}

__device__ inline void
xfer_same_means(float& xfer, int spix,
                superpixel_params* sp_params_prev,
                superpixel_params* sp_params){
  float3 mu_prior = sp_params_prev[spix].mu_i;
  float3 mu = sp_params[spix].mu_i;
  transition_mean_probs(xfer, mu, mu_prior);
}

__device__ inline void
xfer_case0_means(float& xferA, float& xferC,
                 int spix_C, int spix_A, float* pix,
                 superpixel_params* sp_params,
                 superpixel_params* sp_params_prev,
                 superpixel_GPU_helper* sp_gpu_helper){

  // -=-=-=-   updated after "pix" from "C -> A"  -=-=-=-
  xfer_updated_means(xferA,spix_A,1,pix,sp_params_prev,sp_gpu_helper);
  xfer_updated_means(xferA,spix_C,-1,pix,sp_params_prev,sp_gpu_helper);

  // -=-=-=-  no change; compute prob of current parameters under prior  -=-=-=-
  xfer_same_means(xferC,spix_A,sp_params_prev,sp_gpu_helper);
  xfer_same_means(xferC,spix_C,sp_params_prev,sp_gpu_helper);

  /* int sign = -1; */
  /* float3 mu_prior = sp_params_prev[spix_C].mu_i; */
  /* float3 mu = get_updated_means(pix, sp_gpu_helper, spix_C, sign); */
  /* transition_mean_probs(xferA, mu_A, mu_prior); */

  /* mu = sp_params[spix_C].mu_i; */
  /* transition_mean_probs(xferA_means_c, mu_C, mu_prior); */

  /* // -- compute xfer terms 1 -- */
  /* sign = 1; */
  /* mu_prior = sp_params_prev[spix_A].mu_i; */
  /* mu_C = sp_params[spix_A].mu_i; */
  /* mu_A = get_updated_means(pix, sp_gpu_helper, spix_A, sign); */
  /* float xferA_means_p, xferC_means_p; */
  /* transition_mean_probs(xferA_means_p, mu_C, mu_prior); */
  /* transition_mean_probs(xferC_means_p, mu_A, mu_prior); */

  /* // -- total mean probs -- */
  /* xferA = xferA_means_p + xferA_means_c; // NOT MLE updated theta */
  /* xferC = xferC_means_p + xferC_means_c; // MLE updated theta */

}


__device__ inline void
xfer_case1_means(float& xferA, float& xferB,
                 int spix_A, int spix_B,
                 float* pix, superpixel_params* sp_params,
                 superpixel_params* sp_params_prev,
                 superpixel_GPU_helper* sp_gpu_helper){

  // -- compute xfer term 0 -- p(\theta_c) / p(\theta_c^b)
  int sign = -1;
  float3 mu_prior = sp_params_prev[spix_C].mu_i;
  float3 mu_B = get_updated_means(pix, sp_gpu_helper, spix_B, sign);
  float3 mu_A = get_updated_means(pix, sp_gpu_helper, spix_A, sign);
  float xferA_means_c,xferB_means_c;
  transition_mean_probs(xferA_means_c, mu_C, mu_prior);
  transition_mean_probs(xferB_means_c, mu_A, mu_prior);

  // -- compute xfer term 1 --
  sign = 1;
  mu_prior = sp_params_prev[spix_A].mu_i;
  mu_C = sp_params[spix_A].mu_i;
  mu_A = get_updated_means(pix, sp_gpu_helper, spix_A, sign);
  float xferA_means_p, xferB_means_p;
  transition_mean_probs(xferA_means_p, mu_C, mu_prior);
  transition_mean_probs(xferB_means_p, mu_A, mu_prior);

  // -- total mean probs --
  xferA = xferA_means_p + xferA_means_c; // NOT MLE updated theta
  xferB = xferB_means_p + xferB_means_c; // MLE updated theta

}



/***********************************************************************


                Calculate Transition Probs after MLE


***********************************************************************/

__device__ inline void
xfer_case0(float& xferA, float& xferB,
           int spix_C, int spix_A,
           float* pix, int width_ix, int height_ix,
           superpixel_params* sp_params,
           superpixel_params* sp_params_prev,
           superpixel_GPU_helper* sp_gpu_helper){
  //
  // current spix -> A
  //      or spix -> C (stays the same)
  //
  // spix_A != spix_C but spix_B == spix_C
  // assume spix_A is "numerator"

  // means
  float xferA_means,xferB_means;
  xferA_means = xferB_means = 0;
  xfer_case0_means(xferA_means,xferB_means,
                   spix_C, spix_A, pix,
                   sp_params, sp_params_prev,
                   sp_gpu_helper);

  // covariance
  float xferA_cov,xferB_cov;
  xferA_cov = xferB_cov = 0;
  xfer_case0_cov(xferA_cov,xferB_cov,
                 spix_C, spix_A,
                 width_ix, height_ix,
                 sp_params, sp_params_prev,
                 sp_gpu_helper);

  // total
  xferA = xferA_means + xferA_cov;
  xferB = xferB_means + xferB_cov;

}

__device__ inline void
xfer_case1(float& xferA, float& xferB,
           int spix_A, int spix_B, int spix_C,
           float* pix, int width_ix, int height_ix,
           superpixel_params* sp_params,
           superpixel_params* sp_params_prev,
           superpixel_GPU_helper* sp_gpu_helper){


  // means
  float xferA_means,xferB_means;
  xferA_means = xferB_means = 0;
  xfer_case1_means(xferA_means,xferB_means,
                   spix_A, spix_B,
                   pix, sp_params, sp_params_prev,
                   sp_gpu_helper);

  // covariance
  float xferA_cov,xferB_cov;
  xferA_cov = xferB_cov = 0;
  xfer_case1_cov(xferA_cov,xferB_cov,
                 spix_A, spix_B,
                 width_ix, height_ix,
                 sp_params, sp_params_prev,
                 sp_gpu_helper);

  // total
  xferA = xferA_means + xferA_cov;
  xferB = xferB_means + xferB_cov;

}


__device__ inline void
calc_transition(float& xferA,float& xferB,
                int spix_A, int spix_B, int spix_C,
                float* pix, int width_ix, int height_ix,
                superpixel_params* sp_params,
                superpixel_params* sp_params_prev,
                superpixel_GPU_helper* sp_gpu_helper){

  // float xferA,xferB;
  if (spix_A == spix_C){
    xfer_case0(xferA,xferB,spix_A,spix_B,
               pix,width_ix,height_ix,
               sp_params,sp_params_prev,
               sp_gpu_helper);
  }else if (spix_B == spix_C){
    xfer_case0(xferB,xferA,spix_A,spix_B,
               pix,width_ix,height_ix,
               sp_params,sp_params_prev,
               sp_gpu_helper);
  }else{
    xfer_case1(xferB,xferA,
               spix_A,spix_B,spix_C,
               pix,width_ix,height_ix,
               sp_params,sp_params_prev,
               sp_gpu_helper);
  }
}

