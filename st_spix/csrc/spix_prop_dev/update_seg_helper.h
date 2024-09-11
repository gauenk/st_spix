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
  probs += mmtr + logdet + Z;
}

__device__ inline void
xfer_updated_cov(float& xfer, int spix,
                 int sign, int widx, int hidx,
                 superpixel_params* sp_params,
                 superpixel_params* sp_params_prev,
                 superpixel_GPU_helper* sp_gpu_helper){
  float df = sp_params_prev[spix].count;
  double4 cov_prior = read_cov(sp_params_prev,spix);
  double4 cov = get_updated_cov(widx, hidx, sp_params, sp_gpu_helper, spix, sign);
  transition_cov_probs(xfer, cov, cov_prior, df);
}

__device__ inline void
xfer_same_cov(float& xfer, int spix,
              superpixel_params* sp_params,
              superpixel_params* sp_params_prev){
  /* float3 mu_prior = sp_params_prev[spix].mu_i; */
  /* float3 mu = sp_params[spix].mu_i; */
  /* transition_mean_probs(xfer, mu, mu_prior); */
  float df = sp_params_prev[spix].count;
  double4 cov_prior = read_cov(sp_params_prev,spix);
  double4 cov = read_cov(sp_params,spix);
  transition_cov_probs(xfer, cov, cov_prior, df);
}

__device__ inline
void xfer_case0_cov(float& xferA, float& xferB,
                    int spix_A, int spix_B,
                    int width_ix, int height_ix,
                    superpixel_params* sp_params,
                    superpixel_params* sp_params_prev,
                    superpixel_GPU_helper* sp_gpu_helper){

  // -=-=-=-   update "widx,hidx" from "C=B -> A"  -=-=-=-
  xfer_updated_cov(xferA,spix_A,1,width_ix,height_ix,
                   sp_params,sp_params_prev,sp_gpu_helper);
  xfer_updated_cov(xferA,spix_B,-1,width_ix,height_ix,
                   sp_params,sp_params_prev,sp_gpu_helper);

  // -=-=-=-  no change (B=C); compute prob of current parameters under prior  -=-=-=-
  xfer_same_cov(xferB,spix_A,sp_params,sp_params_prev);
  xfer_same_cov(xferB,spix_B,sp_params,sp_params_prev);

}

__device__ inline
void xfer_case1_cov(float& xferA, float& xferB,
                    int spix_A, int spix_B,
                    int width_ix, int height_ix,
                    superpixel_params* sp_params,
                    superpixel_params* sp_params_prev,
                    superpixel_GPU_helper* sp_gpu_helper){

  // -=-=-=-   updated after "pix" from "C -> A"  -=-=-=-
  xfer_updated_cov(xferA,spix_A,1,width_ix,height_ix,
                   sp_params,sp_params_prev,sp_gpu_helper);
  xfer_same_cov(xferA,spix_B,sp_params,sp_params_prev);

  // -=-=-=-   updated after "pix" from "C -> B"  -=-=-=-
  xfer_same_cov(xferB,spix_A,sp_params,sp_params_prev);
  xfer_updated_cov(xferB,spix_B,1,width_ix,height_ix,
                   sp_params,sp_params_prev,sp_gpu_helper);

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
                superpixel_params* sp_params,
                superpixel_params* sp_params_prev){
  float3 mu_prior = sp_params_prev[spix].mu_i;
  float3 mu = sp_params[spix].mu_i;
  transition_mean_probs(xfer, mu, mu_prior);
}

__device__ inline void
xfer_case0_means(float& xferA, float& xferB,
                 int spix_A, int spix_B, float* pix,
                 superpixel_params* sp_params,
                 superpixel_params* sp_params_prev,
                 superpixel_GPU_helper* sp_gpu_helper){
  // -=-=-=-   updated after "pix" from "C=B -> A"  -=-=-=-
  xfer_updated_means(xferA,spix_A,1,pix,sp_params_prev,sp_gpu_helper);
  xfer_updated_means(xferA,spix_B,-1,pix,sp_params_prev,sp_gpu_helper);

  // -=-=-=-  no change (B=C); compute prob of current parameters under prior  -=-=-=-
  xfer_same_means(xferB,spix_A,sp_params,sp_params_prev);
  xfer_same_means(xferB,spix_B,sp_params,sp_params_prev);
}

__device__ inline void
xfer_case1_means(float& xferA, float& xferB,
                 int spix_A, int spix_B, float* pix,
                 superpixel_params* sp_params,
                 superpixel_params* sp_params_prev,
                 superpixel_GPU_helper* sp_gpu_helper){
  // -=-=-=-   updated after "pix" from "C -> A"  -=-=-=-
  xfer_updated_means(xferA,spix_A,1,pix,sp_params_prev,sp_gpu_helper);
  xfer_same_means(xferA,spix_B,sp_params,sp_params_prev);

  // -=-=-=-   updated after "pix" from "C -> B"  -=-=-=-
  xfer_same_means(xferB,spix_A,sp_params,sp_params_prev);
  xfer_updated_means(xferB,spix_B,1,pix,sp_params_prev,sp_gpu_helper);

}



/***********************************************************************


                Calculate Transition Probs after MLE


***********************************************************************/

__device__ inline void
xfer_case0(float& xferA, float& xferB,
           int spix_A, int spix_B,
           float* pix, int width_ix, int height_ix,
           superpixel_params* sp_params,
           superpixel_params* sp_params_prev,
           superpixel_GPU_helper* sp_gpu_helper){
  // A != C but B = C

  // means
  xfer_case0_means(xferA,xferB, spix_A, spix_B, pix,
                   sp_params, sp_params_prev, sp_gpu_helper);
  // covariance
  xfer_case0_cov(xferA,xferB, spix_A, spix_B, width_ix, height_ix,
                 sp_params, sp_params_prev, sp_gpu_helper);

}

__device__ inline void
xfer_case1(float& xferA, float& xferB,
           int spix_A, int spix_B,
           float* pix, int width_ix, int height_ix,
           superpixel_params* sp_params,
           superpixel_params* sp_params_prev,
           superpixel_GPU_helper* sp_gpu_helper){
  // A != C and B != C

  // means
  xfer_case1_means(xferA,xferB, spix_A, spix_B, pix,
                   sp_params, sp_params_prev, sp_gpu_helper);
  // covariance
  xfer_case1_cov(xferA,xferB, spix_A, spix_B, width_ix, height_ix,
                 sp_params, sp_params_prev, sp_gpu_helper);
}


__device__ inline void
calc_transition(float& xferA,float& xferB,
                int spix_A, int spix_B, int spix_C,
                float* pix, int width_ix, int height_ix,
                superpixel_params* sp_params,
                superpixel_params* sp_params_prev,
                superpixel_GPU_helper* sp_gpu_helper){

  if (spix_B == spix_C){ // when called, we know A != C
    xfer_case0(xferA,xferB,spix_A,spix_B,
               pix,width_ix,height_ix,
               sp_params,sp_params_prev,
               sp_gpu_helper);
  }else if (spix_A == spix_C){ // when called, we know B != C
    xfer_case0(xferB,xferA,spix_B,spix_A,
               pix,width_ix,height_ix,
               sp_params,sp_params_prev,
               sp_gpu_helper);
  }else{ // when called, we know A != C and B != C
    xfer_case1(xferB,xferA,spix_A,spix_B,
               pix,width_ix,height_ix,
               sp_params,sp_params_prev,
               sp_gpu_helper);
  }
}

