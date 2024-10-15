#include "cuda.h"
#include "cuda_runtime.h"
#include "device_launch_parameters.h"

#ifndef MY_SP_STRUCT
#define MY_SP_STRUCT
#include "../bass/share/my_sp_struct.h"
#endif

__host__ void update_params(const float* img,const int* seg,
                            spix_params* sp_params,
                            spix_helper* sp_helper,
                            const int npix, const int nspix_buffer,
                            const int nbatch, const int width, const int nftrs);

__host__ void update_params_summ(const float* img, const int* spix,
                                 spix_params* sp_params,spix_helper* sp_helper,
                                 const int npixels, const int nspix_buffer,
                                 const int nbatch, const int width, const int nftrs);

__global__ void clear_fields(spix_params* sp_params,
                             spix_helper* sp_helper,
                             const int nsuperpixel_buffer,
                             const int nftrs);

__global__ void sum_by_label(const float* img, const int* seg,
                             spix_params* sp_params,
                             spix_helper* sp_helper,
                             const int npix, const int nbatch,
                             const int width, const int nftrs);

__global__ void calc_posterior_mode(spix_params* sp_params,
                                    spix_helper* sp_helper,
                                    const int nspix_buffer);

__global__ void calc_summ_stats(spix_params* sp_params,spix_helper* sp_helper,
                                const int nspix_buffer);
/* __global__ */
/* void calculate_mu_and_sigma(spix_params*  sp_params, */
/*                             spix_helper* sp_helper, */
/*                             const int nsuperpixel_buffer) */

/* __device__ float3 compute_mu_mode_fl(double3 mu_sum, float3 prior_mu, */
/*                                      int count, int lam); */
/* __device__ double2 compute_mu_mode_db(double2& mu, double2 prior_mu, */
/*                                       int count, int lam); */
/* __device__ double2 compute_mu_mode_db(double2 mu_sum, double2 prior_mu, */
/*                                       int count, int lam); */
/* __device__ float calc_mu_s_likelihood(double2 mu_s, */
/*                                       double2 prior_mu_s, */
/*                                       int lam, int count); */

// ---- Appearance [mean,cov] ---
__device__ float3 calc_app_mean_mode(double3 sample_sum, float3 prior_mu,
                                     int count, int prior_count);
__device__ float3 calc_app_sigma_mode(double3 sq_sample_sum, double3 sample_sum,
                                      int count, float3 prior_sigma, float3 prior_mu,
                                      int prior_count_sigma, int prior_count_mu);
__device__ double calc_app_mean_ll(float3 mu_app, float3 prior_mu, float3 prior_sigma);
__device__ double calc_app_sigma_ll(float3 sigma, float3 prior_sigma, int prior_count);

// ---- Shape [mean,cov] ---
__device__ double2 calc_shape_mean_mode(double2& mu, double2 prior_mu,
                                        int count, int lam);
__device__ double3 calc_shape_sigma_mode_simp(longlong3 sq_sum, double2 mu,
                                              double3 prior_sigma_s, double2 prior_mu,
                                              int count, int prior_count);
__device__ double3 calc_shape_sigma_mode(longlong3 sigma_s_sum, double2 mu_s,
                                       double3 prior_sigma_s, double2 prior_mu_s,
                                       int count, int lam, int df);
__device__ double calc_shape_mean_ll(double2 mu, double2 prior_mu,
                                     double3 inv_prior_sigma, double det_prior);
__device__ double calc_shape_sigma_ll(double3 sigma_s, double3 prior_sigma_s,
                                      double det_sigma, int df);

/* __device__ float calc_shape_mean_ll(double2 mu_s,double2 prior_mu_s, */
/*                                     int lam, int count); */
/* __device__ float calc_shape_cov_ll(double3 sigma_s, double3 prior_sigma_s, */
/*                                    double det_sigma, int df); */

/* // ---- Shape [mean,cov] --- */
/* __device__ float3 calc_app_mean_mode(double3 mu_sum, float3 prior_mu, */
/*                                      int count, int lam); */

/************************************************************

            Calculate the Prior Cov Likelihood

************************************************************/

__device__ double3 outer_product_term(double2 prior_mu_s, double2 mu_s,
                                      int lam, int count);
__device__ double determinant2x2(double3 sigma);
__device__ double3 inverse2x2(double3 sigma, double det);
__device__ double trace2x2(double3 inv_sigma_prior, double3 sigma);
__device__ float calc_cov_likelihood(double3 sigma_s, double3 prior_sigma_s,
                                     double det_sigma, int df);
