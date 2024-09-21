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

__global__ void clear_fields(spix_params* sp_params,
                             spix_helper* sp_helper,
                             const int nsuperpixel_buffer,
                             const int nftrs);

__global__ void sum_by_label(const float* img, const int* seg,
                             spix_params* sp_params,
                             spix_helper* sp_helper,
                             const int npix, const int nbatch,
                             const int width, const int nftrs);

__global__ void calculate_mu_and_sigma(spix_params* sp_params,
                                       spix_helper* sp_helper,
                                       const int nspix_buffer);
/* __global__ */
/* void calculate_mu_and_sigma(spix_params*  sp_params, */
/*                             spix_helper* sp_helper, */
/*                             const int nsuperpixel_buffer) */

/************************************************************

            Calculate the Prior Cov Likelihood

************************************************************/

__device__ double3 compute_cov_mode(longlong3 sigma_s_sum, double2 mu_s,
                                    double3 prior_sigma_s, double2 prior_mu_s,
                                    int count, int lam, int df);
__device__ double3 outer_product_term(double2 prior_mu_s, double2 mu_s,
                                      int lam, int count);
__device__ double determinant2x2(double3 sigma);
__device__ double3 inverse2x2(double3 sigma, double det);
__device__ double trace2x2(double3 inv_sigma_prior, double3 sigma);
__device__ float calc_cov_likelihood(double3 sigma_s, double3 prior_sigma_s,
                                     double det_sigma, int df);
