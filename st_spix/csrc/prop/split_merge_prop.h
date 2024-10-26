


#include "cuda.h"
#include "cuda_runtime.h"
#include "device_launch_parameters.h"
/* #include <torch/torch.h> */
/* #include <torch/types.h> */

// -- "external" import --
#ifndef MY_SP_STRUCT
#define MY_SP_STRUCT
#include "../bass/share/my_sp_struct.h"
#endif


/************************************************************


                         API


*************************************************************/

__host__
int run_split_prop(const float* img, int* seg, bool* border,
                   spix_params* sp_params, spix_helper* sp_helper,
                   spix_helper_sm* sm_helper,
                   int* sm_seg1 ,int* sm_seg2, int* sm_pairs,
                   float alpha_hastings, float pix_var,
                   int& count, int idx, int max_nspix,
                   const int npix, const int nbatch,
                   const int width, const int height,
                   const int nftrs, const int nspix_buffer);

/* __host__ int run_split_prop(const float* img, int* seg, */
/*                               bool* border, spix_params* sp_params, */
/*                               /\* spix_params* prior_params, int* prior_map, *\/ */
/*                               spix_helper* sp_helper, */
/*                               spix_helper_sm* sm_helper, */
/*                               int* sm_seg1 ,int* sm_seg2, int* sm_pairs, */
/*                               float alpha_hastings, float pix_var, */
/*                               int& count, int idx, int max_nspix, */
/*                               const int npix, const int nbatch, */
/*                               const int width, const int height, */
/*                               const int nftrs, const int nspix_buffer); */

__host__ void run_merge_prop(const float* img, int* seg,
                              bool* border, spix_params* sp_params,
                              spix_params* prior_params, int* prior_map,
                              spix_helper* sp_helper,
                              spix_helper_sm* sm_helper,
                              int* sm_seg1 ,int* sm_seg2, int* sm_pairs,
                              float alpha_hastings, float pix_var,
                              int& count, int idx, int max_nspix,
                              const int npix, const int nbatch,
                              const int width, const int height,
                              const int nftrs, const int nspix_buffer);

__host__ void CudaCalcMergeCandidate_p(const float* img, int* seg,
                                       bool* border, spix_params* sp_params,
                                       spix_helper* sp_helper,
                                       spix_helper_sm* sm_helper,
                                       int* sm_pairs, const int npix, const int nbatch,
                                       const int width, const int height,
                                       const int nftrs, const int nspix_buffer,
                                       const int direction,
                                       float alpha, float pix_var);

__host__ int CudaCalcSplitCandidate_p(const float* img, int* seg, bool* border,
                                      spix_params* sp_params,
                                      spix_helper* sp_helper,
                                      spix_helper_sm* sm_helper,
                                      int* sm_seg1, int* sm_seg2, int* sm_pairs,
                                      const int npix, const int nbatch, const int width,
                                      const int height, const int nftrs,
                                      const int nspix_buffer, int max_nspix,
                                      int direction, int oldnew_choice,
                                      float alpha, float pix_var);

/* __host__ int CudaCalcSplitCandidate(const float* img, int* seg, */
/*                                     bool* border,  spix_params* sp_params, */
/*                                     spix_helper* sp_helper, */
/*                                     spix_helper_sm* sm_helper, */
/*                                     int* sm_seg1, int* sm_seg2, int* sm_pairs, */
/*                                     const int npix, const int nbatch, */
/*                                     const int width, const int height, */
/*                                     const int nftrs, const int nspix_buffer, */
/*                                     int max_nspix, */
/*                                     int direction, float alpha); */

__global__ void init_sm_p(const float* img,
                          const int* seg_gpu,
                          spix_params* sp_params,
                          spix_helper_sm* sm_helper,
                          bool* split_dir,
                          const int nspix_buffer, const int nbatch,
                          const int width,const int nftrs,int* sm_pairs);

/************************************************************


                       Merge Functions


*************************************************************/

__global__  void calc_merge_candidate_p(int* seg, bool* border, int* sm_pairs,
                                      const int npix, const int nbatch,
                                      const int width, const int height,
                                      const int change);

__global__ void sum_by_label_merge_p(const float* img, const int* seg_gpu,
                                   spix_params* sp_params,
                                   spix_helper_sm* sm_helper,
                                   const int npix, const int nbatch,
                                   const int width, const int nftrs);

__global__
void split_marginal_likelihood_p(spix_params* sp_params,
                                spix_helper_sm* sm_helper,
                                const int npix, const int nbatch,
                                const int width, const int nspix_buffer,
                                float sigma2_app, int max_nspix);

__global__ void calc_bn_merge_p(int* seg, int* sm_pairs,
                              spix_params* sp_params,
                              spix_helper* sp_helper,
                              spix_helper_sm* sm_helper,
                              const int npix, const int nbatch,
                              const int width, const int nspix_buffer, float b_0);

__global__ void merge_likelihood_p(const float* img,int* sm_pairs,
                                 spix_params* sp_params,
                                 spix_helper* sp_helper,
                                 spix_helper_sm* sm_helper,
                                 const int npix, const int nbatch,
                                 const int width, const int nftrs,
                                 const int nspix_buffer, float a_0, float b_0);


__global__ void calc_hasting_ratio_p(const float* img, int* sm_pairs,
                                   spix_params* sp_params,
                                   spix_helper* sp_helper,
                                   spix_helper_sm* sm_helper,
                                   const int npix, const int nbatch, const int width,
                                   const int nftrs, const int nspix_buffer,
                                   float alpha_hasting_ratio);

__global__ void calc_hasting_ratio2_p(const float* img, int* sm_pairs,
                                    spix_params* sp_params,
                                    spix_helper* sp_helper,
                                    spix_helper_sm* sm_helper,
                                    const int npix, const int nbatch, const int width,
                                    const int nftrs, const int nspix_buffer,
                                    float alpha_hasting_ratio);

__global__ void remove_sp_p(int* sm_pairs, spix_params* sp_params,
                          spix_helper_sm* sm_helper,
                          const int nspix_buffer);

__global__ void merge_sp_p(int* seg, bool* border, int* sm_pairs,
                         spix_params* sp_params,
                         spix_helper_sm* sm_helper,
                         const int npix, const int nbatch,
                         const int width, const int height);

/***********************************************************


                     Split Functions


************************************************************/

__global__ void init_split_p(const bool* border, int* seg_gpu,
                           spix_params* sp_params,
                           spix_helper_sm* sm_helper,
                             bool* split_dir,
                           const int nspix_buffer,
                           const int nbatch, const int width,
                           const int height, const int offset,
                           const int* seg, int* max_sp, int max_nspix);

__global__ void split_sp_p(int* seg, int* sm_seg1, int* sm_pairs,
                         spix_params* sp_params,
                           spix_helper_sm* sm_helper, bool* split_dir,
                         const int npix, const int nbatch,
                         const int width, const int height, int max_nspix);

__global__ void calc_split_candidate_p(int* dists, int* spix, bool* border,
                                     int distance, int* done_gpu, const int npix,
                                     const int nbatch, const int width, const int height);

__global__ void calc_seg_split_p(int* sm_seg1, int* sm_seg2, int* seg,
                                 int oldnew_choice, const int npix,
                                 int nbatch, int max_nspix);

__global__ void sum_by_label_split_p(const float* img, const int* seg,
                                   spix_params* sp_params,
                                   spix_helper_sm* sm_helper,
                                   const int npix, const int nbatch,
                                   const int width, const int nftrs, int max_nspix);

__global__ void calc_bn_split_p(int* sm_pairs,
                                spix_params* sp_params,
                                spix_helper* sp_helper,
                                spix_helper_sm* sm_helper,
                                int oldnew_direction,
                                const int npix, const int nbatch,
                                const int width, const int nspix_buffer,
                                float b_0, float sigma2_app, int max_nspix);

__global__ void split_likelihood_p(const float* img, int* sm_pairs,
                                   spix_params* sp_params,
                                   spix_helper* sp_helper,
                                   spix_helper_sm* sm_helper,
                                   const int npix, const int nbatch,
                                   const int width, const int nftrs,
                                   const int nspix_buffer,
                                   float a_0, float b_0, int max_nspix);

__global__
void split_hastings_ratio_p(const float* img, int* sm_pairs,
                          spix_params* sp_params,
                          spix_helper* sp_helper,
                          spix_helper_sm* sm_helper,
                            bool* split_dir,
                          const int npix, const int nbatch,
                          const int width, const int nftrs,
                          const int nspix_buffer,
                          float alpha_hasting_ratio,
                          int max_nspix, int* max_sp );

__device__ float3 calc_app_mean_mode_sm(double3 sample_sum, float3 prior_mu,
                                        int count, int prior_count);
__device__ double2 calc_shape_sample_mean_sm(int2 sum_shape, int count);
__device__ double2 calc_shape_mean_mode_sm(double2& mu, double2 prior_mu,
                                           int count, int prior_count);
__device__ double3 calc_shape_sigma_mode_sm(longlong3 sq_sum, double2 mu,
                                            double3 prior_sigma, double2 prior_mu,
                                            int count, int prior_count);
__device__ double3 outer_product_term_sm(double2 prior_mu, double2 mu,
                                      int obs_count, int prior_count);
__device__ double determinant2x2_sm(double3 sigma);

__device__ double marginal_likelihood_app_sm(double3 sum_obs,double3 sq_sum_obs,
                                             float3 prior_mu,int _num_obs,
                                             int _num_prior, float sigma2);
/* __device__ double marginal_likelihood_app_sm(double3 sum_obs,double3 sq_sum_obs, */
/*                                              float3 prior_mu,int _num_obs, */
/*                                              int _num_prior, double sigma2); */
