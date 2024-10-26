


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

/* __host__ int run_split_merge(const float* img, int* seg, */
/*                              bool* border, spix_params* sp_params, */
/*                              spix_helper* sp_helper, */
/*                              spix_helper_sm* sm_helper, */
/*                              int* sm_seg1 ,int* sm_seg2, int* sm_pairs, */
/*                              float alpha_hastings, float pix_var, */
/*                              int& count, int idx, int max_nspix, */
/*                              const int npix, const int nbatch, */
/*                              const int width, const int height, */
/*                              const int nftrs, const int nspix_buffer); */

__host__ int run_split(const float* img, int* seg, bool* border,
                       spix_params* sp_params, spix_helper* sp_helper,
                       spix_helper_sm* sm_helper,
                       int* sm_seg1 ,int* sm_seg2, int* sm_pairs,
                       float alpha_hastings, float pix_var,
                       int& count, int idx, int max_nspix,
                       const int npix, const int nbatch,
                       const int width, const int height,
                       const int nftrs, const int nspix_buffer);

__host__ void run_merge(const float* img, int* seg, bool* border,
                        spix_params* sp_params, spix_helper* sp_helper,
                        spix_helper_sm* sm_helper,
                        int* sm_seg1 ,int* sm_seg2, int* sm_pairs,
                        float alpha_hastings, float pix_var,
                        int& count, int idx, int max_nspix,
                        const int npix, const int nbatch,
                        const int width, const int height,
                        const int nftrs, const int nspix_buffer);

__host__ void CudaCalcMergeCandidate(const float* img, int* seg,
                                     bool* border, spix_params* sp_params,
                                     spix_helper* sp_helper,
                                     spix_helper_sm* sm_helper,
                                     int* sm_pairs, const int npix, const int nbatch,
                                     const int width, const int height,
                                     const int nftrs, const int nspix_buffer,
                                     const int direction, float alpha, float pix_var);

__host__ int CudaCalcSplitCandidate(const float* img, int* seg, bool* border,
                                    spix_params* sp_params,
                                    spix_helper* sp_helper,
                                    spix_helper_sm* sm_helper,
                                    int* sm_seg1, int* sm_seg2, int* sm_pairs,
                                    const int npix, const int nbatch, const int width,
                                    const int height, const int nftrs,
                                    const int nspix_buffer, int max_nspix,
                                    int direction,float alpha, float pix_var);

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

__global__ void init_sm(const float* img,
                        const int* seg_gpu,
                        spix_params* sp_params,
                        spix_helper_sm* sm_helper,
                        const int nspix_buffer, const int nbatch,
                        const int width,const int nftrs,int* sm_pairs);

/************************************************************


                       Merge Functions


*************************************************************/

__global__  void calc_merge_candidate(int* seg, bool* border, int* sm_pairs,
                                      const int npix, const int nbatch,
                                      const int width, const int height,
                                      const int change);

__global__ void sum_by_label_merge(const float* img, const int* seg_gpu,
                                   spix_params* sp_params,
                                   spix_helper_sm* sm_helper,
                                   const int npix, const int nbatch,
                                   const int width, const int nftrs);

__global__ void calc_bn_merge(int* seg, int* sm_pairs,
                              spix_params* sp_params,
                              spix_helper* sp_helper,
                              spix_helper_sm* sm_helper,
                              const int npix, const int nbatch,
                              const int width, const int nspix_buffer, float b_0);

__global__ void merge_likelihood(const float* img,int* sm_pairs,
                                 spix_params* sp_params,
                                 spix_helper* sp_helper,
                                 spix_helper_sm* sm_helper,
                                 const int npix, const int nbatch,
                                 const int width, const int nftrs,
                                 const int nspix_buffer, float a_0, float b_0);


__global__ void calc_hasting_ratio(const float* img, int* sm_pairs,
                                   spix_params* sp_params,
                                   spix_helper* sp_helper,
                                   spix_helper_sm* sm_helper,
                                   const int npix, const int nbatch, const int width,
                                   const int nftrs, const int nspix_buffer,
                                   float alpha_hasting_ratio);

__global__ void calc_hasting_ratio2(const float* img, int* sm_pairs,
                                    spix_params* sp_params,
                                    spix_helper* sp_helper,
                                    spix_helper_sm* sm_helper,
                                    const int npix, const int nbatch, const int width,
                                    const int nftrs, const int nspix_buffer,
                                    float alpha_hasting_ratio);

__global__ void remove_sp(int* sm_pairs, spix_params* sp_params,
                          spix_helper_sm* sm_helper,
                          const int nspix_buffer);

__global__ void merge_sp(int* seg, bool* border, int* sm_pairs,
                         spix_params* sp_params,
                         spix_helper_sm* sm_helper,
                         const int npix, const int nbatch,
                         const int width, const int height);

/***********************************************************


                     Split Functions


************************************************************/

__global__ void init_split(const bool* border, int* seg_gpu,
                           spix_params* sp_params,
                           spix_helper_sm* sm_helper,
                           const int nspix_buffer,
                           const int nbatch, const int width,
                           const int height, const int offset,
                           const int* seg, int* max_sp, int max_nspix);

__global__ void split_sp(int* seg, int* sm_seg1, int* sm_pairs,
                         spix_params* sp_params,
                         spix_helper_sm* sm_helper,
                         const int npix, const int nbatch,
                         const int width, const int height, int max_nspix);

__global__ void calc_split_candidate(int* dists, int* spix, bool* border,
                                     int distance, int* done_gpu, const int npix,
                                     const int nbatch, const int width, const int height);

__global__ void calc_seg_split(int* sm_seg1, int* sm_seg2, int* seg,
                               const int npix, int nbatch, int max_nspix);

__global__ void sum_by_label_split(const float* img, const int* seg,
                                   spix_params* sp_params,
                                   spix_helper_sm* sm_helper,
                                   const int npix, const int nbatch,
                                   const int width, const int nftrs, int max_nspix);

__global__
void split_marginal_likelihood(spix_params* sp_params,
                                 spix_helper_sm* sm_helper,
                                 const int npix, const int nbatch,
                                 const int width, const int nspix_buffer,
                                 float sigma2_app, int max_nspix);

__device__ double marginal_likelihood_app(double3 sum_obs,double3 sq_sum_obs,
                                          int _num_obs, double sigma2);

__global__ void calc_bn_split(int* sm_pairs,
                              spix_params* sp_params,
                              spix_helper* sp_helper,
                              spix_helper_sm* sm_helper,
                              const int npix, const int nbatch,
                              const int width, const int nspix_buffer,
                              float b_0, int max_nspix);
__global__
void split_likelihood(const float* img, int* sm_pairs,
                      spix_params* sp_params,
                      spix_helper* sp_helper,
                      spix_helper_sm* sm_helper,
                      const int npix, const int nbatch,
                      const int width, const int nftrs,
                      const int nspix_buffer,
                      float a_0, float b_0, int max_nspix);

__global__
void split_hastings_ratio(const float* img, int* sm_pairs,
                          spix_params* sp_params,
                          spix_helper* sp_helper,
                          spix_helper_sm* sm_helper,
                          const int npix, const int nbatch,
                          const int width, const int nftrs,
                          const int nspix_buffer,
                          float alpha_hasting_ratio,
                          int max_nspix, int* max_sp );
