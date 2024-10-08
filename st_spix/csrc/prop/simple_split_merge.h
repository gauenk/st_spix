


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

__host__ int run_simple_split(const float* img, int* seg,
                              bool* border, superpixel_params* sp_params,
                              superpixel_params* prior_params, int* prior_map,
                              superpixel_GPU_helper* sp_helper,
                              superpixel_GPU_helper_sm* sm_helper,
                              int* sm_seg1 ,int* sm_seg2, int* sm_pairs,
                              float alpha_hastings, float pix_var,
                              int& count, int idx, int max_nspix,
                              const int npix, const int nbatch,
                              const int width, const int height,
                              const int nftrs, const int nspix_buffer);

__host__ void run_simple_merge(const float* img, int* seg,
                              bool* border, superpixel_params* sp_params,
                              superpixel_params* prior_params, int* prior_map,
                              superpixel_GPU_helper* sp_helper,
                              superpixel_GPU_helper_sm* sm_helper,
                              int* sm_seg1 ,int* sm_seg2, int* sm_pairs,
                              float alpha_hastings, float pix_var,
                              int& count, int idx, int max_nspix,
                              const int npix, const int nbatch,
                              const int width, const int height,
                              const int nftrs, const int nspix_buffer);

__host__ void CudaCalcSimpleMergeCandidate(const float* img, int* seg,
                                     bool* border, superpixel_params* sp_params,
                                     superpixel_GPU_helper* sp_helper,
                                     superpixel_GPU_helper_sm* sm_helper,
                                     int* sm_pairs, const int npix, const int nbatch,
                                     const int width, const int height,
                                     const int nftrs, const int nspix_buffer,
                                     const int direction, float alpha, float pix_var);

__host__ int CudaCalcSimpleSplitCandidate(const float* img, int* seg, bool* border,
                                    superpixel_params* sp_params,
                                    superpixel_GPU_helper* sp_helper,
                                    superpixel_GPU_helper_sm* sm_helper,
                                    int* sm_seg1, int* sm_seg2, int* sm_pairs,
                                    const int npix, const int nbatch, const int width,
                                    const int height, const int nftrs,
                                    const int nspix_buffer, int max_nspix,
                                    int direction,float alpha, float pix_var);

/* __host__ int CudaCalcSplitCandidate(const float* img, int* seg, */
/*                                     bool* border,  superpixel_params* sp_params, */
/*                                     superpixel_GPU_helper* sp_helper, */
/*                                     superpixel_GPU_helper_sm* sm_helper, */
/*                                     int* sm_seg1, int* sm_seg2, int* sm_pairs, */
/*                                     const int npix, const int nbatch, */
/*                                     const int width, const int height, */
/*                                     const int nftrs, const int nspix_buffer, */
/*                                     int max_nspix, */
/*                                     int direction, float alpha); */

__global__ void init_sm_s(const float* img,
                        const int* seg_gpu,
                        superpixel_params* sp_params,
                        superpixel_GPU_helper_sm* sm_helper,
                        const int nspix_buffer, const int nbatch,
                        const int width,const int nftrs,int* sm_pairs);

/************************************************************


                       Merge Functions


*************************************************************/

__global__  void calc_merge_candidate_s(int* seg, bool* border, int* sm_pairs,
                                      const int npix, const int nbatch,
                                      const int width, const int height,
                                      const int change);

__global__ void sum_by_label_merge_s(const float* img, const int* seg_gpu,
                                   superpixel_params* sp_params,
                                   superpixel_GPU_helper_sm* sm_helper,
                                   const int npix, const int nbatch,
                                   const int width, const int nftrs);

__global__ void calc_bn_merge_s(int* seg, int* sm_pairs,
                              superpixel_params* sp_params,
                              superpixel_GPU_helper* sp_helper,
                              superpixel_GPU_helper_sm* sm_helper,
                              const int npix, const int nbatch,
                              const int width, const int nspix_buffer, float b_0);

__global__ void merge_likelihood_s(const float* img,int* sm_pairs,
                                 superpixel_params* sp_params,
                                 superpixel_GPU_helper* sp_helper,
                                 superpixel_GPU_helper_sm* sm_helper,
                                 const int npix, const int nbatch,
                                 const int width, const int nftrs,
                                 const int nspix_buffer, float a_0, float b_0);


__global__ void calc_hasting_ratio_s(const float* img, int* sm_pairs,
                                   superpixel_params* sp_params,
                                   superpixel_GPU_helper* sp_helper,
                                   superpixel_GPU_helper_sm* sm_helper,
                                   const int npix, const int nbatch, const int width,
                                   const int nftrs, const int nspix_buffer,
                                   float alpha_hasting_ratio);

__global__ void calc_hasting_ratio2_s(const float* img, int* sm_pairs,
                                    superpixel_params* sp_params,
                                    superpixel_GPU_helper* sp_helper,
                                    superpixel_GPU_helper_sm* sm_helper,
                                    const int npix, const int nbatch, const int width,
                                    const int nftrs, const int nspix_buffer,
                                    float alpha_hasting_ratio);

__global__ void remove_sp_s(int* sm_pairs, superpixel_params* sp_params,
                          superpixel_GPU_helper_sm* sm_helper,
                          const int nspix_buffer);

__global__ void merge_sp_s(int* seg, bool* border, int* sm_pairs,
                         superpixel_params* sp_params,
                         superpixel_GPU_helper_sm* sm_helper,
                         const int npix, const int nbatch,
                         const int width, const int height);

/***********************************************************


                     Split Functions


************************************************************/

__global__ void init_split_s(const bool* border, int* seg_gpu,
                           superpixel_params* sp_params,
                           superpixel_GPU_helper_sm* sm_helper,
                           const int nspix_buffer,
                           const int nbatch, const int width,
                           const int height, const int offset,
                           const int* seg, int* max_sp, int max_nspix);

__global__ void split_sp_s(int* seg, int* sm_seg1, int* sm_pairs,
                         superpixel_params* sp_params,
                         superpixel_GPU_helper_sm* sm_helper,
                         const int npix, const int nbatch,
                         const int width, const int height, int max_nspix);

__global__ void calc_split_candidate_s(int* dists, int* spix, bool* border,
                                     int distance, int* done_gpu, const int npix,
                                     const int nbatch, const int width, const int height);

__global__ void calc_seg_split_s(int* sm_seg1, int* sm_seg2, int* seg,
                               const int npix, int nbatch, int max_nspix);

__global__ void sum_by_label_split_s(const float* img, const int* seg,
                                   superpixel_params* sp_params,
                                   superpixel_GPU_helper_sm* sm_helper,
                                   const int npix, const int nbatch,
                                   const int width, const int nftrs, int max_nspix);

__global__ void calc_bn_split_s(int* sm_pairs,
                              superpixel_params* sp_params,
                              superpixel_GPU_helper* sp_helper,
                              superpixel_GPU_helper_sm* sm_helper,
                              const int npix, const int nbatch,
                              const int width, const int nspix_buffer,
                              float b_0, int max_nspix);
__global__
void split_likelihood_s(const float* img, int* sm_pairs,
                      superpixel_params* sp_params,
                      superpixel_GPU_helper* sp_helper,
                      superpixel_GPU_helper_sm* sm_helper,
                      const int npix, const int nbatch,
                      const int width, const int nftrs,
                      const int nspix_buffer,
                      float a_0, float b_0, int max_nspix);

__global__
void split_hastings_ratio_s(const float* img, int* sm_pairs,
                          superpixel_params* sp_params,
                          superpixel_GPU_helper* sp_helper,
                          superpixel_GPU_helper_sm* sm_helper,
                          const int npix, const int nbatch,
                          const int width, const int nftrs,
                          const int nspix_buffer,
                          float alpha_hasting_ratio,
                          int max_nspix, int* max_sp );
