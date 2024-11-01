
#include "cuda.h"
#include "cuda_runtime.h"
#include "device_launch_parameters.h"

// -- "external" import --
#ifndef MY_SP_STRUCT
#define MY_SP_STRUCT
#include "../bass/share/my_sp_struct.h"
#endif



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

