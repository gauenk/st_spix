
#include "cuda.h"
#include "cuda_runtime.h"
#include "device_launch_parameters.h"

#ifndef MY_SP_STRUCT
#define MY_SP_STRUCT
#include "../bass/share/my_sp_struct.h"
#endif


__host__ void prop_bass(float* img, int* seg,
                        superpixel_params* sp_params,
                        superpixel_params* prior_params,
                        int* prior_map, bool* border,
                        superpixel_GPU_helper* sp_helper,
                        superpixel_GPU_helper_sm* sm_helper,
                        int* sm_seg1 ,int* sm_seg2, int* sm_pairs,
                        int niters, int niters_seg, int sm_start,
                        float3 pix_cov,float logdet_pix_cov,float potts,
                        int nspix, int nbatch, int width, int height, int nftrs);
__host__
int run_split_merge(const float* img, int* seg,
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
/* __host__ */
/* int run_split_merge(const float* img, int* seg, */
/*                     bool* border, superpixel_params* sp_params, */
/*                     superpixel_params* prior_params, int* prior_map, */
/*                     superpixel_GPU_helper* sp_helper, */
/*                     superpixel_GPU_helper_sm* sm_helper, */
/*                     int* sm_seg1 ,int* sm_seg2, int* sm_pairs, */
/*                     float alpha_hastings, float sm_std, */
/*                     int count, int i, int max_SP, */
/*                     const int npix, const int nbatch, */
/*                     const int width, const int height, */
/*                     const int nftrs, const int nspix_buffer); */
