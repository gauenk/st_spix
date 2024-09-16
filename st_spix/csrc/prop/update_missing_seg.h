
#include "cuda.h"
#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#include "seg_utils.h"

__global__
void update_missing_seg_subset(float* img, int* seg, bool* border, bool* missing,
                               superpixel_params* sp_params, const float3 pix_cov,
                               const float logdet_pix_cov,  const float potts,
                               const int npix, const int nbatch,
                               const int width, const int height, const int nftrs,
                               const int width_mod, const int height_mod);

__host__
void update_missing_seg(float* img, int* seg, bool* border, bool* missing,
                        superpixel_params* sp_params, const int niters,
                        const float3 pix_cov, const float logdet_pix_cov,
                        const float potts, const int npix,
                        int nbatch, int width, int height, int nftrs);

