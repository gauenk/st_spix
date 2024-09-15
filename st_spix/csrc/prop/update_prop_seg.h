
#include "cuda.h"
#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#include "seg_utils.h"

__global__
void update_prop_seg_subset(float* img, int* seg, bool* border,
                               superpixel_params* sp_params, const float3 pix_cov,
                               const float logdet_pix_cov,  const float potts,
                               const int npix, const int nspix, const int nbatch,
                               const int xdim, const int ydim, const int nftrs,
                               const int xmod3, const int ymod3);

__host__
void update_prop_seg(float* img, int* seg, bool* border,
                      superpixel_params* sp_params, const int niters,
                      const float3 pix_cov, const float logdet_pix_cov,
                      const float potts, const int npix, const int nspix,
                      int nbatch, int xdim, int ydim, int nftrs);

