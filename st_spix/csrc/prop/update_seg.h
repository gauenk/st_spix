
#include "cuda.h"
#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#include "seg_utils.h"

__global__
void update_seg_subset(float* img, int* seg, bool* border,
                       spix_params* sp_params, const float3 pix_cov,
                       const float logdet_pix_cov,  const float potts,
                       const int npix, const int nbatch,
                       const int width, const int height, const int nftrs,
                       const int xmod3, const int ymod3);

__host__
void update_seg(float* img, int* seg, bool* border,
                spix_params* sp_params, const int niters,
                const float3 pix_cov, const float logdet_pix_cov,
                const float potts, const int npix,
                int nbatch, int width, int height, int nftrs);

