#include "cuda.h"
#include "cuda_runtime.h"
#include "device_launch_parameters.h"

#ifndef MY_SP_STRUCT
#define MY_SP_STRUCT
#include "../share/my_sp_struct.h"
#endif

#ifndef SEG_UTILS_H
#define SEG_UTILS_H

__device__ inline float2 cal_prop_posterior(
    float* imgC, int* seg, int x, int y,
    superpixel_params* sp_params,
    int seg_idx, float3 pix_cov,
    float logdet_pix_cov, float i_std, int s_std,
    float potts, float beta, float2 res_max){

    // -- init res --
    float res = -1000; // some large negative number // why?

    // -- compute color/spatial differences --
    const float x0 = __ldg(&imgC[0])-__ldg(&sp_params[seg_idx].mu_i.x);
    const float x1 = __ldg(&imgC[1])-__ldg(&sp_params[seg_idx].mu_i.y);
    const float x2 = __ldg(&imgC[2])-__ldg(&sp_params[seg_idx].mu_i.z);
    const int d0 = x - __ldg(&sp_params[seg_idx].mu_s.x);
    const int d1 = y - __ldg(&sp_params[seg_idx].mu_s.y);

    // -- color component --
    const float pix_cov_x = pix_cov.x;
    const float pix_cov_y = pix_cov.y;
    const float pix_cov_z = pix_cov.z;
    const float sigma_s_x = __ldg(&sp_params[seg_idx].sigma_s.x);
    const float sigma_s_y = __ldg(&sp_params[seg_idx].sigma_s.y);
    const float sigma_s_z = __ldg(&sp_params[seg_idx].sigma_s.z);
    const float logdet_sigma_s = __ldg(&sp_params[seg_idx].logdet_Sigma_s);

    // -- color component --
    res = res - (x0*x0*pix_cov_x + x1*x1*pix_cov_y + x2*x2*pix_cov_z);
    res = res - logdet_pix_cov; // okay; log p(x,y) = -log detSigma

    // -- space component --
    res = res - d0*d0*sigma_s_x - d1*d1*sigma_s_z - 2*d0*d1*sigma_s_y; // sign(s_y) = -1
    res = res - logdet_sigma_s;

    // -- potts term --
    res = res - beta*potts;

    // -- update res --
    if( res>res_max.x)
    {
      res_max.x = res;
      res_max.y = seg_idx;
    }

    return res_max;
}

#endif


__host__ void CudaFindBorderPixels( const int* seg, bool* border, const int nPixels, const int nbatch, const int xdim, const int ydim, const int single_border);
__host__ void CudaFindBorderPixels_end( const int* seg, bool* border, const int nPixels, const int nbatch, const int xdim, const int ydim, const int single_border);

__global__  void find_border_pixels( const int* seg, bool* border,
                                     const int nPixels, const int nbatch,
                                     const int xdim, const int ydim,
                                     const int single_border);
__global__  void find_border_pixels_end(const int* seg, bool* border,
                                        const int nPixels, const int nbatch,
                                        const int xdim, const int ydim,
                                        const int single_border);
