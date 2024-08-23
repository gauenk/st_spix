#include "cuda.h"
#include "cuda_runtime.h"
#include "device_launch_parameters.h"

#ifndef MY_SP_STRUCT
#define MY_SP_STRUCT
#include "../share/my_sp_struct.h"
#endif

__global__ void warm_up_gpu();
__host__ void warm_up();

__host__ void CudaFindBorderPixels( const int* seg, bool* border, const int nPixels, const int nbatch, const int xdim, const int ydim, const int single_border);
__host__ void CudaFindBorderPixels_end( const int* seg, bool* border, const int nPixels, const int nbatch, const int xdim, const int ydim, const int single_border);

__global__  void find_border_pixels( const int* seg, bool* border, const int nPixels, const int nbatch, const int xdim, const int ydim, const int single_border);
__global__  void find_border_pixels_end(const int* seg, bool* border, const int nPixels, const int nbatch, const int xdim, const int ydim, const int single_border);

__host__ void update_seg(float* img, int* seg,int* seg_potts_label, bool* border,
                         superpixel_params* sp_params,
                         const float3 J_i, const float logdet_Sigma_i,
                         bool cal_cov, float i_std, int s_std, int nInnerIters,
                         const int nPixels, const int nSPs, int nSPs_buffer,
                         int nbatch, int xdim, int ydim, int nftrs,
                         float beta_potts_term);

__global__  void update_seg_subset(
    float* img, int* seg,int* seg_potts_label, bool* border,
    superpixel_params* sp_params,
    const float3 J_i, const float logdet_Sigma_i,
    bool cal_cov, float i_std, int s_std,
    const int nPts,const int nSuperpixels,
    int nbatch, int xdim, int ydim, int nftrs,
    const int xmod3, const int ymod3,
    const float betta_potts_term);

/* __global__ void update_seg_label(int* seg, int* seg_potts_label,const int nPts); */
__global__  void cal_posterior( float* img, int* seg, bool* border, superpixel_params* sp_params, float3 J_i, float logdet_Sigma_i, float i_std, int s_std, int* changes, int nPts , int xdim);
