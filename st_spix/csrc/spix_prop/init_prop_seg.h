#include "cuda.h"
#include "cuda_runtime.h"
#include "device_launch_parameters.h"

#ifndef MY_SP_SHARE_H
#define MY_SP_SHARE_H
#include "../bass/share/sp.h"
#endif

#ifndef MY_SP_STRUCT
#define MY_SP_STRUCT
#include "../bass/share/my_sp_struct.h"
#endif
/* #ifndef MY_SP_PROP_STRUCT */
/* #define MY_SP_PROP_STRUCT */
/* #include "../bass/share/my_sp_struct.h" */
/* #endif */


/* __global__  void find_border_pixels( const int* seg, bool* border, const int nPixels, const int nbatch, const int xdim, const int ydim, const int single_border); */
/* __global__  void find_border_pixels_end(const int* seg, bool* border, const int nPixels, const int nbatch, const int xdim, const int ydim, const int single_border); */

__host__ void init_prop_seg(float* img, int* seg,
                            int* missing, bool* border,
                            superpixel_params* sp_params,
                            const int nPixels, const int nMissing,
                            int nbatch, int xdim, int ydim, int nftrs,
                            const float3 J_i, const float logdet_Sigma_i,
                            float i_std, int s_std, int nInnerIters,
                            const int nSPs, int nSPs_buffer,
                            float beta_potts_term,int* debug, bool debug_fill);
                            /* const int nPixels, const int nMissing, */
                            /* int nbatch, int xdim, int ydim, int nftrs); */
                            /* const float3 J_i, const float logdet_Sigma_i, */
                            /* bool cal_cov, float i_std, int s_std, */
                            /* int nInnerIters, */
                            /* const int nSPs, int nSPs_buffer, */
                            /* float beta_potts_term); */

__global__  void update_prop_seg_subset(
    float* img, int* seg, bool* border,
    superpixel_params* sp_params,
    const float3 J_i, const float logdet_Sigma_i,
    float i_std, int s_std, const int nPts,const int nSuperpixels,
    const int nbatch, const int xdim, const int ydim, const int nftrs,
    const int xmod3, const int ymod3, const float beta_potts_term);

__global__
void find_prop_border_pixels(const int* seg, const int* missing,
                             bool* border, const int nMissing, const int nbatch,
                             const int xdim, const int ydim);

__device__ inline float2 cal_posterior_new_v0(
    float* imgC, int* seg, int x, int y,
    superpixel_params* sp_params,
    int seg_idx, float3 J_i,
    float logdet_Sigma_i, float i_std, int s_std,
    float potts, float beta, float2 res_max){

    // -- init res --
    float res = -1000; // some large negative number
    /* float* imgC = img + idx * 3; */

    // -- compute color/spatial differences --
    const float x0 = __ldg(&imgC[0])-__ldg(&sp_params[seg_idx].mu_i.x);
    const float x1 = __ldg(&imgC[1])-__ldg(&sp_params[seg_idx].mu_i.y);
    const float x2 = __ldg(&imgC[2])-__ldg(&sp_params[seg_idx].mu_i.z);
    const int d0 = x - __ldg(&sp_params[seg_idx].mu_s.x);
    const int d1 = y - __ldg(&sp_params[seg_idx].mu_s.y);

    // -- color component --
    const float J_i_x = J_i.x;
    const float J_i_y = J_i.y;
    const float J_i_z = J_i.z;
    const float sigma_s_x = __ldg(&sp_params[seg_idx].sigma_s.x);
    const float sigma_s_y = __ldg(&sp_params[seg_idx].sigma_s.y);
    const float sigma_s_z = __ldg(&sp_params[seg_idx].sigma_s.z);
    const float logdet_sigma_s = __ldg(&sp_params[seg_idx].logdet_Sigma_s);

    // -- color component --
    res = res - (x0*x0*J_i_x + x1*x1*J_i_y + x2*x2*J_i_z);
    //res = -calc_squared_mahal_3d(imgC,mu_i,J_i);
    res = res - logdet_Sigma_i;

    // -- space component --
    res = res - d0*d0*sigma_s_x;
    res = res - d1*d1*sigma_s_z;
    res = res -  2*d0*d1*sigma_s_y;
    // res -= calc_squared_mahal_2d(pt,mu_s,J_s);
    res = res -  logdet_sigma_s;
    res = res -beta*potts;
    /*if (res > atomicMaxFloat2(&post_changes[idx].post[4],res))
    {
      seg[idx] = seg_idx;

    }*/

    if( res>res_max.x)
    {
      res_max.x = res;
      res_max.y = seg_idx;

    }


    return res_max;
}



/* __global__  void update_seg_subset( */
/*     float* img, int* seg,int* seg_potts_label, bool* border, */
/*     superpixel_params* sp_params, */
/*     const float3 J_i, const float logdet_Sigma_i, */
/*     bool cal_cov, float i_std, int s_std, */
/*     const int nPts,const int nSuperpixels, */
/*     int nbatch, int xdim, int ydim, int nftrs, */
/*     const int xmod3, const int ymod3, */
/*     const float betta_potts_term, post_changes_helper* post_changes); */

/* __global__ void update_seg_label(int* seg, int* seg_potts_label,const int nPts); */
