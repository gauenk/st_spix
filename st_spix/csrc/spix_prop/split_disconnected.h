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

__host__
void run_split_disconnected(float* img, int* seg,
                            int* missing, bool* border,
                            superpixel_params* sp_params, 
                            const int nPixels, const int nMissing,
                            int nbatch, int xdim, int ydim, int nftrs,
                            const float3 J_i, const float logdet_Sigma_i, 
                            float i_std, int s_std, int nInnerIters,
                            const int nSPs, int nSPs_buffer,
                            float beta_potts_term, int* debug_spix,
                            bool* debug_border, bool debug_fill);

__global__ void mark_disconnected(int* seg, bool* border, bool* disc);


