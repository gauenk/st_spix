#include "cuda.h"
#include "cuda_runtime.h"
#include "device_launch_parameters.h"

#ifndef MY_SP_STRUCT
#define MY_SP_STRUCT
#include "../bass/share/my_sp_struct.h"
#endif


__host__ int bass(float* img, int* seg,spix_params* sp_params,bool* border,
                  spix_helper* sp_helper,spix_helper_sm* sm_helper,
                  int* sm_seg1 ,int* sm_seg2, int* sm_pairs,
                  int niters, int niters_seg, int sm_start,
                  float3 pix_ivar,float logdet_pix_var,
                  float potts, float alpha_hastings,
                  int nspix, int nbatch, int width, int height, int nftrs);

