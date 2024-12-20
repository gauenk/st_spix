
#include "cuda.h"
#include "cuda_runtime.h"
#include "device_launch_parameters.h"

#ifndef MY_SP_STRUCT
#define MY_SP_STRUCT
#include "../bass/share/my_sp_struct.h"
#endif


__host__ void refine(float* img, int* seg, spix_params* sp_params,
                             bool* border, spix_helper* sp_helper,
                             int niters, int niters_seg,
                             /* float3 pix_ivar,float logdet_pix_var, */
                             float sigma2_app, float potts,
                             int sp_size, int nspix, int nspix_buffer,
                             int nbatch, int width, int height, int nftrs);
