
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


__host__ void calc_prop_seg(float* image_gpu_double, int* seg_gpu,
                            int* missing_gpu, int* seg_potts_label, bool* border_gpu,
                            superpixel_params* sp_params,
                            superpixel_params* sp_params_prev,
                            superpixel_GPU_helper* sp_gpu_helper,
                            const float3 J_i, const float logdet_Sigma_i,
                            superpixel_options sp_options,
                            int nbatch, int nftrs, int dim_x, int dim_y, int nSPs,
                            bool use_transition, float* debug_seg);

