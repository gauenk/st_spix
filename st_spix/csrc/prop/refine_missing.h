
#include "cuda.h"
#include "cuda_runtime.h"
#include "device_launch_parameters.h"

#ifndef MY_SP_STRUCT
#define MY_SP_STRUCT
#include "../bass/share/my_sp_struct.h"
#endif


__host__ void refine_missing(float* img, int* seg, spix_params* sp_params,
                             bool* missing, bool* border, spix_helper* sp_helper,
                             int niters, int niters_seg,
                             /* float3 pix_ivar,float logdet_pix_var, */
                             float sigma_app, float potts,
                             int nspix, int nbatch, int width, int height, int nftrs);

/* __global__ */
/* void find_border_along_missing(const int* seg, const int* missing, */
/*                                bool* border, const int nmissing, */
/*                                const int nbatch, const int width, */
/*                                const int height, int* num_neg); */

/* __global__ */
/* void update_missing_seg(int* seg, float* centers, bool* border, */
/*                         const int nbatch, const int width, const int height, */
/*                         const int npix, const int xmod3, const int ymod3); */
