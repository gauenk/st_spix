/*************************************************

       Initialize Superpixel Segmentation

**************************************************/

#include "cuda.h"
#include "cuda_runtime.h"
#include "device_launch_parameters.h"

#ifndef MY_SP_STRUCT
#define MY_SP_STRUCT
#include "../bass/share/my_sp_struct.h"
#endif

__host__ int init_seg(int* seg, int sp_size, int width, int height, int nbatch);
__global__ void InitHexCenter(double* centers, double H, double w, int npix,
                              int max_num_sp_x, int xdim, int ydim);
__global__ void InitHexSeg(int* seg, double* centers,
                           int K, int npix, int xdim);

