/*************************************************

       Initialize Superpixel Segmentation

**************************************************/

#include "cuda.h"
#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#include "structs.h"
#include <cfloat>


__host__ int nspix_from_spsize(int sp_size, int width, int height);
__host__ int init_seg(int* seg, int sp_size, int width, int height, int nbatch);
__global__ void InitHexCenter(double* centers, double H, double w, int npix,
                              int max_num_sp_x, int xdim, int ydim);
__global__ void InitHexSeg(int* seg, double* centers,
                           int K, int npix, int xdim);

__host__ int init_square_seg(int* seg, int sp_size, int width, int height, int nbatch);
__global__ void InitSquareSeg(int* seg,int sp_size,int max_num_sp_x, int npix, int width);
