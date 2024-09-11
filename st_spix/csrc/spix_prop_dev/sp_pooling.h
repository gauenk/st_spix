

#include "cuda.h"
#include "cuda_runtime.h"
#include "device_launch_parameters.h"

__global__ void
run_sp_downsample(float* img, int* seg,
                  float* down_sampled, int* down_count,
                  const int npix, const int nftrs);

__global__
void run_sp_pooling( float* pooled, int* seg,
                     float* down_sampled, int* down_count,
                     const int npix, const int nftrs);

