

#include "cuda.h"
#include "cuda_runtime.h"
#include "device_launch_parameters.h"

__global__
void run_sp_video_downsample(float* img, int* seg,
                             float* downsampled, float* downcount,
                             int nspix, int nframes, int npix, int nftrs);

__global__
void run_sp_video_pooling(float* pooled, int* seg, float* downsampled,
                          int nspix, int nframes, int npix, int nftrs);
