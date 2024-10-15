
#include "cuda.h"
#include "cuda_runtime.h"
#include "device_launch_parameters.h"

__host__ void rgb2lab(float* img_rgb, float* img_lab, int npix, int nbatch);
__host__ void lab2rgb(float* img_rgb, float* img_lab, int npix, int nbatch);
__global__ void rgb_to_lab(float* img_rgb,float* img_lab, int npix);
__global__ void lab_to_rgb(float* img_rgb,float* img_lab, int npix);

