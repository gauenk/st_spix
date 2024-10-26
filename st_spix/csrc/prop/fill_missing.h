#include "cuda.h"
#include "cuda_runtime.h"
#include "device_launch_parameters.h"


__host__
void fill_missing(int* seg,  float* centers, int* missing, bool* border,
                  int nbatch, int width, int height,
                  int nmissing, int break_iter);

__global__
void find_border_along_missing(const int* seg, const int* missing,
                               bool* border, const int nmissing,
                               const int nbatch, const int width,
                               const int height, int* num_neg);

__global__
void update_missing_seg_nn(int* seg, float* centers, bool* border,
                           const int nbatch, const int width, const int height,
                           const int npix, const int xmod3, const int ymod3);

