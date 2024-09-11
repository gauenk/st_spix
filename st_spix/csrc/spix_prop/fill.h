#include "cuda.h"
#include "cuda_runtime.h"
#include "device_launch_parameters.h"

/* #ifndef MY_SP_SHARE_H */
/* #define MY_SP_SHARE_H */
/* #include "../bass/share/sp.h" */
/* #endif */

/* #ifndef MY_SP_STRUCT */
/* #define MY_SP_STRUCT */
/* #include "../bass/share/my_sp_struct.h" */
/* #endif */
/* #ifndef MY_SP_PROP_STRUCT */
/* #define MY_SP_PROP_STRUCT */
/* #include "../bass/share/my_sp_struct.h" */
/* #endif */


/* __global__  void find_border_pixels( const int* seg, bool* border, const int nPixels, const int nbatch, const int xdim, const int ydim, const int single_border); */
/* __global__  void find_border_pixels_end(const int* seg, bool* border, const int nPixels, const int nbatch, const int xdim, const int ydim, const int single_border); */

__host__ void fill_seg(int* seg,  float* centers, int* missing, bool* border,
                       int nbatch, int width, int height,
                       int nspix, int nmissing, int break_iter);

__global__
void find_border_along_missing(const int* seg, const int* missing,
                               bool* border, const int nmissing,
                               const int nbatch, const int width,
                               const int height, int* num_neg);

__global__  void update_missing_seg(
    int* seg, float* centers, bool* border,
    const int nbatch, const int width, const int height,
    const int npix, const int xmod3, const int ymod3);

