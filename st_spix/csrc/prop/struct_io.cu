/*****************************************************************


        We want a nicely formatted IO for superpixel parameters.
        We don't really want a ton of tensors each function call

******************************************************************/



#include <stdio.h>
#include <math.h>
#include <cuda.h>
#include <cuda_runtime.h>
#include <cmath>

#include "core/Superpixels.h"


__global__
void copy_spix_to_params(float* means, float* cov, int* counts,
                         superpixel_params* sp_params, int* ids, int nspix){
}


__global__
void copy_spix_to_params_parents(float* means, float* cov,
                                 int* counts, int* spix_parents,
                                 superpixel_params* sp_params, int* ids, int nspix){

}


