
#include <stdio.h>
#include <math.h>
#include <cuda.h>
#include <cuda_runtime.h>
#include <cmath>

#include "core/Superpixels.h"


__global__
void copy_spix_to_params(float* means, float* cov, int* counts,
                         superpixel_params* sp_params, int* ids, int nspix){

    // -- filling superpixel params into image --
    int ix = threadIdx.x + blockIdx.x * blockDim.x;  
    if (ix>=nspix) return; 

    // -- offset memory access --
    float* means_ix = means + ix * 5;
    float* cov_ix = cov + ix * 4;
    int* counts_ix = counts + ix;

    // -- read spix --
    int sp_index = ids[ix];
    if (sp_index < 0){ return; }
    auto params_ix = sp_params[sp_index];
      
    // -- fill params --
    cov_ix[0] = params_ix.sigma_s.x;
    cov_ix[1] = params_ix.sigma_s.y;
    cov_ix[2] = params_ix.sigma_s.z;
    cov_ix[3] = params_ix.logdet_Sigma_s;
    means_ix[0] = params_ix.mu_i.x;
    means_ix[1] = params_ix.mu_i.y;
    means_ix[2] = params_ix.mu_i.z;
    means_ix[3] = params_ix.mu_s.x;
    means_ix[4] = params_ix.mu_s.y;
    counts_ix[0] = params_ix.count;
}


__global__
void copy_spix_to_params_parents(float* means, float* cov,
                                 int* counts, int* spix_parents,
                                 superpixel_params* sp_params, int* ids, int nspix){

    // -- filling superpixel params into image --
    int ix = threadIdx.x + blockIdx.x * blockDim.x;  
    if (ix>=nspix) return; 

    // -- offset memory access --
    float* means_ix = means + ix * 5;
    float* cov_ix = cov + ix * 4;
    int* counts_ix = counts + ix;
    int* spix_ix = spix_parents + ix;

    // -- read spix --
    int sp_index = ids[ix];
    if (sp_index < 0){ return; }
    auto params_ix = sp_params[sp_index];
      
    // -- fill params --
    cov_ix[0] = params_ix.sigma_s.x;
    cov_ix[1] = params_ix.sigma_s.y;
    cov_ix[2] = params_ix.sigma_s.z;
    cov_ix[3] = params_ix.logdet_Sigma_s;
    means_ix[0] = params_ix.mu_i.x;
    means_ix[1] = params_ix.mu_i.y;
    means_ix[2] = params_ix.mu_i.z;
    means_ix[3] = params_ix.mu_s.x;
    means_ix[4] = params_ix.mu_s.y;
    counts_ix[0] = params_ix.count;
    spix_ix[0] = params_ix.parent_spix;
}


__global__
void copy_spix_to_params_icov2cov(float* means, float* cov, int* counts,
                                  superpixel_params* sp_params, int* ids, int nspix){

    // -- filling superpixel params into image --
    int ix = threadIdx.x + blockIdx.x * blockDim.x;  
    if (ix>=nspix) return; 

    // -- offset memory access --
    float* means_ix = means + ix * 5;
    float* cov_ix = cov + ix * 4;
    int* counts_ix = counts + ix;

    // -- read spix --
    int sp_index = ids[ix];
    if (sp_index < 0){ return; }
    auto params_ix = sp_params[sp_index];
      
    // -- fill params --
    cov_ix[0] = params_ix.sigma_s.x;
    cov_ix[1] = params_ix.sigma_s.y;
    cov_ix[2] = params_ix.sigma_s.z;
    cov_ix[3] = params_ix.logdet_Sigma_s;
    means_ix[0] = params_ix.mu_i.x;
    means_ix[1] = params_ix.mu_i.y;
    means_ix[2] = params_ix.mu_i.z;
    means_ix[3] = params_ix.mu_s.x;
    means_ix[4] = params_ix.mu_s.y;
    counts_ix[0] = params_ix.count;

    // -- invert cov --
    // .x => sx   .z => sy
    // .y => rho * sx * sy
    double inv_detC = exp(cov_ix[3]);
    cov_ix[0] = inv_detC * cov[2];
    cov_ix[1] = -inv_detC * cov[1];
    cov_ix[2] = inv_detC * cov[0];

}



__global__
void copy_params_to_spix(float* means, float* cov, int* counts,
                         superpixel_params* sp_params, int* ids, int nspix){

    // -- filling superpixel params into image --
    int ix = threadIdx.x + blockIdx.x * blockDim.x;  
    if (ix>=nspix) return; 

    // -- offset memory access --
    float* means_ix = means + ix * 5;
    float* cov_ix = cov + ix * 4;
    int* counts_ix = counts + ix;

    // -- read spix --
    int sp_index = ids[ix];
    if (sp_index < 0){ return; }
    auto params_ix = sp_params[sp_index];
      
    // -- fill params --
    double3 sigma_s;
    sigma_s.x = cov_ix[0];
    sigma_s.y = cov_ix[1];
    sigma_s.z = cov_ix[2];
    params_ix.sigma_s = sigma_s;
    // params_ix.sigma_s.x = cov_ix[0];
    // params_ix.sigma_s.y = cov_ix[1];
    // params_ix.sigma_s.z = cov_ix[2];
    params_ix.logdet_Sigma_s = cov_ix[3];

    float3 mu_i;
    mu_i.x = means_ix[0];
    mu_i.y = means_ix[1];
    mu_i.z = means_ix[2];
    params_ix.mu_i = mu_i;
    // params_ix.mu_i.x = means_ix[0];
    // params_ix.mu_i.y = means_ix[1];
    // params_ix.mu_i.z = means_ix[2];
    params_ix.mu_s.x = means_ix[3];
    params_ix.mu_s.y = means_ix[4];
    params_ix.count = counts_ix[0];
}
