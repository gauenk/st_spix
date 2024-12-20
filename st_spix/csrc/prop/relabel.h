
// -- cpp imports --
#include <stdio.h>
#include "pch.h"

// -- "external" import --
#ifndef MY_SP_STRUCT
#define MY_SP_STRUCT
#include "../bass/share/my_sp_struct.h"
#endif

__global__
void search_kernel(float* sims, bool* living_mask,
                   float* mu_app, float* prior_mu_app,
                   double* mu_shape, double* prior_mu_shape,
                   double* sigma_shape, double* prior_sigma_shape,
                   int* counts, float* prior_counts,
                   int* living_ids, int* dead_ids,
                   int nliving, int ndead);
__global__
void relabel_kernel(int* spix, float* sim_vals, int* sim_inds,
                    int* dead, bool* living_mask, int npix, int num, float thresh);


void run_relabel(torch::Tensor spix, const PySuperpixelParams params, float thresh);
