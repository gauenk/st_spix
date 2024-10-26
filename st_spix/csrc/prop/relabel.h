
// -- cpp imports --
#include <stdio.h>
#include "pch.h"

// -- "external" import --
#ifndef MY_SP_STRUCT
#define MY_SP_STRUCT
#include "../bass/share/my_sp_struct.h"
#endif

__global__
void search_kernel(float* sims, float* mu_app, float* prior_mu_app,
                   float* mu_shape, float* prior_mu_shape,
                   float* sigma_shape, float* prior_sigma_shape,
                   int* counts, int* prior_counts,
                   int* living_ids, int* dead_ids, int nliving, int ndead);
__global__
void relabel_kernel(int* spix, float* sim_vals, int* sim_inds,
                    int* dead, int npix, int num, float thresh);
