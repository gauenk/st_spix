
/*********************************************************************************

             Run the pairwise comparision between the living superpixels
             and all dead superpixels to possibly relabel the
             living superpixels with dead ones


*********************************************************************************/


// -- cpp imports --
#include <stdio.h>
#include "pch.h"

// -- "external" import --
#ifndef MY_SP_STRUCT
#define MY_SP_STRUCT
#include "../bass/share/my_sp_struct.h"
#endif

// -- utils --
#include "relabel.h"


/**********************************************************

                  Kernel

***********************************************************/

__global__
void search_kernel(float* sims, float* mu_app, float* prior_mu_app,
                   float* mu_shape, float* prior_mu_shape,
                   float* sigma_shape, float* prior_sigma_shape,
                   int* counts, int* prior_counts,
                   int* living_ids, int* dead_ids, int nliving, int ndead) {

  // -- filling superpixel params into image --
  int ix = threadIdx.x + blockIdx.x * blockDim.x;
  if (ix >= nliving) return;
  int jx = threadIdx.y + blockIdx.y * blockDim.y;
  if (jx >= ndead) return;
  int isp = living_ids[ix];
  int jsp = dead_ids[jx];
  if (isp < 0) return;
  if (jsp < 0) return;
  float thresh = -100;

  /* spix_id = ids[ix]; */
  float delta = 0;
  float delta_i = 0;
  float delta_j = 0;

  // -- index by parameters --
  params_s = params[s];
  params_k = params[k];

  lprob_s = compute_lprob(params_s,summary_stats);
  lprob_k = compute_lprob(params_k,summary_stats);

  // -- compute difference --
  float* mu_app_ix = mu_app+3*isp;
  float* mu_app_jx = mu_app+3*jsp;
  delta_i += (mu_app_ix[0] - prior_mu_app[0])*(mu_app_ix[0] - prior_mu_app[0]);
  delta_i += (mu_app_ix[1] - prior_mu_app[1])*(mu_app_ix[1] - prior_mu_app[1]);
  delta_i += (mu_app_ix[2] - prior_mu_app[2])*(mu_app_ix[2] - prior_mu_app[2]);

  delta_j += (mu_app_jx[0] - prior_mu_app[0])*(mu_app_jx[0] - prior_mu_app[0]);
  delta_j += (mu_app_jx[1] - prior_mu_app[1])*(mu_app_jx[1] - prior_mu_app[1]);
  delta_j += (mu_app_jx[2] - prior_mu_app[2])*(mu_app_jx[2] - prior_mu_app[2]);

  delta = delta_i - delta_j; // compare with "0"

  // -- write to output --
  *(sims + isp + nliving*jsp) = delta;
  /* sims[isp][jsp] = delta; */

  // -- assign new label [used for split] --
  if (lprob_s < thresh){
    relabel[k] = atomicAdd(max_sp,1);
  }else if(lprob_k < thresh){
    relabel[k] = s;
  }else{
    relabel[k] = -1;
  }

}


__global__
void relabel_kernel(int* spix, float* sim_vals, int* sim_inds,
                    int* dead, int npix, float thresh) {

  // -- relabel the list --
  int ix = threadIdx.x + blockIdx.x * blockDim.x;
  if (ix >= npix) return;
  int s = *(spix + ix); // read current superpixel label
  // assert(s "isin" living_superpixels);
  float sim_val = *(sim_vals + s); // get value to check if we swap
  if (sim_val < thresh){
    int sim_ind = *(sim_inds + s); // read location of new id
    int new_id = *(dead + sim_ind); // read new id
    *(spix + ix) = new_id; // update the pixel
  }
}

/**********************************************************

             -=-=-=-=- Python API  -=-=-=-=-=-

***********************************************************/

void run_relabel(torch::Tensor spix, const PySuperpixelParams params, float thresh){

    // -- check --
    CHECK_INPUT(spix);
    CHECK_INPUT(params.mu_app);
    CHECK_INPUT(params.mu_shape);
    CHECK_INPUT(params.sigma_shape);
    CHECK_INPUT(params.logdet_sigma_shape);
    CHECK_INPUT(params.counts);
    CHECK_INPUT(params.prior_counts);

    // -- unpack --
    int nbatch = spix.size(0);
    int height = spix.size(1);
    int width = spix.size(2);
    int npix = height*width;
    assert(nbatch==1);

    // -- allocate filled spix --
    auto options_i32 = torch::TensorOptions().dtype(torch::kInt32)
      .layout(torch::kStrided).device(spix.device());
    auto options_f32 = torch::TensorOptions().dtype(torch::kFloat32)
      .layout(torch::kStrided).device(spix.device());
    torch::Tensor filled_spix = spix.clone();

    // --------------------------------
    //
    // --   living/dead superpixels  --
    //
    // --------------------------------

    auto unique_ids = std::get<0>(at::_unique(spix));
    auto living_ids = unique_ids.data<int>();
    int nliving = unique_ids.size(0);
    int max_nspix = params.ids.size(0);
    torch::Tensor all_of_em = torch::arange(0,max_nspix);
    auto comp = (unique_ids.unsqueeze(1) != all_of_em);
    auto mask = comp.all(1);
    auto dead_th = all_of_em.masked_select(mask);
    fprintf(stdout,"comp.size(0),comp.size(1): %d,%d\n",comp.size(0),comp.size(1));
    fprintf(stdout,"dead_th.size(0): %d\n",dead_th.size(0));
    int ndead = dead_th.size(0);
    int* dead_ids = dead_th.data<int>();
    /* int nliving = nspix; */
    /* int ndead = nspix; */
    /* int* dead_ids = ids; */

    // -- unpack --
    int* spix_ptr = spix.data<int>();
    float* mu_app = params.mu_app.data<float>();
    float* prior_mu_app = params.prior_mu_app.data<float>();
    float* mu_shape = params.mu_shape.data<float>();
    float* prior_mu_shape = params.prior_mu_shape.data<float>();
    float* sigma_shape = params.sigma_shape.data<float>();
    float* prior_sigma_shape = params.prior_sigma_shape.data<float>();
    int* prior_counts = params.prior_counts.data<int>();
    int* counts = params.counts.data<int>();
    // a list of DEAD ids is needed

    // -- compare each living spix with dead ones  --
    torch::Tensor sims = torch::zeros({nliving,ndead},options_f32);
    float* sims_ptr = sims.data<float>();
    int num_blocks = ceil( double(max_nspix) / double(THREADS_PER_BLOCK) );
    dim3 nblocks(num_blocks);
    dim3 nthreads(THREADS_PER_BLOCK);
    search_kernel<<<nblocks,nthreads>>>(sims_ptr, mu_app, prior_mu_app,
                                        mu_shape, prior_mu_shape, sigma_shape,
                                        prior_sigma_shape, counts, prior_counts,
                                        living_ids,dead_ids,nliving,ndead);

    // -- find the minimum --
    auto result = torch::min(sims, 1);
    torch::Tensor minValues = std::get<0>(result);
    torch::Tensor minIndices = std::get<1>(result);
    float* sim_vals = minValues.data<float>();
    int* sim_inds = minIndices.data<int>();

    // -- relabel --
    int num_blocks1 = ceil( double(npix) / double(THREADS_PER_BLOCK) );
    dim3 nblocks1(num_blocks1);
    dim3 nthreads1(THREADS_PER_BLOCK);
    relabel_kernel<<<nblocks1,nthreads1>>>(spix_ptr, sim_vals, sim_inds,
                                           dead_ids, npix, thresh);
}



void init_relabel(py::module &m){
  m.def("relabel", &run_relabel,"relabel living labels from the dead #zombie");
}
