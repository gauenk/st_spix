
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
void search_kernel(float* sims, bool* living_mask,
                   float* mu_app, float* prior_mu_app,
                   double* mu_shape, double* prior_mu_shape,
                   double* sigma_shape, double* prior_sigma_shape,
                   int* counts, float* prior_counts,
                   int* living_ids, int* dead_ids, int nliving, int ndead) {

  // -- filling superpixel params into image --
  int ntotal = nliving + ndead;
  int ix = threadIdx.x + blockIdx.x * blockDim.x;
  if (ix >= nliving) return;
  int jx = threadIdx.y + blockIdx.y * blockDim.y;
  if (jx >= ndead) return;
  int isp = living_ids[ix];
  int jsp = dead_ids[jx];
  if (isp < 0) return;
  if (jsp < 0) return;

  /* spix_id = ids[ix]; */
  float delta = 0;
  float delta_i = 0;
  float delta_j = 0;

  // -- compute difference --
  float* mu_app_ix = mu_app+3*isp;
  double* mu_shape_ix = mu_shape+2*isp;
  float* mu_app_jx = mu_app+3*jsp;
  double* mu_shape_jx = mu_shape+2*jsp;
  float* _prior_mu_app = prior_mu_app+3*isp;
  double* _prior_mu_shape = prior_mu_shape+2*isp;

  // -- [appearance] sum deltas --
  delta_i += (mu_app_ix[0] - _prior_mu_app[0])*(mu_app_ix[0] - _prior_mu_app[0]);
  delta_i += (mu_app_ix[1] - _prior_mu_app[1])*(mu_app_ix[1] - _prior_mu_app[1]);
  delta_i += (mu_app_ix[2] - _prior_mu_app[2])*(mu_app_ix[2] - _prior_mu_app[2]);

  delta_j += (mu_app_jx[0] - _prior_mu_app[0])*(mu_app_jx[0] - _prior_mu_app[0]);
  delta_j += (mu_app_jx[1] - _prior_mu_app[1])*(mu_app_jx[1] - _prior_mu_app[1]);
  delta_j += (mu_app_jx[2] - _prior_mu_app[2])*(mu_app_jx[2] - _prior_mu_app[2]);

  // -- [shape] sum delats --
  delta_i += (mu_shape_ix[0] - _prior_mu_shape[0])*(mu_shape_ix[0] - _prior_mu_shape[0]);
  delta_i += (mu_shape_ix[1] - _prior_mu_shape[1])*(mu_shape_ix[1] - _prior_mu_shape[1]);
  delta_i += (mu_shape_ix[2] - _prior_mu_shape[2])*(mu_shape_ix[2] - _prior_mu_shape[2]);

  delta_j += (mu_shape_jx[0] - _prior_mu_shape[0])*(mu_shape_jx[0] - _prior_mu_shape[0]);
  delta_j += (mu_shape_jx[1] - _prior_mu_shape[1])*(mu_shape_jx[1] - _prior_mu_shape[1]);
  delta_j += (mu_shape_jx[2] - _prior_mu_shape[2])*(mu_shape_jx[2] - _prior_mu_shape[2]);

  // -- delta --
  // delta = delta_i - delta_j; // compare with "0"

  // -- write to output --
  sims[isp + ntotal*jsp] = delta_i - delta_j;
  living_mask[isp] = true;
  /* sims[isp][jsp] = delta; */
  // *(sims + isp + nliving*jsp) = delta;
}

// -- spawn a new label --

__global__
void relabel_kernel(int* spix, float* sim_vals, int* sim_inds,
                    int* dead, bool* living_mask, int npix, float thresh) {
  // -- relabel the list --
  int ix = threadIdx.x + blockIdx.x * blockDim.x;
  if (ix >= npix) return;
  int s = spix[ix]; // read current superpixel label
  if (living_mask[s] == false){ return; }
  // assert(s "isin" living_superpixels);
  float sim_val = sim_vals[s]; // get value to check if we swap
  int sim_ind = sim_inds[s]; // read location of new id
  int dead_id = dead[sim_ind]; // update the pixel

  if (sim_val > 0){
    printf("relabeling!\n");
    // spix[ix] = dead[sim_ind]; // update the pixel
    return;
  }
  if (sim_val > thresh){
    // ...
  }
  else if (sim_val < thresh){
    // spix[ix] = dead[sim_ind]; // update the pixel
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
    auto options_b = torch::TensorOptions().dtype(torch::kBool)
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
    torch::Tensor all_of_em = torch::arange(0,max_nspix).to(spix.device()).toType(torch::kInt32);
    fprintf(stdout,"unique_ids.dim(),all_of_em.dim(): %d,%d\n",
            unique_ids.dim(),all_of_em.dim());

    auto comp = (unique_ids.unsqueeze(1) != all_of_em.unsqueeze(0));
    fprintf(stdout,"b\n");
    auto mask = comp.all(0);
    fprintf(stdout,"mask.dim(): %d\n",mask.dim());
    fprintf(stdout,"mask.size(0): %d\n",mask.size(0));
    auto dead_th = all_of_em.masked_select(mask);
    fprintf(stdout,"comp.size(0),comp.size(1): %d,%d\n",comp.size(0),comp.size(1));
    fprintf(stdout,"dead_th.size(0): %d\n",dead_th.size(0));
    int ndead = dead_th.size(0);
    int* dead_ids = dead_th.data<int>();
    if (ndead <= 0) { return; }
    /* int nliving = nspix; */
    /* int ndead = nspix; */
    /* int* dead_ids = ids; */

    // -- unpack --
    int* spix_ptr = spix.data<int>();
    float* mu_app = params.mu_app.data<float>();
    float* prior_mu_app = params.prior_mu_app.data<float>();
    double* mu_shape = params.mu_shape.data<double>();
    double* prior_mu_shape = params.prior_mu_shape.data<double>();
    double* sigma_shape = params.sigma_shape.data<double>();
    double* prior_sigma_shape = params.prior_sigma_shape.data<double>();
    float* prior_counts = params.prior_counts.data<float>();
    int* counts = params.counts.data<int>();
    // a list of DEAD ids is needed

    // -- compare each living spix with dead ones  --
    // torch::Tensor sims = torch::zeros({nliving,ndead},options_f32);
    torch::Tensor sims = -1000.*torch::ones({max_nspix,ndead},options_f32);
    torch::Tensor living_mask = torch::zeros({max_nspix},options_b);
    float* sims_ptr = sims.data<float>();
    bool* living_mask_ptr = living_mask.data<bool>();
    int num_blocks = ceil( double(max_nspix) / double(THREADS_PER_BLOCK) );
    dim3 nblocks(num_blocks);
    dim3 nthreads(THREADS_PER_BLOCK);
    search_kernel<<<nblocks,nthreads>>>(sims_ptr, living_mask_ptr,
                                        mu_app, prior_mu_app,
                                        mu_shape, prior_mu_shape, sigma_shape,
                                        prior_sigma_shape, counts, prior_counts,
                                        living_ids,dead_ids,nliving,ndead);
    printf(".\n");

    // -- find the minimum --
    auto result = torch::max(sims, 1);
    torch::Tensor minValues = std::get<0>(result);
    torch::Tensor minIndices = std::get<1>(result).toType(torch::kInt32);
    printf("minIndices.min(), minIndices.max() minIndices.size(0): %d,%d,%d\n",
           minIndices.min().item<int>(),minIndices.max().item<int>(),
           minIndices.size(0));
    float* sim_vals = minValues.data<float>();
    int* sim_inds = minIndices.data<int>();


    // -- relabel --
    int num_blocks1 = ceil( double(npix) / double(THREADS_PER_BLOCK) );
    dim3 nblocks1(num_blocks1);
    dim3 nthreads1(THREADS_PER_BLOCK);
    relabel_kernel<<<nblocks1,nthreads1>>>(spix_ptr, sim_vals, sim_inds,
                                           dead_ids, living_mask_ptr, npix, thresh);
}



void init_relabel(py::module &m){
  m.def("relabel", &run_relabel,"relabel living labels from the dead #zombie");
}
