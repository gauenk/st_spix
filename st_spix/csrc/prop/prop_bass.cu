
/********************************************************************

      Run BASS using the propograted superpixel segs and params

********************************************************************/


// -- cpp imports --
#include <stdio.h>
#include "pch.h"

// -- "external" import --
#ifndef MY_SP_STRUCT
#define MY_SP_STRUCT
#include "../bass/share/my_sp_struct.h"
#endif

// -- utils --
#include "rgb2lab.h"
#include "init_utils.h"
#include "init_sparams.h"
// #include "simple_init_sparams.h"
#include "seg_utils.h"
#include "sparams_io.h"
// #include "simple_sparams_io.h"

// -- primary functions --
#include "prop_bass.h"
// #include "simple_split_merge.h"
#include "split_merge_prop.h"
// #include "update_prop_params.h"
// #include "update_prop_seg.h"
#include "update_params.h"
#include "update_seg.h"


// -- define --
#define CHECK_CUDA(x) TORCH_CHECK(x.device().is_cuda(), #x " must be a CUDA tensor")
#define CHECK_CONTIGUOUS(x) TORCH_CHECK(x.is_contiguous(), #x " must be contiguous")
#define CHECK_INPUT(x) CHECK_CUDA(x); CHECK_CONTIGUOUS(x)
#define THREADS_PER_BLOCK 512


/**********************************************************

             -=-=-=-=- Main Function -=-=-=-=-=-

***********************************************************/

__host__ int prop_bass(float* img, int* seg,
                       spix_params* sp_params, bool* border,
                       spix_helper* sp_helper, spix_helper_sm* sm_helper,
                       int* sm_seg1 ,int* sm_seg2, int* sm_pairs,
                       int niters, int niters_seg, int sm_start,
                       float sigma2_app, float potts, float alpha_hastings,
                       int nspix, int nspix_buffer,
                       int nbatch, int width, int height, int nftrs){

    // -- init --
    int count = 1;
    int npix = height * width;
    // int nspix_buffer = nspix * 45;
    int max_spix = nspix;

    // -- run splits --
    count = 0;
    for (int idx = 0; idx < 1; idx++) { // num of different splits; each only once.
      max_spix = run_split_prop(img, seg, border, sp_params,
                                sp_helper, sm_helper, sm_seg1, sm_seg2, sm_pairs,
                                alpha_hastings, sigma2_app, count, idx, max_spix,
                                npix,nbatch,width,height,nftrs,nspix_buffer);
      update_params(img, seg, sp_params, sp_helper, sigma2_app,
                    npix, nspix_buffer, nbatch, width, nftrs);
    }

    // -- refine --
    for (int idx = 0; idx < niters; idx++) {
      // -- Update Parameters with Previous SuperpixelParams as Prior --
      update_params(img, seg, sp_params, sp_helper, sigma2_app,
                    npix, nspix_buffer, nbatch, width, nftrs);

      // -- Update Segmentation --
      update_seg(img, seg, border, sp_params, niters_seg,
                 sigma2_app, potts, npix, nbatch, width, height, nftrs);
    }

    // // -- run merge --
    // for (int idx = 0; idx < 4; idx++) {
    //   run_merge_prop(img, seg, border, sp_params,
    //                  sp_helper, sm_helper, sm_seg1, sm_seg2, sm_pairs,
    //                  alpha_hastings, sigma2_app, count, idx, max_spix, nspix,
    //                  npix,nbatch,width,height,nftrs,nspix_buffer);
    // }

    // -- apply changes from merge --
    update_params(img, seg, sp_params, sp_helper, sigma2_app,
                  npix, nspix_buffer, nbatch, width, nftrs);
    // update_seg(img, seg, border, sp_params,
    //            niters_seg, sigma2_app, potts,
    //            npix, nbatch, width, height, nftrs);

    // -- relabel from previou spix -- ?
    // relabel_from_history(....);

    // -- final border [legacy code; idk why we keep it] --
    CudaFindBorderPixels_end(seg, border, npix, nbatch, width, height);
    return max_spix;
}


/**********************************************************

             -=-=-=-=- Python API  -=-=-=-=-=-

***********************************************************/

std::tuple<torch::Tensor,PySuperpixelParams>
run_prop_bass(const torch::Tensor img, const torch::Tensor spix,
              const PySuperpixelParams prior_params,
              int niters, int niters_seg, int sm_start,
              int sp_size, float pix_var_i, float potts, float alpha_hastings){

    // -- check --
    CHECK_INPUT(img);
    CHECK_INPUT(spix);
    CHECK_INPUT(prior_params.mu_app);
    CHECK_INPUT(prior_params.mu_shape);
    CHECK_INPUT(prior_params.sigma_shape);
    CHECK_INPUT(prior_params.logdet_sigma_shape);
    CHECK_INPUT(prior_params.counts);
    CHECK_INPUT(prior_params.prior_counts);

    // -- unpack --
    int nbatch = spix.size(0);
    int height = spix.size(1);
    int width = spix.size(2);
    int nftrs = img.size(3);
    int npix = height*width;
    int nspix = prior_params.ids.size(0);

    // -- allocate filled spix --
    auto options_i32 = torch::TensorOptions().dtype(torch::kInt32)
      .layout(torch::kStrided).device(spix.device());
    auto options_f64 = torch::TensorOptions().dtype(torch::kFloat64)
      .layout(torch::kStrided).device(spix.device());
    torch::Tensor filled_spix = spix.clone();
    assert(nbatch==1);

    // -- allocate memory --
    // int nspix_buffer = nspix*50;
    int nspix_buffer = nspix*10;
    const int sparam_size = sizeof(spix_params);
    const int helper_size = sizeof(spix_helper);
    bool* border = (bool*)easy_allocate(nbatch*npix,sizeof(bool));
    spix_params* sp_params = get_tensors_as_params(prior_params,sp_size,
                                                         npix,nspix,nspix_buffer);
    // spix_params* sp_params=(spix_params*)easy_allocate(nspix_buffer,sparam_size);
    spix_helper* sp_helper=(spix_helper*)easy_allocate(nspix_buffer,helper_size);

    // -- init sp params from past --
    // assert(rescales.size(0) == 4);// must be of size 4
    // float4 rescale; // marked for deletion
    // rescale.x = rescales[0].item<int>();
    // rescale.y = rescales[1].item<int>();
    // rescale.z = rescales[2].item<int>();
    // rescale.w = rescales[3].item<int>();
    //init_sp_params_from_past(sp_params,prior_sp_params,rescale,nspix,nspix_buffer,npix);

    // -- compute pixel (inverse) variance info --
    float pix_half = float(pix_var_i/2) * float(pix_var_i/2);
    float sigma2_app =  pix_var_i;//1.0/pix_half;
    // float pix_var = std::sqrt(1./(4*pix_ivar.x));

    // pix_var.x = 1.0/pix_half;
    // pix_var.y = 1.0/pix_half;
    // pix_var.z = 1.0/pix_half;
    // float logdet_pix_var = log(pix_half * pix_half * pix_half);

    // -- Get pointers --
    float* img_ptr = img.data<float>();
    int* filled_spix_ptr = filled_spix.data<int>();

    // -- split/merge memory --
    const int sm_helper_size = sizeof(spix_helper_sm);
    int* sm_seg1 = (int*)easy_allocate(npix,sizeof(int));
    int* sm_seg2 = (int*)easy_allocate(npix,sizeof(int));
    int* sm_pairs = (int*)easy_allocate(2*npix,sizeof(int));
    spix_helper_sm* sm_helper=(spix_helper_sm*)easy_allocate(nspix_buffer,sm_helper_size);

    // -- allocate larger memory for prior map --
    // int* prior_map_ptr = (int*)easy_allocate(nspix_buffer,sizeof(int));
    // cudaMemset(prior_map_ptr, -1, nspix_buffer*sizeof(int));
    // cudaMemcpy(prior_map_ptr,prior_map_r_ptr,
    //            init_map_size*sizeof(int),cudaMemcpyDeviceToDevice);

    // -- init superpixel params --
    // float prior_sigma_app = float(pix_var_i/2) * float(pix_var_i/2);
    // init_sp_params_from_past(sp_params,prior_sp_params,prior_map_ptr,
    //                          rescale,nspix,nspix_buffer,npix);

    // -=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-
    //
    //                 Run BASS
    //
    // -=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-

    // fprintf(stdout,"max_spix: %d\n",max_spix);
    int max_spix = prop_bass(img_ptr,filled_spix_ptr,sp_params,
                             border,sp_helper, sm_helper, sm_seg1, sm_seg2, sm_pairs,
                             niters, niters_seg, sm_start, sigma2_app, potts,
                             alpha_hastings, nspix, nspix_buffer,
                             nbatch, width, height, nftrs);
    // fprintf(stdout,"max_spix: %d\n",max_spix);

    // -- ensure new superpixels are compactly added to previous superpixels --
    int prev_max_spix = prior_params.ids.size(0);
    // fprintf(stdout,"prev_max_spix: %d\n",prev_max_spix);
    max_spix = compactify_new_superpixels(filled_spix,sp_params,
                                          prev_max_spix,max_spix,npix);


    // -- get spixel parameters as tensors --
    auto unique_ids = std::get<0>(at::_unique(filled_spix));
    auto ids = unique_ids.data<int>();
    int num_ids = unique_ids.sizes()[0];
    PySuperpixelParams params = get_output_params(sp_params,prior_params,
                                                  ids, num_ids, max_spix);
    run_update_prior(spix,params); // shift estimates to prior information @ spix

    // -- free --
    cudaFree(border);
    cudaFree(sp_params);
    cudaFree(sp_helper);
    cudaFree(sm_helper);
    cudaFree(sm_seg1);
    cudaFree(sm_seg2);
    cudaFree(sm_pairs);

    return std::make_tuple(filled_spix,params);
}

void init_prop_bass(py::module &m){
  m.def("prop_bass", &run_prop_bass,"run propogated bass");
}

