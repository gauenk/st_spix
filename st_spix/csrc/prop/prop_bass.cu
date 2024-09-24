
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
#include "simple_init_sparams.h"
#include "seg_utils.h"
#include "simple_sparams_io.h"

// -- primary functions --
#include "prop_bass.h"
#include "simple_split_merge.h"
#include "update_prop_params.h"
#include "update_prop_seg.h"

// -- define --
#define CHECK_CUDA(x) TORCH_CHECK(x.device().is_cuda(), #x " must be a CUDA tensor")
#define CHECK_CONTIGUOUS(x) TORCH_CHECK(x.is_contiguous(), #x " must be contiguous")
#define CHECK_INPUT(x) CHECK_CUDA(x); CHECK_CONTIGUOUS(x)
#define THREADS_PER_BLOCK 512


/**********************************************************

             -=-=-=-=- Main Function -=-=-=-=-=-

***********************************************************/

__host__ void prop_bass(float* img, int* seg,
                        superpixel_params* sp_params,
                        superpixel_params* prior_params,
                        int* prior_map, bool* border,
                        superpixel_GPU_helper* sp_helper,
                        superpixel_GPU_helper_sm* sm_helper,
                        int* sm_seg1 ,int* sm_seg2, int* sm_pairs,
                        int niters, int niters_seg, int sm_start,
                        float3 pix_ivar,float logdet_pix_var,
                        float potts, float alpha_hastings,
                        int nspix, int nbatch, int width, int height, int nftrs){

    // -- init --
    int count = 1;
    int npix = height * width;
    int nspix_buffer = nspix * 45;
    int max_spix = nspix;
    float pix_var = std::sqrt(1./(4*pix_ivar.x));

    for (int idx = 0; idx < niters; idx++) {

      // -- Update Parameters with Previous SuperpixelParams as Prior --
      update_prop_params(img, seg, sp_params, sp_helper,
                         prior_params, prior_map, npix,
                         nspix_buffer, nbatch, width, nftrs);

      // -- Run Split/Merge --
      if (idx >= sm_start){
        if(idx%4 == 0){
          max_spix = run_simple_split(img, seg, border, sp_params,
                                      prior_params, prior_map,
                                      sp_helper, sm_helper, sm_seg1, sm_seg2, sm_pairs,
                                      alpha_hastings, pix_var, count, idx, max_spix,
                                      npix,nbatch,width,height,nftrs,nspix_buffer);
          update_prop_params(img, seg, sp_params, sp_helper,
                             prior_params, prior_map, npix,
                             nspix_buffer, nbatch, width, nftrs);
        }
        if( idx%4 == 2){
          run_simple_merge(img, seg, border, sp_params,
                           prior_params, prior_map,
                           sp_helper, sm_helper, sm_seg1, sm_seg2, sm_pairs,
                           alpha_hastings, pix_var, count, idx, max_spix,
                           npix,nbatch,width,height,nftrs,nspix_buffer);
          update_prop_params(img, seg, sp_params, sp_helper,
                             prior_params, prior_map, npix,
                             nspix_buffer, nbatch, width, nftrs);
        }
      }

      // -- Update Segmentation --
      update_prop_seg(img, seg, border, sp_params,
                      niters_seg, pix_ivar, logdet_pix_var, potts,
                      npix, nbatch, width, height, nftrs);

    }

    CudaFindBorderPixels_end(seg, border, npix, nbatch, width, height);

}


/**********************************************************

             -=-=-=-=- Python API  -=-=-=-=-=-

***********************************************************/

std::tuple<torch::Tensor,PySuperpixelParams>
run_prop_bass(const torch::Tensor img_rgb,
              const torch::Tensor spix,
              const PySuperpixelParams prior_params,
              const torch::Tensor prior_map, int nspix,
              int niters, int niters_seg, int sm_start,
              int sp_size, float pix_var_i, float potts, float alpha_hastings){

    // -- check --
    CHECK_INPUT(img_rgb);
    CHECK_INPUT(spix);
    CHECK_INPUT(prior_params.mu_app);
    CHECK_INPUT(prior_params.mu_shape);
    CHECK_INPUT(prior_params.sigma_shape);
    CHECK_INPUT(prior_params.logdet_sigma_shape);
    CHECK_INPUT(prior_params.counts);
    CHECK_INPUT(prior_params.prior_counts);
    CHECK_INPUT(prior_map);

    // -- unpack --
    int nbatch = spix.size(0);
    int height = spix.size(1);
    int width = spix.size(2);
    int nftrs = img_rgb.size(3);
    int npix = height*width;
    int init_map_size = prior_map.size(0);
    
    // -- allocate filled spix --
    auto options_i32 = torch::TensorOptions().dtype(torch::kInt32)
      .layout(torch::kStrided).device(spix.device());
    torch::Tensor filled_spix = spix.clone();
    assert(nbatch==1);

    // -- allocate memory --
    int nspix_buffer = nspix*50;
    const int sparam_size = sizeof(superpixel_params);
    const int helper_size = sizeof(superpixel_GPU_helper);
    bool* border = (bool*)easy_allocate(nbatch*npix,sizeof(bool));
    superpixel_params* prior_sp_params = get_tensors_as_params_s(prior_params,sp_size,
                                                                 npix,nspix,nspix_buffer);
    superpixel_params* sp_params=(superpixel_params*)easy_allocate(nspix_buffer,
                                                                   sparam_size);
    superpixel_GPU_helper* sp_helper=(superpixel_GPU_helper*)easy_allocate(nspix_buffer,
                                                                           helper_size);
    init_sp_params_s(sp_params,sp_size,nspix,nspix_buffer,npix);

    // -- compute pixel (inverse) covariance info --
    float pix_half = float(pix_var_i/2) * float(pix_var_i/2);
    float3 pix_var;
    pix_var.x = 1.0/pix_half;
    pix_var.y = 1.0/pix_half;
    pix_var.z = 1.0/pix_half;
    float logdet_pix_var = log(pix_half * pix_half * pix_half);

    // -- convert image color --
    auto img_lab = img_rgb.clone();
    rgb2lab(img_rgb.data<float>(),img_lab.data<float>(),npix,nbatch);

    // -- Get pointers --
    float* img_ptr = img_lab.data<float>();
    int* filled_spix_ptr = filled_spix.data<int>();
    int* prior_map_r_ptr = prior_map.data<int>();

    // -- allocate larger memory for prior map --
    int* prior_map_ptr = (int*)easy_allocate(nspix_buffer,sizeof(int));
    cudaMemset(prior_map_ptr, -1, nspix_buffer*sizeof(int));
    cudaMemcpy(prior_map_ptr,prior_map_r_ptr,
               init_map_size*sizeof(int),cudaMemcpyDeviceToDevice);
    const int sm_helper_size = sizeof(superpixel_GPU_helper_sm);
    int* sm_seg1 = (int*)easy_allocate(npix,sizeof(int));
    int* sm_seg2 = (int*)easy_allocate(npix,sizeof(int));
    int* sm_pairs = (int*)easy_allocate(2*npix,sizeof(int));
    superpixel_GPU_helper_sm* sm_helper = (superpixel_GPU_helper_sm*)easy_allocate(nspix_buffer,sm_helper_size);



    // -=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-
    //
    //                 Run BASS
    //
    // -=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-

    prop_bass(img_ptr,filled_spix_ptr,sp_params, prior_sp_params, prior_map_ptr,
              border, sp_helper, sm_helper, sm_seg1, sm_seg2, sm_pairs,
              niters, niters_seg, sm_start, pix_var, logdet_pix_var,
              potts, alpha_hastings, nspix, nbatch, width, height, nftrs);

    // -- get spixel parameters as tensors --
    auto unique_ids = std::get<0>(at::_unique(filled_spix));
    auto ids = unique_ids.data<int>();
    int nspix_post = unique_ids.sizes()[0];
    PySuperpixelParams params = get_params_as_tensors_s(sp_params,ids,nspix_post);

    // -- free --
    cudaFree(prior_map_ptr);
    cudaFree(border);
    cudaFree(sp_params);
    cudaFree(sp_helper);
    cudaFree(sm_seg1);
    cudaFree(sm_seg2);
    cudaFree(sm_pairs);

    return std::make_tuple(filled_spix,params);
}

void init_prop_bass(py::module &m){
  m.def("prop_bass", &run_prop_bass,"run propogated bass");
}

