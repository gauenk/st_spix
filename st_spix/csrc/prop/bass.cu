
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
#include "sparams_io.h"
#include "seg_utils.h"
#include "init_utils.h"
#include "init_sparams.h"
#include "init_seg.h"

// -- primary functions --
#include "prop_bass.h"
#include "split_merge.h"
#include "update_params.h"
#include "update_seg.h"


/**********************************************************

             -=-=-=-=- Main Function -=-=-=-=-=-

***********************************************************/

__host__ void bass(float* img, int* seg,spix_params* sp_params,bool* border,
                   spix_helper* sp_helper,spix_helper_sm* sm_helper,
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
      update_params(img, seg, sp_params, sp_helper,
                    npix, nspix_buffer, nbatch, width, nftrs);

      // -- Run Split/Merge --
      if ((sm_start <=0) or (idx > sm_start)){
        // printf("split merge.\n");
        max_spix = run_split_merge(img, seg, border, sp_params,
                                   sp_helper, sm_helper, sm_seg1, sm_seg2, sm_pairs,
                                   alpha_hastings, pix_var, count, idx, max_spix,
                                   npix,nbatch,width,height,nftrs,nspix_buffer);
        update_params(img, seg, sp_params, sp_helper,
                      npix, nspix_buffer, nbatch, width, nftrs);
      }

      // -- Update Segmentation --
      update_seg(img, seg, border, sp_params,
                 niters_seg, pix_ivar, logdet_pix_var, potts,
                 npix, nbatch, width, height, nftrs);

    }

    CudaFindBorderPixels_end(seg, border, npix, nbatch, width, height);

}

/**********************************************************

             -=-=-=-=- Python API  -=-=-=-=-=-

***********************************************************/

std::tuple<torch::Tensor,PySuperpixelParams>
run_bass(const torch::Tensor img_rgb,
         int niters, int niters_seg, int sm_start,
         int sp_size, float pix_var_i, float potts, float alpha_hastings){

    // -- check --
    CHECK_INPUT(img_rgb);

    // -- unpack --
    int nbatch = img_rgb.size(0);
    int height = img_rgb.size(1);
    int width = img_rgb.size(2);
    int nftrs = img_rgb.size(3);
    int npix = height*width;
    assert(nbatch==1);    

    // -- allocate filled spix --
    auto options_i32 = torch::TensorOptions().dtype(torch::kInt32)
      .layout(torch::kStrided).device(img_rgb.device());
    torch::Tensor spix = torch::zeros({nbatch,height,width},options_i32);

    // -- init superpixels --
    int nspix = init_seg(spix.data<int>(),sp_size,width,height,nbatch);

    // -- allocate memory --
    int nspix_buffer = nspix*50;
    const int sparam_size = sizeof(spix_params);
    const int helper_size = sizeof(spix_helper);
    bool* border = (bool*)easy_allocate(nbatch*npix,sizeof(bool));
    spix_params* sp_params=(spix_params*)easy_allocate(nspix_buffer,sparam_size);
    spix_helper* sp_helper=(spix_helper*)easy_allocate(nspix_buffer,helper_size);

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
    int* spix_ptr = spix.data<int>();

    // -- allocate larger memory for prior map --
    const int sm_helper_size = sizeof(spix_helper_sm);
    int* sm_seg1 = (int*)easy_allocate(npix,sizeof(int));
    int* sm_seg2 = (int*)easy_allocate(npix,sizeof(int));
    int* sm_pairs = (int*)easy_allocate(2*npix,sizeof(int));
    spix_helper_sm* sm_helper=(spix_helper_sm*)easy_allocate(nspix_buffer,sm_helper_size);

    // -=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-
    //
    //                 Run BASS
    //
    // -=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-

    // -- init spix_params --
    init_sp_params(sp_params,img_ptr,spix_ptr,sp_helper,
                   npix,nspix_buffer,nbatch,width,nftrs);
    //                  int npix, int nspix_buffer,ftrs);
    //              int nbatch, int width, int nftrs)
    // init_sp_params(sp_paramsimg,,nspix,nspix_buffer,npix);

    // -- run method --
    bass(img_ptr, spix_ptr, sp_params,
         border, sp_helper, sm_helper, sm_seg1, sm_seg2, sm_pairs,
         niters, niters_seg, sm_start, pix_var, logdet_pix_var,
         potts, alpha_hastings, nspix, nbatch, width, height, nftrs);

    // -- get spixel parameters as tensors --
    auto unique_ids = std::get<0>(at::_unique(spix));
    auto ids = unique_ids.data<int>();
    int nspix_post = unique_ids.sizes()[0];
    // PySuperpixelParams params;
    PySuperpixelParams params = get_params_as_tensors(sp_params,ids,nspix_post);


    // -- free --
    cudaFree(border);
    cudaFree(sp_params);
    cudaFree(sp_helper);
    cudaFree(sm_seg1);
    cudaFree(sm_seg2);
    cudaFree(sm_pairs);

    return std::make_tuple(spix,params);
}

void init_bass(py::module &m){
  m.def("bass", &run_bass,"run bass");
}

