
/*******************************************************

      - This finds a likely superpixel state
      after missing pixels are filled-in using "filled.cu",
      and after splitting with "split_disconnected.du".

      - This is necessary, because the "filled" superpixels are
      not in a likely state after the shift. The missing pixels have
      merely been assigned their spatial neighbor. This section of
      code actually runs BASS using the posterior of the parameter estimates.

      - This section of the code is different from "XXXX.cu"
      because updates can only effect the "missing" region.

*******************************************************/

// -- cpp imports --
#include <stdio.h>
#include <cuda.h>
#include <cuda_runtime.h>
#include <cuda/std/type_traits>
#include <torch/types.h>
#include <cuda.h>
#include <cuda_runtime.h>
#include <torch/extension.h>

// -- "external" import --
#ifndef MY_SP_STRUCT
#define MY_SP_STRUCT
#include "../bass/share/my_sp_struct.h"
#endif

// -- local import --
#include "rgb2lab.h"
#include "init_utils.h"
#include "seg_utils.h"
#include "refine_missing.h"
#include "update_prop_params.h"
#include "update_missing_seg.h"
#include "sparams_io.h"

// -- define --
#define CHECK_CUDA(x) TORCH_CHECK(x.device().is_cuda(), #x " must be a CUDA tensor")
#define CHECK_CONTIGUOUS(x) TORCH_CHECK(x.is_contiguous(), #x " must be contiguous")
#define CHECK_INPUT(x) CHECK_CUDA(x); CHECK_CONTIGUOUS(x)
#define THREADS_PER_BLOCK 512


/**********************************************************

             -=-=-=-=- Main Function -=-=-=-=-=-

***********************************************************/

__host__ void refine_missing(float* img, int* seg,
                             superpixel_params* sp_params,
                             superpixel_params* prior_params,
                             int* prior_map, bool* missing, bool* border,
                             superpixel_GPU_helper* sp_helper,
                             int niters, int niters_seg,
                             float3 pix_cov,float logdet_pix_cov,float potts,
                             int nspix, int nbatch, int width, int height, int nftrs){

    // -- init --
    int npix = height * width;
    int nspix_buffer = nspix * 45;
    for (int i = 0; i < niters; i++) {

      // -- Update Parameters with Previous SuperpixelParams as Prior --
      update_prop_params(img, seg, sp_params, sp_helper,
                         prior_params, prior_map, npix, nspix,
                         nspix_buffer, nbatch, width, height, nftrs);

      // -- Update Segmentation ONLY within missing pix --
      update_missing_seg(img, seg, border, missing, sp_params,
                         niters_seg, pix_cov, logdet_pix_cov, potts,
                         npix, nspix, nbatch, width, height, nftrs);

    }

    CudaFindBorderPixels_end(seg, border, npix, nbatch, width, height);
}


/**********************************************************

             -=-=-=-=- Python API  -=-=-=-=-=-

***********************************************************/

// torch::Tensor
std::tuple<torch::Tensor,PySuperpixelParams>
run_refine_missing(const torch::Tensor img_rgb,
                   const torch::Tensor spix,
                   const torch::Tensor missing,
                   const PySuperpixelParams prior_params,
                   const torch::Tensor prior_map,
                   int nspix, int niters, int niters_seg,
                   int sp_size, float pix_cov_i, float potts){

    // -- check --
    CHECK_INPUT(img_rgb);
    CHECK_INPUT(spix);
    CHECK_INPUT(missing);
    CHECK_INPUT(prior_params.mu_i);
    CHECK_INPUT(prior_params.mu_s);
    CHECK_INPUT(prior_params.sigma_s);
    CHECK_INPUT(prior_params.logdet_Sigma_s);
    CHECK_INPUT(prior_params.counts);
    CHECK_INPUT(prior_params.prior_counts);
    CHECK_INPUT(prior_map);

    // -- unpack --
    int nbatch = spix.size(0);
    int height = spix.size(1);
    int width = spix.size(2);
    int nftrs = img_rgb.size(3);
    int npix = height*width;
    int nmissing = missing.sum().item<int>();
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
    superpixel_params* prior_sp_params = get_tensors_as_params(prior_params,sp_size,
                                                               npix,nspix,nspix_buffer);
    superpixel_params* sp_params=(superpixel_params*)easy_allocate(nspix_buffer,
                                                                   sparam_size);
    superpixel_GPU_helper* sp_helper=(superpixel_GPU_helper*)easy_allocate(nspix_buffer,
                                                                           helper_size);
    init_sp_params(sp_params,sp_size,nspix,nspix_buffer,npix);

    // bool* border = allocate_border(nbatch*npix);
    // superpixel_params* sp_params = allocate_sp_params(nspix_buffer);
    // superpixel_GPU_helper* sp_helper = allocate_sp_helper(nspix_buffer);
    // init_sp_params(sp_params,sp_size,nspix,nspix_buffer,npix);

    // -- compute pixel (inverse) covariance info --
    float pix_half = float(pix_cov_i/2) * float(pix_cov_i/2);
    float3 pix_cov;
    pix_cov.x = 1.0/pix_half;
    pix_cov.y = 1.0/pix_half;
    pix_cov.z = 1.0/pix_half;
    float logdet_pix_cov = log(pix_half * pix_half * pix_half);

    // -- convert image color --
    auto img_lab = img_rgb.clone();
    rgb2lab(img_rgb.data<float>(),img_lab.data<float>(),npix,nbatch);

    // -- Get pointers --
    float* img_ptr = img_lab.data<float>();
    int* filled_spix_ptr = filled_spix.data<int>();
    bool* missing_ptr = missing.data<bool>();
    int* prior_map_r_ptr = prior_map.data<int>();

    // -- allocate larger memory for prior map --
    int* prior_map_ptr = (int*)easy_allocate(nspix_buffer,sizeof(int));
    cudaMemset(prior_map_ptr, -1, nspix_buffer*sizeof(int));
    cudaMemcpy(prior_map_ptr,prior_map_r_ptr,
               init_map_size*sizeof(int),cudaMemcpyDeviceToDevice);

    // -- run fill --
    if (nmissing>0){
      refine_missing(img_ptr,filled_spix_ptr,sp_params,
                     prior_sp_params, prior_map_ptr, missing_ptr, border,
                     sp_helper, niters, niters_seg, pix_cov, logdet_pix_cov,
                     potts, nspix, nbatch, width, height, nftrs);
    }


    // -- get spixel parameters as tensors --
    auto unique_ids = std::get<0>(at::_unique(filled_spix));
    auto ids = unique_ids.data<int>();
    int nspix_post = unique_ids.sizes()[0];
    PySuperpixelParams params = get_params_as_tensors(sp_params,ids,nspix_post);

    // -- free --
    cudaFree(border);
    cudaFree(sp_params);
    cudaFree(sp_helper);

    return std::make_tuple(filled_spix,params);
}

void init_refine_missing(py::module &m){
  m.def("refine_missing", &run_refine_missing,"refine missing labels");
}

