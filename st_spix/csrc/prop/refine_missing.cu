
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

// -- local import --
#include "refine_missing.h"
#include "seg_utils.h"
#include "update_prop_params.h"
#include "update_missing_seg.h"

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
                             superpixel_GPU_helper* sp_gpu_helper,
                             int* prev_means, int* prev_spix,
                             int* missing, bool* border,
                             int niters, int niters_seg,
                             float3 pix_cov,float logdet_pix_cov,float potts,
                             int nbatch, int width, int height, int nspix){

  // "border" & "sp_gpu_helper" _maybe_ be allocated here.
    
    // -- init --
    int nspix_buffer = nspix * 45;
    for (int i = 0; i < niters; i++) {

      // -- Update Parameters with Previous Frame --
      update_prop_params(img, seg, sp_params, sp_gpu_helper,
                         prev_means, prev_spix, npix, nspix,
                         nspix_buffer, nbatch, dim_x, dim_y, nftrs);

      // -- Update Segmentation ONLY within missing pix --
      update_missing_seg(img, seg, border, missing, sp_params,
                         niters_seg, pix_cov, logdet_pix_cov, potts,
                         npix, nspix, nbatch, dim_x, dim_y, nfts);


    }

    CudaFindBorderPixels_end(seg, border, npix, nbatch, dim_x, dim_y, 1);
}


/**********************************************************

             -=-=-=-=- Python API  -=-=-=-=-=-

***********************************************************/

torch::Tensor run_refine_missing(const torch::Tensor img,
                                 const torch::Tensor spix,
                                 const torch::Tensor missing,
                                 const torch::Tensor prev_spix,
                                 const torch::Tensor prev_means,
                                 int nspix, int niters, int niters_seg,
                                 float pix_cov_i, float potts){

    // -- check --
    CHECK_INPUT(img);
    CHECK_INPUT(spix);
    CHECK_INPUT(missing);
    CHECK_INPUT(prev_spix);
    CHECK_INPUT(prev_means);

    // -- unpack --
    int nbatch = spix.size(0);
    int height = spix.size(1);
    int width = spix.size(2);
    int nftrs = img.size(3);
    int npix = height*width;
    int nmissing = missing.size(1);

    // -- allocate filled spix --
    auto options_i32 = torch::TensorOptions().dtype(torch::kInt32)
      .layout(torch::kStrided).device(imgs.device());
    torch::Tensor filled_spix = spix.clone();
    int* filled_spix_ptr = filled_spix.data<int>();
    assert(nbatch==1);

    // -- allocate border --
    bool* border;
    try {
      throw_on_cuda_error(cudaMalloc((void**)&border,nbatch*npix*sizeof(bool)));
      // throw_on_cuda_error(malloc((void*)num_neg_cpu,sizeofint));
    }
    catch (thrust::system_error& e) {
        std::cerr << "CUDA error after cudaMalloc: " << e.what() << std::endl;
    }

    // -- init parameters --
    superpixel_params* sp_params;
    superpixel_GPU_helper* sp_gpu_helper,

    // -- compute pixel (inverse) covariance info --
    float pix_half = float(pix_cov_i/2) * float(pix_cov_i/2);
    float3 pix_cov;
    pix_cov.x = 1.0/pix_half;
    pix_cov.y = 1.0/pix_half;
    pix_cov.z = 1.0/pix_half;
    float logdet_pix_cov = log(pix_half * pix_half * pix_half);

    // -- get pointers --
    float* img_ptr = img.data<float>();
    float* prev_means_ptr = prev_means.data<float>();
    int* prev_spix_ptr = prev_spix.data<float>();
    int* missing_ptr = missing.data<int>();

    // -- run fill --
    if (nmissing>0){
      refine_missing(img_ptr,filled_spix_ptr,sp_params,sp_gpu_helper,
                     prev_means_ptr, prev_spix_ptr, missing_ptr, border,
                     niters, niters_seg, pix_cov, logdet_pix_cov, potts,
                     nbatch, width, height, nspix){
    }
    cudaFree(border);

    return filled_spix;
}

void init_refine_missing(py::module &m){
  m.def("refine_missing", &run_refine_missing,"refine missing labels");
}

