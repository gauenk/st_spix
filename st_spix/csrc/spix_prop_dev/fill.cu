
#include <stdio.h>
#include <math.h>
#include <cuda.h>
#include <cuda_runtime.h>
#include <cmath>

#include <cuda/std/type_traits>
#include <torch/types.h>
#include <cuda.h>
#include <cuda_runtime.h>
#include <torch/extension.h>
#include <vector>

// #include <thrust/host_vector.h>
// #include <thrust/device_vector.h>
// #include <thrust/copy.h>
// #include <thrust/fill.h>
// #include <thrust/reduce.h>

// -- local import --
#ifndef MY_PROP_SP_STRUCT
#define MY_PROP_SP_STRUCT
#include "../bass/share/refine.h"
#include "../bass/core/Superpixels.h"
#include "../bass/sparams_io.h"
// #include "../bass/share/my_sp_struct.h"
#endif
// #include "../bass/core/Superpixels.h"
#include "calc_prop_seg.h"
#include "init_prop_seg.h"
#include "init_prop_seg_space.h"
#ifndef SPLIT_DISC
#define SPLIT_DISC
#include "split_disconnected.h"
#endif



// -- define --
#define CHECK_CUDA(x) TORCH_CHECK(x.device().is_cuda(), #x " must be a CUDA tensor")
#define CHECK_CONTIGUOUS(x) TORCH_CHECK(x.is_contiguous(), #x " must be contiguous")
#define CHECK_INPUT(x) CHECK_CUDA(x); CHECK_CONTIGUOUS(x)
#define THREADS_PER_BLOCK 512

std::tuple<torch::Tensor,torch::Tensor,torch::Tensor,torch::Tensor,torch::Tensor,
             torch::Tensor,torch::Tensor,torch::Tensor,torch::Tensor,torch::Tensor>
run_fill(const torch::Tensor imgs,
         const torch::Tensor in_spix,
         const torch::Tensor in_missing,
         const torch::Tensor in_means,
         const torch::Tensor in_cov,
         const torch::Tensor in_counts,
         int nPixels_in_square_side, float i_std,
         float alpha, float beta,
         int niters, int inner_niters, int niters_refine,
         int nspix, int max_SP, bool debug_fill, bool prop_type,
         bool use_transition){


    // -- check --
    CHECK_INPUT(imgs);
    CHECK_INPUT(in_spix);

    // -- unpack --
    int nbatch = imgs.size(0);
    int height = imgs.size(1);
    int width = imgs.size(2);
    int nftrs = imgs.size(3);
    int nPix = height*width;
    int nMissing = in_missing.size(1);
    auto options_i32 =torch::TensorOptions().dtype(torch::kInt32)
      .layout(torch::kStrided).device(imgs.device());
    auto options_bool =torch::TensorOptions().dtype(torch::kBool)
      .layout(torch::kStrided).device(imgs.device());
    auto options_f32 =torch::TensorOptions().dtype(torch::kFloat32)
      .layout(torch::kStrided).device(imgs.device());
    auto options_b =torch::TensorOptions().dtype(torch::kBool)
      .layout(torch::kStrided).device(imgs.device());
    int* seg_gpu = in_spix.data<int>();


    // -- init superpixels --
    superpixel_options spoptions = get_sp_options(nPixels_in_square_side,
                                                  i_std, beta, alpha);
    if (niters >= 0){ spoptions.nEMIters = niters; }
    if (inner_niters >= 0){ spoptions.nInnerIters = inner_niters; }
    Superpixels sp = Superpixels(nbatch, width, height, nftrs, spoptions, nspix, seg_gpu);
    sp.load_gpu_img((float*)(imgs.data<uint8_t>()));

    // -- use current image to compute params, skipping invalid --
    sp.run_update_param();
    bool* filled_gpu;
    const int sizeofbool = sizeof(bool);
    throw_on_cuda_error( cudaMalloc((void**) &filled_gpu, nbatch*nMissing*sizeofbool));

    if (nMissing>0){
        init_prop_seg(sp.image_gpu_double, seg_gpu,
                      missing_gpu, border_gpu, sp.sp_params, nPix, nMissing,
                      nbatch, width, height, nftrs,sp.J_i,sp.logdet_Sigma_i,
                      i_std,sp.sp_options.s_std,sp.sp_options.nInnerIters,
                      sp.nSPs,sp.nSPs_buffer,sp.sp_options.beta_potts_term,
                      debug_spix_gpu, debug_border_gpu, debug_fill);
    }

    torch::Tensor spix = torch::zeros({nbatch, height, width}, options_i32);

    // -- copy spix --
    cudaMemcpy(spix.data<int>(),seg_gpu,nPix*sizeof(int),cudaMemcpyDeviceToDevice);

    return spix;
}
