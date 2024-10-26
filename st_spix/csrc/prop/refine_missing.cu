
/*********************************************************************************

     - This finds a likely superpixel state
     after missing pixels are filled-in using "filled.cu",
     and after splitting with "split_disconnected.du".

     - This is necessary, because the "filled" superpixels are
     not in a likely state after the shift. The missing pixels have
     merely been assigned their spatial neighbor. This section of
     code actually runs BASS using the posterior of the parameter estimates.

     - This section of the code is different from "XXXX.cu"
     because updates can only effect the "missing" region.

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
#include "rgb2lab.h"
#include "init_utils.h"
#include "init_sparams.h"
#include "seg_utils.h"
#include "sparams_io.h"

// -- primary functions --
#include "refine_missing.h"
#include "update_params.h"
#include "update_seg.h"
// #include "update_prop_params.h"
// #include "update_missing_seg.h"

// -- define --
#define CHECK_CUDA(x) TORCH_CHECK(x.device().is_cuda(), #x " must be a CUDA tensor")
#define CHECK_CONTIGUOUS(x) TORCH_CHECK(x.is_contiguous(), #x " must be contiguous")
#define CHECK_INPUT(x) CHECK_CUDA(x); CHECK_CONTIGUOUS(x)
#define THREADS_PER_BLOCK 512


/**********************************************************

             -=-=-=-=- Main Function -=-=-=-=-=-

***********************************************************/

__host__ void refine_missing(float* img, int* seg, spix_params* sp_params,
                             bool* missing, bool* border, spix_helper* sp_helper,
                             int niters, int niters_seg,
                             // float3 pix_ivar,float logdet_pix_var,
                             float sigma2_app, float potts,
                             int nspix, int nbatch, int width, int height, int nftrs,
                             double* logging_aprior){

    // -- init --
    int npix = height * width;
    int nspix_buffer = nspix * 45;
    for (int i = 0; i < niters; i++) {

      // -- Update Parameters with Previous SuperpixelParams as Prior --
      update_params(img, seg, sp_params, sp_helper, sigma2_app,
                    npix, nspix_buffer, nbatch, width, nftrs);

      // -- Update Segmentation --
      update_seg(img, seg, border, sp_params, niters_seg, 
                 sigma2_app,potts,npix, nbatch, width, height, nftrs);

      // -- logging --
      cudaMemcpy(logging_aprior+i,&(sp_params[10].prior_lprob),sizeof(double),
                 cudaMemcpyDeviceToDevice);

    }

    // -- Update Parameters with Previous SuperpixelParams as Prior --
    update_params(img, seg, sp_params, sp_helper, sigma2_app,
                  npix, nspix_buffer, nbatch, width, nftrs);

    CudaFindBorderPixels_end(seg, border, npix, nbatch, width, height);
}


/**********************************************************

             -=-=-=-=- Python API  -=-=-=-=-=-

***********************************************************/

// torch::Tensor
std::tuple<torch::Tensor,PySuperpixelParams,torch::Tensor>
run_refine_missing(const torch::Tensor img,
                   const torch::Tensor spix,
                   const torch::Tensor missing,
                   const PySuperpixelParams prior_params,
                   // const torch::Tensor prior_map,
                   int nspix, int niters, int niters_seg,
                   int sp_size, float pix_var_i, float potts){

    // -- check --
    CHECK_INPUT(img);
    CHECK_INPUT(spix);
    CHECK_INPUT(missing);
    CHECK_INPUT(prior_params.mu_app);
    CHECK_INPUT(prior_params.mu_shape);
    CHECK_INPUT(prior_params.sigma_shape);
    CHECK_INPUT(prior_params.logdet_sigma_shape);
    CHECK_INPUT(prior_params.counts);
    CHECK_INPUT(prior_params.prior_counts);
    // CHECK_INPUT(prior_map);

    // -- unpack --
    int nbatch = spix.size(0);
    int height = spix.size(1);
    int width = spix.size(2);
    int nftrs = img.size(3);
    int npix = height*width;
    int nmissing = missing.sum().item<int>();
    // int init_map_size = prior_map.size(0);
    
    // -- allocate filled spix --
    auto options_i32 = torch::TensorOptions().dtype(torch::kInt32)
      .layout(torch::kStrided).device(spix.device());
    auto options_f64 = torch::TensorOptions().dtype(torch::kFloat64)
      .layout(torch::kStrided).device(spix.device());
    torch::Tensor filled_spix = spix.clone();
    assert(nbatch==1);

    // -- allocate memory --
    int nspix_buffer = nspix*50;
    const int sparam_size = sizeof(spix_params);
    const int helper_size = sizeof(spix_helper);
    bool* border = (bool*)easy_allocate(nbatch*npix,sizeof(bool));
    spix_params* sp_params = get_tensors_as_params(prior_params,sp_size,
                                                   npix,nspix,nspix_buffer);
    // spix_params* prior_sp_params = get_tensors_as_params(prior_params,sp_size,
    //                                                      npix,nspix,nspix_buffer);
    // spix_params* sp_params=(spix_params*)easy_allocate(nspix_buffer,sparam_size);
    spix_helper* sp_helper=(spix_helper*)easy_allocate(nspix_buffer,helper_size);
    // init_sp_params_s(sp_params,sp_size,nspix,nspix_buffer,npix);
    // init_prior_counts(sp_params,prior_params.prior_counts.data<int>(),
    //                   prior_map.data<int>(),init_map_size);


    // bool* border = allocate_border(nbatch*npix);
    // spix_params* sp_params = allocate_sp_params(nspix_buffer);
    // spix_helper* sp_helper = allocate_sp_helper(nspix_buffer);
    // init_sp_params(sp_params,sp_size,nspix,nspix_buffer,npix);

    // -- init sp params from past --
    // assert(rescales.size(0) == 4);// must be of size 4
    // float4 rescale;
    // rescale.x = rescales[0].item<int>();
    // rescale.y = rescales[1].item<int>();
    // rescale.z = rescales[2].item<int>();
    // rescale.w = rescales[3].item<int>();
    //init_sp_params_from_past(sp_params,prior_sp_params,rescale,nspix,nspix_buffer,npix);

    // -- compute pixel (inverse) variance info --
    float sigma_app = pix_var_i;
    // float pix_half = float(pix_var_i/2) * float(pix_var_i/2);
    // float3 pix_var;
    // pix_var.x = 1.0/pix_half;
    // pix_var.y = 1.0/pix_half;
    // pix_var.z = 1.0/pix_half;
    // float logdet_pix_var = log(pix_half * pix_half * pix_half);

    // -- convert image color --
    // auto img_lab = img_rgb.clone();
    // rgb2lab(img_rgb.data<float>(),img_lab.data<float>(),npix,nbatch);

    // -- Get pointers --
    float* img_ptr = img.data<float>();
    int* filled_spix_ptr = filled_spix.data<int>();
    bool* missing_ptr = missing.data<bool>();
    // int* prior_map_r_ptr = prior_map.data<int>();

    // -- allocate larger memory for prior map --
    // int* prior_map_ptr = (int*)easy_allocate(nspix_buffer,sizeof(int));
    // cudaMemset(prior_map_ptr, -1, nspix_buffer*sizeof(int));
    // cudaMemcpy(prior_map_ptr,prior_map_r_ptr,
    //            init_map_size*sizeof(int),cudaMemcpyDeviceToDevice);

    // -- init superpixel params --
    // float prior_sigma_app = float(pix_var_i/2) * float(pix_var_i/2);
    // init_sp_params(sp_params,sigma_app,img_ptr,filled_spix_ptr,
    //                sp_helper,npix,nspix,nspix_buffer,nbatch,width,nftrs);
    // init_sp_params_from_past(sp_params,prior_sp_params,prior_map_ptr,
    //                          rescale,nspix,nspix_buffer,npix);

    // -- init logging_lprior --
    torch::Tensor logging = torch::zeros({niters,1},options_f64);
    double* logging_ptr = logging.data<double>();

    // -- run fill --
    if (nmissing>0){
      refine_missing(img_ptr, filled_spix_ptr, sp_params,
                     missing_ptr, border, sp_helper,
                     niters, niters_seg, sigma_app, potts,
                     nspix, nbatch, width, height, nftrs, logging_ptr);
    }

    // -- get spixel parameters as tensors --
    auto unique_ids = std::get<0>(at::_unique(filled_spix));
    auto ids = unique_ids.data<int>();
    int num_ids = unique_ids.sizes()[0];
    int max_spix = prior_params.ids.size(0);
    PySuperpixelParams params = get_output_params(sp_params,prior_params,
                                                  ids,num_ids,max_spix);
    // run_update_prior(spix,params); // shift estimates to prior information @ spix


    // -- free --
    // cudaFree(prior_map_ptr);
    cudaFree(border);
    cudaFree(sp_params);
    cudaFree(sp_helper);

    return std::make_tuple(filled_spix,params,logging);
}

void init_refine_missing(py::module &m){
  m.def("refine_missing", &run_refine_missing,"refine missing labels");
}

