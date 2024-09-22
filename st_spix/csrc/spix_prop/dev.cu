/*********************************************************

                Not supported :D

*********************************************************/


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

#include <thrust/host_vector.h>
#include <thrust/device_vector.h>
#include <thrust/copy.h>
#include <thrust/fill.h>
#include <thrust/reduce.h>

// -- local import --
#ifndef MY_PROP_SP_STRUCT
#define MY_PROP_SP_STRUCT
#include "../bass/share/refine.h"
#include "../bass/core/Superpixels.h"
#include "../bass/simple_sparams_io.h"
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

__global__
void copy_only_spatial_mean(superpixel_params* sp_params_dest,
                            superpixel_params* sp_params_src, int nspix){

    // -- filling superpixel params into image --
    int ix = threadIdx.x + blockIdx.x * blockDim.x;  
    if (ix>=nspix) return; 

    // -- read params --
    auto params_src = sp_params_src[ix];
    auto params_dest = sp_params_dest[ix];
      
    // -- fill params --
    params_dest.mu_s = params_src.mu_s;

}

std::tuple<torch::Tensor,torch::Tensor,torch::Tensor,torch::Tensor,torch::Tensor,
             torch::Tensor,torch::Tensor,torch::Tensor,torch::Tensor,torch::Tensor>
spix_prop_dev_cuda(const torch::Tensor imgs,
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
    CHECK_INPUT(in_means);
    CHECK_INPUT(in_counts);
    CHECK_INPUT(in_cov);
    // return std::make_tuple(in_spix,in_means,in_cov);

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
    // fprintf(stdout,"0\n");
    // cudaDeviceSynchronize();
    Superpixels sp = Superpixels(nbatch, width, height, nftrs, spoptions, nspix, seg_gpu);
    // fprintf(stdout,"1\n");
    // cudaDeviceSynchronize();

    // -- load single image --
    sp.load_gpu_img((float*)(imgs.data<uint8_t>()));
    // fprintf(stdout,"2\n");
    // cudaDeviceSynchronize();

    // -- load previous results --
    // sp.init_from_previous(in_spix.data<int>(),
    //                       in_means.data<float>(),
    //                       in_cov.data<float>(),
    //                       in_counts.data<int>(),
    //                       nspix,max_SP);
      


    // -- use current image to compute params, skipping invalid --
    sp.run_update_param();

    // fprintf(stdout,"3\n");
    // cudaDeviceSynchronize();

    // -- launch prop spix --
    // float* image_gpu_double = imgs.data<float>();
    // float* image_gpu_double = sp.image_gpu_double;
    int* missing_gpu = in_missing.data<int>();
    torch::Tensor boarder = torch::zeros({nbatch, height, width}, options_b);
    bool* border_gpu = boarder.data<bool>();
    // cudaMemset(border_gpu, 0, nbatch*nPix*sizeofint);


    bool* filled_gpu;
    const int sizeofbool = sizeof(bool);
    throw_on_cuda_error( cudaMalloc((void**) &filled_gpu, nbatch*nMissing*sizeofbool));
    // cudaMemset(filled_gpu, 0, nbatch*nMissing*sizeofbool);


    int* debug_spix_gpu = nullptr;
    bool* debug_border_gpu = nullptr;
    float* debug_seg_gpu = nullptr;
    int niters_total = niters * inner_niters;
    int total_refine = niters_refine * inner_niters;
    int debug_size = niters_total*nPix;
    int debug_size_seg = total_refine*nPix;
    if (debug_fill){

      throw_on_cuda_error(cudaMalloc((void**)&debug_spix_gpu,debug_size*sizeof(int)));
      throw_on_cuda_error(cudaMalloc((void**)&debug_border_gpu,debug_size*sizeof(bool)));
      throw_on_cuda_error(cudaMalloc((void**)&debug_seg_gpu,
                                     45*debug_size_seg*sizeof(float)));
    }


    // const int nPixels, const int nMissing,
    //   int nbatch, int xdim, int ydim, int nftrs,
    //   const float3 J_i, const float logdet_Sigma_i, 
    //   float i_std, int s_std, int nInnerIters,
    //   const int nSPs, int nSPs_buffer,
    //   float beta_potts_term){

    // -- load sp params --

    // superpixel_params* sp_params = nullptr;
    // float i_std = sp_options.i_std;
    // float alpha = sp_options.alpha_hasting;
    // int s_std = sp_options.s_std;
    // int nInnerIters = sp_options.nInnerIters;
    // fprintf(stdout,"4\n");
    // cudaDeviceSynchronize();
    // fprintf(stdout,"nSPs: %d\n",sp.nSPs);

    if (nMissing>0){
      if (prop_type == false){
        init_prop_seg(sp.image_gpu_double, seg_gpu,
                      missing_gpu, border_gpu, sp.sp_params, nPix, nMissing,
                      nbatch, width, height, nftrs,sp.J_i,sp.logdet_Sigma_i,
                      i_std,sp.sp_options.s_std,sp.sp_options.nInnerIters,
                      sp.nSPs,sp.nSPs_buffer,sp.sp_options.beta_potts_term,
                      debug_spix_gpu, debug_border_gpu, debug_fill);
      }else{
        init_prop_seg(sp.image_gpu_double, seg_gpu,
                      missing_gpu, border_gpu, sp.sp_params, nPix, nMissing,
                      nbatch, width, height, nftrs,sp.J_i,sp.logdet_Sigma_i,
                      i_std,sp.sp_options.s_std,sp.sp_options.nInnerIters,
                      sp.nSPs,sp.nSPs_buffer,sp.sp_options.beta_potts_term,
                      debug_spix_gpu, debug_border_gpu, debug_fill);
        // init_prop_seg_space(sp.image_gpu_double, seg_gpu,
        //               missing_gpu, border_gpu, sp.sp_params, nPix, nMissing,
        //               nbatch, width, height, nftrs,sp.J_i,sp.logdet_Sigma_i,
        //               i_std,sp.sp_options.s_std,sp.sp_options.nInnerIters,
        //               sp.nSPs,sp.nSPs_buffer,sp.sp_options.beta_potts_term,
        //               debug_spix_gpu, debug_border_gpu, debug_fill);
      }
    }

    // gpuErrchk( cudaPeekAtLastError() );
    // gpuErrchk( cudaDeviceSynchronize() );
    // cudaMemcpy(sp.seg_cpu, seg_gpu, nPix*sizeof(int), cudaMemcpyDeviceToHost);
    // gpuErrchk( cudaPeekAtLastError() );
    // gpuErrchk( cudaDeviceSynchronize() );

    // cudaMemcpy(sp.seg_gpu, seg_gpu, nPix*sizeof(int), cudaMemcpyDeviceToDevice);

    // torch::Tensor children = run_split_disconnected(seg_gpu, nbatch,
    //                                                 height, width, nspix);
    // torch::Tensor children = run_split_disconnected(sp.image_gpu_double, seg_gpu,
    //                        missing_gpu, border_gpu, sp.sp_params, nPix, nMissing,
    //                        nbatch, width, height, nftrs,sp.J_i,sp.logdet_Sigma_i,
    //                        i_std,sp.sp_options.s_std,sp.sp_options.nInnerIters,
    //                        sp.nSPs,sp.nSPs_buffer,sp.sp_options.beta_potts_term,
    //                        debug_spix_gpu, debug_border_gpu, debug_fill);

    // gpuErrchk( cudaPeekAtLastError() );
    // gpuErrchk( cudaDeviceSynchronize() );

    // fprintf(stdout,"5\n");
    // cudaError_t err =cudaDeviceSynchronize();
    // cudaDeviceSynchronize();


    // -- run BASS for a few iterations to allow for merge/split --
    // fprintf(stdout,"5\n");
    if (niters_refine>0){

      // -- re-format previous superpixel parameters --
      const int sofsparams = sizeof(superpixel_params);
      superpixel_params* sp_params_prev;
      throw_on_cuda_error(cudaMalloc((void**)&sp_params_prev,sp.nSPs_buffer*sofsparams));

      // -- copy from input data --
      torch::Tensor ids = torch::arange(nspix).to(torch::kInt32);
      int num_blocks_i = ceil( double(nspix) / double(THREADS_PER_BLOCK) ); 
      dim3 nthreads_i(THREADS_PER_BLOCK);
      dim3 nblocks_i(num_blocks_i);
      copy_params_to_spix<<<nblocks_i,nthreads_i>>>(in_means.data<float>(),
                                                    in_cov.data<float>(),
                                                    in_counts.data<int>(),
                                                    sp_params_prev,
                                                    ids.data<int>(), nspix);
      // copy_only_spatial_mean<<<nblocks_i,nthreads_i>>>(sp_params_prev,
      //                                                  sp.sp_params,nspix);

      // -- run modified bass iters using sp_params_prev prior --
      sp.sp_options.nEMIters = niters_refine; // set iters 
      calc_prop_seg(sp.image_gpu_double, seg_gpu,
                    missing_gpu, sp.seg_potts_label,border_gpu,
                    sp.sp_params, sp_params_prev,
                    // sp_params_prev, sp_params_prev,
                    sp.sp_gpu_helper,sp.J_i,sp.logdet_Sigma_i,sp.sp_options,
                    nbatch, nftrs, width, height, nspix, use_transition,
                    debug_seg_gpu);

      // -- free previous superpixels --
      cudaFree(sp_params_prev);
    }

    /*****************************************************
                      Copy Spix
    *****************************************************/

    // -- init spix --
    torch::Tensor spix = torch::zeros({nbatch, height, width}, options_i32);

    // -- copy spix --
    cudaMemcpy(spix.data<int>(),seg_gpu,nPix*sizeof(int),cudaMemcpyDeviceToDevice);


    // // -- dispatch info --
    // int num_blocks1 = ceil( double(npix) / double(THREADS_PER_BLOCK) ); 
    // dim3 nthreads1(THREADS_PER_BLOCK);
    // dim3 nblocks1(num_blocks1);

    // // -- relabel spix --
    // relabel_spix<false><<<nblocks1,nthreads1>>>(spix.data<int>(),
    //                                             unique_ids.data<int>(),
    //                                             npix, nspix);
    gpuErrchk( cudaPeekAtLastError() );
    gpuErrchk( cudaDeviceSynchronize() );


    /*****************************************************
                    Copy Parameters
    *****************************************************/


    // -- init covariance --
    auto unique_ids = std::get<0>(at::_unique(spix));
    int nspix_r = unique_ids.sizes()[0];
    int label_min = torch::min(unique_ids).item<int>();
    int label_max = torch::max(unique_ids).item<int>();
    // assert(label_min>=0); // should only be negative when vizualizing

    // fprintf(stdout,"unique [min,max]: %d,%d\n",label_min,label_max);
    // fprintf(stdout,"nspix,nspix_r: %d,%d\n",nspix,nspix_r);
    assert(nspix_r <= nspix);//"Must be equal; no superpixels added/removed yet."
    torch::Tensor means = torch::zeros({nbatch, nspix, 5}, options_f32);
    torch::Tensor cov = torch::zeros({nbatch, nspix, 4}, options_f32);
    torch::Tensor counts = torch::zeros({nbatch, nspix}, options_i32);
    torch::Tensor spix_parents = torch::zeros({nbatch, nspix}, options_i32);

    gpuErrchk( cudaPeekAtLastError() );
    gpuErrchk( cudaDeviceSynchronize() );


    // -- dispatch info --
    int num_blocks0 = ceil( double(nspix_r) / double(THREADS_PER_BLOCK) ); 
    dim3 nthreads0(THREADS_PER_BLOCK);
    dim3 nblocks0(num_blocks0);

    // -- launch --
    // copy_spix_to_params<<<nblocks0,nthreads0>>>(means.data<float>(),
    //                                             cov.data<float>(),
    //                                             counts.data<int>(),
    //                                             sp.sp_params,
    //                                             unique_ids.data<int>(),nspix_r);
    copy_spix_to_params_parents<<<nblocks0,nthreads0>>>(means.data<float>(),
                                                        cov.data<float>(),
                                                        counts.data<int>(),
                                                        spix_parents.data<int>(),
                                                        sp.sp_params,
                                                        unique_ids.data<int>(),nspix_r);

    gpuErrchk( cudaPeekAtLastError() );
    gpuErrchk( cudaDeviceSynchronize() );

    /*****************************************************
                    Copy Debug
    *****************************************************/

    // -- copy from pointer to pytorch tensor --

    // int debug_shape[3];
    // torch::Tensor debug_spix = torch::zeros(debug_shape, options_i32);
    // torch::Tensor debug_border = torch::zeros(debug_shape,options_bool);
    torch::Tensor debug_spix;
    torch::Tensor debug_border;
    torch::Tensor debug_seg;
    if (debug_fill){
      debug_spix = torch::zeros({niters_total, height, width}, options_i32);
      debug_border = torch::zeros({niters_total, height, width},options_bool);
      debug_seg = torch::zeros({total_refine, height, width, 45},options_f32);
    }else{
      // debug_seg.data<float>());
      debug_spix = torch::zeros({1, 1, 1}, options_i32);
      debug_border = torch::zeros({1, 1, 1},options_bool);
      debug_seg = torch::zeros({1, 1, 1, 1},options_f32);
    }

    if (debug_fill){
      int* debug_spix_th_ptr = debug_spix.data_ptr<int>();
      bool* debug_border_th_ptr = debug_border.data_ptr<bool>();
      float* debug_seg_th_ptr = debug_seg.data_ptr<float>();
      cudaMemcpy(debug_spix_th_ptr,debug_spix_gpu,
                 debug_size*sizeof(int),cudaMemcpyDeviceToDevice);
      cudaMemcpy(debug_border_th_ptr,debug_border_gpu,
                 debug_size*sizeof(bool),cudaMemcpyDeviceToDevice);
      cudaMemcpy(debug_seg_th_ptr,debug_seg_gpu,
                 45*debug_size_seg*sizeof(float),cudaMemcpyDeviceToDevice);
    }



    return std::make_tuple(boarder,spix,spix_parents,debug_spix,
                           debug_border,debug_seg,means,cov,counts,unique_ids);
}


void init_spix_prop_dev(py::module &m){
  m.def("spix_prop_dev", &spix_prop_dev_cuda,
        "neighborhood superpixel attention forward");
}
