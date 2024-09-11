
/********************************************************************

      Run BASS using the propograted superpixel segs and params

********************************************************************/


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
#include "bass_iters.h"
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

__host__ void bass_iters(float* img, int* seg,  float* centers,
                         int* missing, bool* border,
                         int nbatch, int width, int height,
                         int nspix, int nmissing, int break_iter){

    // -- init --
    int prior_sigma_s = sp_size * sp_size;//sp_options.area * sp_options.area;
    int prior_count = sp_size; //?
    int nspix_buffer = nspix * 45 ;
    // bool cal_cov = sp_options.calc_cov;
    // float i_std = sp_options.i_std;
    // float alpha = sp_options.alpha_hasting;
    // int s_std = sp_options.s_std;
    // int nInnerIters = sp_options.nInnerIters;
    // int split_merge_start = sp_options.split_merge_start;


    int count = 1;
    int count_split =0;

    for (int i = 0; i < sp_options.nEMIters*1; i++) {

        // -- Update Parameters with Prior! --
        update_param(img, seg, sp_params, sp_gpu_helper,
                     npix, nspix, nspix_buffer, nbatch, dim_x, dim_y, nftrs,
                     prior_sigma_s, prior_count);

        // -- Run Split/Merge --
        if (i > split_merge_start){
            max_SP = run_split_merge(img, split_merge_pairs,
                                     seg, border, sp_params,
                                     sp_gpu_helper,sp_gpu_helper_sm,
                                     seg_split1,  seg_split2, seg_split3,
                                     i_std, alpha, max_SP, count%2, i,
                                     npix,nbatch,dim_x,dim_y,nftrs,nspix_buffer):
            update_param(img, seg, sp_params,
                         sp_gpu_helper, npix, nspix, nspix_buffer,
                         nbatch, dim_x, dim_y, nftrs, prior_sigma_s, prior_count);
        }

        // -- Update Segmentation ONLY within missing pix --
        update_seg(img, seg, seg_potts_label, border, sp_params,
                   J_i, logdet_Sigma_i, cal_cov, i_std, s_std, nInnerIters, npix,
                   nspix, nspix_buffer, nbatch, dim_x, dim_y, nftrs,
                   sp_options.beta_potts_term);

    }
    CudaFindBorderPixels_end(seg, border, npix, nbatch, dim_x, dim_y, 1);

}

__host__ int run_split_merge(const float* img,
             int* split_merge_pairs, int* seg, bool* border,
             superpixel_params* sp_params, superpixel_GPU_helper* sp_gpu_helper,
             superpixel_GPU_helper_sm* sp_gpu_helper_sm,
             int* seg_split1 ,int* seg_split2, int* seg_split3,
             float i_std, float alpha, int count, int i, int max_SP,
             const int npix, const int nbatch,
             const int xdim, const int ydim, const int nftrs, const int nspix_buffer){
  if(i%4 == 0){
    // -- run split --
    max_SP = CudaCalcSplitCandidate(img, split_merge_pairs,
             seg, border, sp_params ,sp_gpu_helper,sp_gpu_helper_sm,
             npix,nbatch,xdim,ydim,nftrs,nspix_buffer,seg_split1,seg_split2,
             seg_split3,max_SP, count, i_std, alpha);
  }else if( i%4 == 2){
        CudaCalcMergeCandidate(img,split_merge_pairs,seg,
               border, sp_params ,sp_gpu_helper,sp_gpu_helper_sm,
               npix,nbatch,dim_x,dim_y,nftrs,
               nspix_buffer,count%2,i_std, alpha);
  }
  return max_SP;
}


/**********************************************************

             -=-=-=-=- Python API  -=-=-=-=-=-

***********************************************************/

torch::Tensor run_bass_iters(const torch::Tensor img,
                             const torch::Tensor spix,
                             const torch::Tensor centers,
                             int nspix, int break_iter){

    // -- check --
    CHECK_INPUT(img);
    CHECK_INPUT(spix);
    CHECK_INPUT(centers);

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

    // -- run fill --
    float* centers_ptr = centers.data<float>();
    int* missing_ptr = missing.data<int>();
    if (nmissing>0){
      bass_iters(filled_spix_ptr, centers_ptr, missing_ptr, border,
                 nbatch, width, height, nspix, nmissing, break_iter);
    }
    cudaFree(border);

    return filled_spix;
}

void init_bass_iters(py::module &m){
  m.def("bass_iters", &run_bass_iters,"run propogated bass iters");
}

