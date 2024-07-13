
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

// -- local import --
#include "../bass/core/Superpixels.h"

// -- define --
#define CHECK_CUDA(x) TORCH_CHECK(x.device().is_cuda(), #x " must be a CUDA tensor")
#define CHECK_CONTIGUOUS(x) TORCH_CHECK(x.is_contiguous(), #x " must be contiguous")
#define CHECK_INPUT(x) CHECK_CUDA(x); CHECK_CONTIGUOUS(x)
#define THREADS_PER_BLOCK 512

// __global__
// void copy_spix_params(float* means, float* cov, int* counts,
//                       superpixel_params* sp_params, int* ids, int K){

//     // -- filling superpixel params into image --
//     int ix = threadIdx.x + blockIdx.x * blockDim.x;  
//     if (ix>=K) return; 

//     // -- offset memory access --
//     float* cov_ix = cov + ix * 4;
//     float* means_ix = means + ix * 5;
//     int* counts_ix = counts + ix;

//     // -- read spix --
//     int sp_index = ids[ix];
//     auto params_ix = sp_params[sp_index];
      
//     // -- fill params --
//     cov_ix[0]  = params_ix.sigma_s.x;
//     cov_ix[1]  = params_ix.sigma_s.y;
//     cov_ix[2]  = params_ix.sigma_s.z;
//     cov_ix[3]  = params_ix.logdet_Sigma_s;
//     means_ix[0] = params_ix.mu_i.x;
//     means_ix[1] = params_ix.mu_i.y;
//     means_ix[2] = params_ix.mu_i.z;
//     means_ix[3] = params_ix.mu_s.x;
//     means_ix[4] = params_ix.mu_s.y;
//     counts_ix[0] = params_ix.count;
// }


__global__
void dev_kernel(float* sprobs,
                superpixel_params* sp_params,
                float* lab_img, int* ids,
                float inv_sigma_i, float logdet_sigma_i,
                int npix, int K, int width){
  
    // -- filling superpixel params into image --
    int ix = threadIdx.x + blockIdx.x * blockDim.x;  
    if (ix>=npix*K) return; 
    int K_ix = ix%K;
    int img_ix = ix/K;

    // -- offset memory access --
    float* img_p = lab_img + img_ix * 3; // always 3 channels right now
    float* sprobs_p = sprobs + ix;

    // -- convert to height,width
    int hi = img_ix % width;
    int wi = img_ix / width;

    // -- read Kth valid spix --
    int sp_index = ids[K_ix];
    auto params_ix = sp_params[sp_index];
      
    // -- compute color/spatial differences --
    // const float x0 = __ldg(&img_p[0])-__ldg(&params_ix.mu_i.x);
    // const float x1 = __ldg(&img_p[1])-__ldg(&params_ix.mu_i.y);
    // const float x2 = __ldg(&img_p[2])-__ldg(&params_ix.mu_i.z);
    // const int d0 = wi - __ldg(&params_ix.mu_s.x);
    // const int d1 = hi - __ldg(&params_ix.mu_s.y);
    const float x0 = img_p[0]-params_ix.mu_i.x;
    const float x1 = img_p[1]-params_ix.mu_i.y;
    const float x2 = img_p[2]-params_ix.mu_i.z;
    const int d0 = wi - params_ix.mu_s.x;
    const int d1 = hi - params_ix.mu_s.y;

    // -- color component --
    // const float sigma_s_x = __ldg(&params_ix.sigma_s.x);
    // const float sigma_s_y = __ldg(&params_ix.sigma_s.y);
    // const float sigma_s_z = __ldg(&params_ix.sigma_s.z);
    // const float logdet_sigma_s = __ldg(&params_ix.logdet_Sigma_s);
    const float sigma_s_x = params_ix.sigma_s.x;
    const float sigma_s_y = params_ix.sigma_s.y;
    const float sigma_s_z = params_ix.sigma_s.z;
    const float logdet_sigma_s = params_ix.logdet_Sigma_s;


    // -- [color component] log 2d gaussian (isotropic cov) --
    float res = -inv_sigma_i*(x0*x0 + x1*x1 + x2*x2) - logdet_sigma_i;

    // -- [space component] log 2d gaussian prob --
    res = res - d0*d0*sigma_s_x - d1*d1*sigma_s_z - 2*d0*d1*sigma_s_y - logdet_sigma_s;

    // -- fill sprobs --
    sprobs_p[0] = res;
}

static superpixel_options dev_get_sp_options(int nPixels_in_square_side,
                                         float i_std,float beta,
                                         float alpha_hasting){
    superpixel_options opt;
    opt.nPixels_in_square_side = nPixels_in_square_side;
    opt.i_std = i_std;
    opt.beta_potts_term = beta;
    opt.area = opt.nPixels_in_square_side*opt.nPixels_in_square_side;
    opt.s_std = opt.nPixels_in_square_side;
    opt.prior_count = opt.area*opt.area ;
    opt.calc_cov = true;
    opt.use_hex = false;
    opt.alpha_hasting = alpha_hasting;
    opt.nEMIters = opt.nPixels_in_square_side;
    //opt.nEMIters = 15;
    opt.nInnerIters = 4;
    return opt;
}

std::tuple<torch::Tensor,torch::Tensor,torch::Tensor,torch::Tensor>
spix_prop_dev_cuda_cuda(const torch::Tensor imgs,
                         const torch::Tensor in_spix,
                         const torch::Tensor in_means,
                         const torch::Tensor in_cov,
                         const torch::Tensor in_counts,
                         int nPixels_in_square_side, float i_std,
                         float alpha, float beta, int niters,
                         int in_K, int max_SP){

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
    int npix = height*width;
    auto options_i32 =torch::TensorOptions().dtype(torch::kInt32)
      .layout(torch::kStrided).device(imgs.device());
    auto options_f32 =torch::TensorOptions().dtype(torch::kFloat32)
      .layout(torch::kStrided).device(imgs.device());

    // -- init superpixel --
    // float i_std = 0.018;
    // float beta = 0.5;
    // float alpha = 0.5;
    superpixel_options spoptions = dev_get_sp_options(nPixels_in_square_side,
                                                  i_std, beta, alpha);
    // std::cout << "niter: " << niters << std::endl;
    if (niters >= 0){
      spoptions.nEMIters = niters;
    }
    Superpixels sp = Superpixels(nbatch, width, height, nftrs, spoptions, in_K);

    // -- load single image --
    sp.load_gpu_img((float*)(imgs.data<uint8_t>()));

    // -- load previous results --
    // std::cout << "----" << std::endl;
    // std::cout << "K: " << in_K << std::endl;    
    // std::cout << "max_SP: " << max_SP << std::endl;
    sp.init_from_previous(in_spix.data<int>(),
                          in_means.data<float>(),
                          in_cov.data<float>(),
                          in_counts.data<int>(),
                          in_K,max_SP);

    // std::cout << "----" << std::endl;

    // -- run segmentation --
    sp.calc_seg();

    /*****************************************************

                      Copy Spix

    *****************************************************/

    // -- init spix --
    torch::Tensor spix = torch::zeros({nbatch, height, width}, options_i32);

    // -- copy spix --
    cudaMemcpy(spix.data<int>(), sp.get_seg_cuda(),
               npix * sizeof(int), cudaMemcpyDeviceToHost);


    /*****************************************************

                    Copy Covariance 

    *****************************************************/

    // -- init covariance --
    auto unique_ids = std::get<0>(at::_unique(spix));
    int K = unique_ids.sizes()[0];
    // std::cout << "----" << std::endl;
    // std::cout << "K2: " << K << std::endl;    
    torch::Tensor means = torch::zeros({nbatch, K, 5}, options_f32);
    torch::Tensor cov = torch::zeros({nbatch, K, 4}, options_f32);
    torch::Tensor counts = torch::zeros({nbatch, K}, options_i32);

    // -- dispatch info --
    int num_blocks0 = ceil( double(K) / double(THREADS_PER_BLOCK) ); 
    dim3 nthreads0(THREADS_PER_BLOCK);
    dim3 nblocks0(num_blocks0);

    // -- launch --
    // copy_spix_params<<<nblocks0,nthreads0>>>(means.data<float>(),
    //                                          cov.data<float>(),
    //                                          counts.data<int>(),
    //                                          sp.get_cuda_sp_params(),
    //                                          unique_ids.data<int>(),K);

    // -- return --
    return std::make_tuple(spix,means,cov,counts);
}


void init_spix_prop_dev(py::module &m){
  m.def("spix_prop_dev", &spix_prop_dev_cuda,
        "neighborhood superpixel attention forward");
}
