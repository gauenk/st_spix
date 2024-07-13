
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
#include "core/Superpixels.h"

// -- define --
#define CHECK_CUDA(x) TORCH_CHECK(x.device().is_cuda(), #x " must be a CUDA tensor")
#define CHECK_CONTIGUOUS(x) TORCH_CHECK(x.is_contiguous(), #x " must be contiguous")
#define CHECK_INPUT(x) CHECK_CUDA(x); CHECK_CONTIGUOUS(x)
#define THREADS_PER_BLOCK 512

__global__
void copy_spix_params(float* means, float* cov,
                      superpixel_params* sp_params, int* ids, int K){

    // -- filling superpixel params into image --
    int ix = threadIdx.x + blockIdx.x * blockDim.x;  
    if (ix>=K) return; 

    // -- offset memory access --
    float* cov_ix = cov + ix * 4;
    float* means_ix = means + ix * 5;

    // -- read spix --
    int sp_index = ids[ix];
    auto params_ix = sp_params[sp_index];
      
    // -- fill params --
    cov_ix[0]  = params_ix.sigma_s.x;
    cov_ix[1]  = params_ix.sigma_s.y;
    cov_ix[2]  = params_ix.sigma_s.z;
    cov_ix[3]  = params_ix.logdet_Sigma_s;
    means_ix[0] = params_ix.mu_i.x;
    means_ix[1] = params_ix.mu_i.y;
    means_ix[2] = params_ix.mu_i.z;
    means_ix[3] = params_ix.mu_s.x;
    means_ix[4] = params_ix.mu_s.y;
}


__global__
void compute_sprobs_from_sp_params(float* sprobs,
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

// __global__
// void compute_sprobs(float* sprobs,
//                     float * means, float* cov,
//                     float* lab_img, int* ids,
//                     float inv_sigma_i, float logdet_sigma_i,
//                     int npix, int K, int width){
  
//     // -- filling superpixel params into image --
//     int ix = threadIdx.x + blockIdx.x * blockDim.x;  
//     if (ix>=npix*K) return; 
//     int K_ix = ix%K;
//     int img_ix = ix/K;

//     // -- offset memory access --
//     float* img_p = lab_img + img_ix * 3; // always 3 channels right now
//     float* sprobs_p = sprobs + ix;

//     // -- convert to height,width
//     int hi = img_ix % width;
//     int wi = img_ix / width;

//     // -- read Kth valid spix --
//     // int sp_index = ids[K_ix];
//     auto means_ix = means[K_ix*5];
//     auto cov_ix = cov[K_ix*4];
      
//     // -- compute color/spatial differences --
//     // const float x0 = __ldg(&img_p[0])-__ldg(&params_ix.mu_i.x);
//     // const float x1 = __ldg(&img_p[1])-__ldg(&params_ix.mu_i.y);
//     // const float x2 = __ldg(&img_p[2])-__ldg(&params_ix.mu_i.z);
//     // const int d0 = wi - __ldg(&params_ix.mu_s.x);
//     // const int d1 = hi - __ldg(&params_ix.mu_s.y);
//     const float x0 = img_p[0]-means_ix[0];
//     const float x1 = img_p[1]-means_ix[1];
//     const float x2 = img_p[2]-means_ix[2];
//     const int d0 = wi - means_ix[3];
//     const int d1 = hi - means_ix[4];

//     // -- color component --
//     // const float sigma_s_x = __ldg(&params_ix.sigma_s.x);
//     // const float sigma_s_y = __ldg(&params_ix.sigma_s.y);
//     // const float sigma_s_z = __ldg(&params_ix.sigma_s.z);
//     // const float logdet_sigma_s = __ldg(&params_ix.logdet_Sigma_s);
//     const float sigma_s_x = cov_ix[0];
//     const float sigma_s_y = cov_ix[1];
//     const float sigma_s_z = cov_ix[2];
//     const float logdet_sigma_s = cov_ix[3];

//     // -- [color component] log 2d gaussian (isotropic cov) --
//     float res = -inv_sigma_i*(x0*x0 + x1*x1 + x2*x2) - logdet_sigma_i;

//     // -- [space component] log 2d gaussian prob --
//     res = res - d0*d0*sigma_s_x - d1*d1*sigma_s_z - 2*d0*d1*sigma_s_y - logdet_sigma_s;

//     // -- fill sprobs --
//     sprobs_p[0] = res;
// }


static superpixel_options get_sp_options(int nPixels_in_square_side,
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
dev_bass_forward_cuda(const torch::Tensor imgs,
                  torch::Tensor spix,
                  int nPixels_in_square_side){

    // -- check --
    CHECK_INPUT(imgs);
    CHECK_INPUT(spix);

    // -- unpack --
    int nbatch = imgs.size(0);
    int height = imgs.size(1);
    int width = imgs.size(2);
    int nftrs = imgs.size(3);
    int npix = height*width;
    // assert((nftrs == 3),"For now, this must be 3.")

    // fprintf(stdout,"npix: %d\n",npix);
    // std::cout << "npix " << npix << std::endl;

    // -- init superpixel --
    float i_std = 0.018;
    float beta = 0.5;
    float alpha = 0.5;
    superpixel_options spoptions = get_sp_options(nPixels_in_square_side,
                                                  i_std, beta, alpha);
    Superpixels sp = Superpixels(width, height, spoptions);

    // -- load single image --
    sp.load_gpu_img((float*)(imgs.data<uint8_t>()));

    // -- run segmentation --
    sp.calc_seg();
    // sp.convert_lab_to_rgb();

    // -- viz --
    // int* tmp_ptr = sp.get_seg_cuda();
    // int* tmp = (int*) malloc(100 * sizeof(int));
    // cudaMemcpy(tmp, tmp_ptr, 100 * sizeof(int), cudaMemcpyDeviceToHost);
    // for(int i = 0; i < 100; i++)
    //   std::cout << "H[" << i << "] = " << tmp[i] << std::endl;

    // -- launch -- 
    // AT_DISPATCH_FLOATING_TYPES(imgs.type(), "forward_kernel", ([&] {
          // }));

    /*****************************************************

                      Copy Spix

    *****************************************************/
    // -- copy spix --
    cudaMemcpy(spix.data<int>(), sp.get_seg_cuda(),
               npix * sizeof(int), cudaMemcpyDeviceToHost);


    /*****************************************************

                    Copy Covariance 

    *****************************************************/

    // -- init covariance --
    auto unique_ids = std::get<0>(at::_unique(spix));
    int K = unique_ids.sizes()[0];
    auto options =torch::TensorOptions().dtype(torch::kFloat32)
      .layout(torch::kStrided).device(imgs.device());
    torch::Tensor cov = torch::zeros({K, 4}, options);
    torch::Tensor means = torch::zeros({K, 5}, options);

    // -- dispatch info --
    int num_blocks0 = ceil( double(K) / double(THREADS_PER_BLOCK) ); 
    dim3 nthreads0(THREADS_PER_BLOCK);
    dim3 nblocks0(num_blocks0);

    // -- launch --
    copy_spix_params<<<nblocks0,nthreads0>>>(means.data<float>(),
                                             cov.data<float>(),
                                             sp.get_cuda_sp_params(),
                                             unique_ids.data<int>(),K);


    /*****************************************************

                    Compute Probs

    *****************************************************/

    // -- constants --
    float half_i_std_square = float(i_std/2) * float(i_std/2);
    float i_std_square = float(i_std) * float(i_std);
    float inv_sigma_i = 1.0/half_i_std_square;
    float logdet_sigma_i = log(half_i_std_square * half_i_std_square * half_i_std_square);

    // -- init covariance --
    torch::Tensor sprobs = torch::zeros({height,width,K}, options);

    // -- dispatch info --
    int num_blocks1 = ceil( double(npix*K) / double(THREADS_PER_BLOCK) ); 
    dim3 nthreads1(THREADS_PER_BLOCK);
    dim3 nblocks1(num_blocks1);

    // -- launch --
    compute_sprobs_from_sp_params<<<nblocks1,nthreads1>>>(\
               sprobs.data<float>(),sp.get_cuda_sp_params(),
               sp.get_image_gpu_double(),unique_ids.data<int>(),
               inv_sigma_i,logdet_sigma_i,npix,K,width);
    // compute_sprobs<<<nblocks1,nthreads1>>>(sprobs.data<float>(),
    //                                        means.data<float>(),
    //                                        cov.data<float>(),
    //                                        sp.get_image_gpu_double(),
    //                                        unique_ids.data<int>(),
    //                                        inv_sigma_i,logdet_sigma_i,npix,K,width);


    // -- return --
    return std::make_tuple(means,cov,sprobs,unique_ids);
}


void init_dev_bass(py::module &m){
  m.def("dev_bass_forward", &dev_bass_forward_cuda,
        "neighborhood superpixel attention forward");
}
