
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
#include "share/refine.h"
#include "core/Superpixels.h"
#include "../prop/simple_sparams_io.h"
#include "../prop/simple_init_sparams.h"
#include "../prop/init_utils.h"
#include "relabel.h"

// -- define --
#define CHECK_CUDA(x) TORCH_CHECK(x.device().is_cuda(), #x " must be a CUDA tensor")
#define CHECK_CONTIGUOUS(x) TORCH_CHECK(x.is_contiguous(), #x " must be contiguous")
#define CHECK_INPUT(x) CHECK_CUDA(x); CHECK_CONTIGUOUS(x)
#define THREADS_PER_BLOCK 512


// __global__
// void compute_sprobs_from_sp_params(float* sprobs,
//                                    superpixel_params* sp_params,
//                                    float* lab_img, int* ids,
//                                    float inv_sigma_i, float logdet_sigma_i,
//                                    int npix, int nspix, int width){
  
//     // -- filling superpixel params into image --
//     int ix = threadIdx.x + blockIdx.x * blockDim.x;  
//     if (ix>=npix*nspix) return; 
//     int K_ix = ix%nspix;
//     int img_ix = ix/nspix;

//     // -- offset memory access --
//     float* img_p = lab_img + img_ix * 3; // always 3 channels right now
//     float* sprobs_p = sprobs + ix;

//     // -- convert to height,width
//     int hi = img_ix % width;
//     int wi = img_ix / width;

//     // -- read Kth valid spix --
//     int sp_index = ids[K_ix];
//     auto params_ix = sp_params[sp_index];
      
//     // -- compute color/spatial differences --
//     // const float x0 = __ldg(&img_p[0])-__ldg(&params_ix.mu_i.x);
//     // const float x1 = __ldg(&img_p[1])-__ldg(&params_ix.mu_i.y);
//     // const float x2 = __ldg(&img_p[2])-__ldg(&params_ix.mu_i.z);
//     // const int d0 = wi - __ldg(&params_ix.mu_s.x);
//     // const int d1 = hi - __ldg(&params_ix.mu_s.y);
//     const float x0 = img_p[0]-params_ix.mu_i.x;
//     const float x1 = img_p[1]-params_ix.mu_i.y;
//     const float x2 = img_p[2]-params_ix.mu_i.z;
//     const int d0 = wi - params_ix.mu_s.x;
//     const int d1 = hi - params_ix.mu_s.y;

//     // -- color component --
//     // const float sigma_s_x = __ldg(&params_ix.sigma_s.x);
//     // const float sigma_s_y = __ldg(&params_ix.sigma_s.y);
//     // const float sigma_s_z = __ldg(&params_ix.sigma_s.z);
//     // const float logdet_sigma_s = __ldg(&params_ix.logdet_Sigma_s);
//     const float sigma_s_x = params_ix.sigma_s.x;
//     const float sigma_s_y = params_ix.sigma_s.y;
//     const float sigma_s_z = params_ix.sigma_s.z;
//     const float logdet_sigma_s = params_ix.logdet_Sigma_s;


//     // -- [color component] log 2d gaussian (isotropic cov) --
//     float res = -inv_sigma_i*(x0*x0 + x1*x1 + x2*x2) - logdet_sigma_i;

//     // -- [space component] log 2d gaussian prob --
//     // res = res - d0*d0*sigma_s_x - d1*d1*sigma_s_z - 2*d0*d1*sigma_s_y - logdet_sigma_s;
//     res = res - d0*d0*sigma_s_x - d1*d1*sigma_s_z + 2*d0*d1*sigma_s_y - logdet_sigma_s;

//     // -- fill sprobs --
//     sprobs_p[0] = res;
// }


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


// std::tuple<torch::Tensor,torch::Tensor,torch::Tensor,torch::Tensor,torch::Tensor>
std::tuple<torch::Tensor,PySuperpixelParams>
bass_forward_cuda(const torch::Tensor imgs,
                  int nPixels_in_square_side,
                  float pix_cov, float alpha, float beta){
    // -- init timer --
    // float elapsed = 0;
    // cudaEvent_t start;
    // cudaEvent_t stop;
    // cudaEventCreate(&start);
    // cudaEventCreate(&stop);
    // cudaEventRecord(start);

    // -- check --
    CHECK_INPUT(imgs);

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
    // float pix_cov = 0.018;
    // float beta = 0.5;
    // float alpha = 0.5;
    superpixel_options spoptions = get_sp_options(nPixels_in_square_side,
                                                  pix_cov, beta, alpha);
    Superpixels sp = Superpixels(nbatch, width, height, nftrs, spoptions);

    // -- load single image --
    sp.load_gpu_img((float*)(imgs.data<uint8_t>()));

    // -- run segmentation --
    sp.calc_seg();

    /*****************************************************

                      Copy Spix

    *****************************************************/

    // -- init spix --
    torch::Tensor spix = torch::zeros({nbatch, height, width}, options_i32);

    // -- copy spix --
    cudaMemcpy(spix.data<int>(), sp.get_seg_cuda(),
               npix * sizeof(int), cudaMemcpyDeviceToDevice);

    // cudaEventRecord(stop);
    // cudaEventSynchronize(stop);
    // elapsed = 0;
    // cudaEventElapsedTime(&elapsed, start, stop);
    // fprintf(stdout,"[0] elapsed: %e\n",elapsed);


    /*****************************************************

                    Copy Covariance 

    *****************************************************/

    // -- unique with elapsed time --

    // cudaEventRecord(start);

    // auto unique_ids = std::get<0>(at::_unique(spix));

    // cudaEventRecord(stop);
    // cudaEventSynchronize(stop);
    // elapsed = 0;
    // cudaEventElapsedTime(&elapsed, start, stop);
    // fprintf(stdout,"[uniq] elapsed: %e\n",elapsed);

    // cudaEventRecord(start);
    // cudaEventSynchronize(start);

    // -- init covariance --
    auto unique_ids = std::get<0>(at::_unique(spix));
    int nspix = unique_ids.sizes()[0];
    // torch::Tensor means = torch::zeros({nbatch, nspix, 5}, options_f32);
    // torch::Tensor cov = torch::zeros({nbatch, nspix, 4}, options_f32);
    // torch::Tensor counts = torch::zeros({nbatch, nspix}, options_i32);

    // -- dispatch info --
    // int num_blocks0 = ceil( double(nspix) / double(THREADS_PER_BLOCK) ); 
    // dim3 nthreads0(THREADS_PER_BLOCK);
    // dim3 nblocks0(num_blocks0);

    // -- launch --
    // copy_spix_to_params<<<nblocks0,nthreads0>>>(means.data<float>(),
    //                                             cov.data<float>(),
    //                                             counts.data<int>(),
    //                                             sp.get_cuda_sp_params(),
    //                                             unique_ids.data<int>(),nspix);
    // fprintf(stdout,"a.\n");
    PySuperpixelParams params = get_params_as_tensors_s(sp.get_cuda_sp_params(),
                                                        unique_ids.data<int>(),nspix);
    // fprintf(stdout,"b.\n");

    
    // -- relabel spix --
    int num_blocks1 = ceil( double(npix) / double(THREADS_PER_BLOCK) ); 
    dim3 nthreads1(THREADS_PER_BLOCK);
    dim3 nblocks1(num_blocks1);
    relabel_spix<false><<<nblocks1,nthreads1>>>(spix.data<int>(),
                                                unique_ids.data<int>(),
                                                npix, nspix);

    // -- destroy --
    // cudaEventRecord(stop);
    // cudaEventSynchronize(stop);
    // elapsed = 0;
    // cudaEventElapsedTime(&elapsed, start, stop);
    // fprintf(stdout,"[2] elapsed: %e\n",elapsed);

    // cudaEventDestroy(start);
    // cudaEventDestroy(stop);

    // -- return --
    // return std::make_tuple(spix,means,cov,counts,unique_ids);
    return std::make_tuple(spix,params);
}


// std::tuple<torch::Tensor,torch::Tensor,torch::Tensor,torch::Tensor>
// bass_forward_refine_cuda(const torch::Tensor imgs,
//                          const torch::Tensor in_spix,
//                          const torch::Tensor in_means,
//                          const torch::Tensor in_cov,
//                          const torch::Tensor in_counts,
//                          int nPixels_in_square_side, float pix_cov,
//                          float alpha, float beta, int niters,
//                          int nspix, int max_SP){

//     // -- check --
//     CHECK_INPUT(imgs);
//     CHECK_INPUT(in_spix);
//     CHECK_INPUT(in_means);
//     CHECK_INPUT(in_counts);
//     CHECK_INPUT(in_cov);
//     // return std::make_tuple(in_spix,in_means,in_cov);

//     // -- unpack --
//     int nbatch = imgs.size(0);
//     int height = imgs.size(1);
//     int width = imgs.size(2);
//     int nftrs = imgs.size(3);
//     int npix = height*width;
//     auto options_i32 =torch::TensorOptions().dtype(torch::kInt32)
//       .layout(torch::kStrided).device(imgs.device());
//     auto options_f32 =torch::TensorOptions().dtype(torch::kFloat32)
//       .layout(torch::kStrided).device(imgs.device());

//     // -- init superpixel --
//     // float pix_cov = 0.018;
//     // float beta = 0.5;
//     // float alpha = 0.5;
//     superpixel_options spoptions = get_sp_options(nPixels_in_square_side,
//                                                   pix_cov, beta, alpha);
//     // std::cout << "niter: " << niters << std::endl;
//     if (niters >= 0){
//       spoptions.nEMIters = niters;
//     }
//     Superpixels sp = Superpixels(nbatch, width, height, nftrs, spoptions, nspix);

//     // -- load single image --
//     sp.load_gpu_img((float*)(imgs.data<uint8_t>()));

//     // -- load previous results --
//     // std::cout << "----" << std::endl;
//     // std::cout << "K: " << nspix << std::endl;    
//     // std::cout << "max_SP: " << max_SP << std::endl;
//     sp.init_from_previous(in_spix.data<int>(),
//                           in_means.data<float>(),
//                           in_cov.data<float>(),
//                           in_counts.data<int>(),
//                           nspix,max_SP);

//     // std::cout << "----" << std::endl;

//     // -- run segmentation --
//     sp.calc_seg();

//     /*****************************************************

//                       Copy Spix

//     *****************************************************/

//     // -- init spix --
//     torch::Tensor spix = torch::zeros({nbatch, height, width}, options_i32);

//     // -- copy spix --
//     cudaMemcpy(spix.data<int>(), sp.get_seg_cuda(),
//                npix * sizeof(int), cudaMemcpyDeviceToDevice);


//     /*****************************************************

//                     Copy Covariance 

//     *****************************************************/

//     // -- elapsed --
//     // float elapsed;
//     // cudaEvent_t start;
//     // cudaEvent_t stop;
//     // cudaEventCreate(&start);
//     // cudaEventCreate(&stop);
//     // cudaEventSynchronize(stop);
//     // cudaEventElapsedTime(&elapsed, start, stop);

//     // -- init covariance --
//     // cudaEventSynchronize(stop);
//     // cudaEventRecord(start, 0);
//     // auto unique_ids = std::get<0>(at::_unique(spix));
//     // cudaEventRecord(stop, 0);
//     // cudaEventElapsedTime(&elapsed, start, stop);
//     // cudaEventDestroy(start);
//     // cudaEventDestroy(stop);
//     // fprintf(stdout,"elapsed: %e\n",elapsed);

//     auto unique_ids = std::get<0>(at::_unique(spix));
//     int nspix_r = unique_ids.sizes()[0];
//     // std::cout << "----" << std::endl;
//     // std::cout << "K2: " << K << std::endl;    
//     torch::Tensor means = torch::zeros({nbatch, nspix_r, 5}, options_f32);
//     torch::Tensor cov = torch::zeros({nbatch, nspix_r, 4}, options_f32);
//     torch::Tensor counts = torch::zeros({nbatch, nspix_r}, options_i32);

//     // -- dispatch info --
//     int num_blocks0 = ceil( double(nspix_r) / double(THREADS_PER_BLOCK) ); 
//     dim3 nthreads0(THREADS_PER_BLOCK);
//     dim3 nblocks0(num_blocks0);

//     // -- launch --
//     // copy_spix_to_params_icov2cov<<<nblocks0,nthreads0>>>(means.data<float>(),
//     //                                             cov.data<float>(),
//     //                                             counts.data<int>(),
//     //                                             sp.get_cuda_sp_params(),
//     //                                             unique_ids.data<int>(),nspix_r);
//     // copy_spix_to_params<<<nblocks0,nthreads0>>>(means.data<float>(),
//     //                                             cov.data<float>(),
//     //                                             counts.data<int>(),
//     //                                             sp.get_cuda_sp_params(),
//     //                                             unique_ids.data<int>(),nspix_r);

//     // -- dispatch info --
//     int num_blocks1 = ceil( double(npix) / double(THREADS_PER_BLOCK) ); 
//     dim3 nthreads1(THREADS_PER_BLOCK);
//     dim3 nblocks1(num_blocks1);

//     // -- relabel spix --
//     relabel_spix<false><<<nblocks1,nthreads1>>>(spix.data<int>(),
//                                                 unique_ids.data<int>(),
//                                                 npix, nspix_r);

//     // -- return --
//     return std::make_tuple(spix,means,cov,counts);
// }


void init_bass(py::module &m){
  m.def("bass_forward", &bass_forward_cuda,
        "neighborhood superpixel attention forward");
  // m.def("bass_forward_refine", &bass_forward_refine_cuda,
  //       "neighborhood superpixel attention forward");
}
