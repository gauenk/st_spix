
// -- pytorch api --
#include <torch/extension.h>
#include <torch/types.h>
#define CHECK_CUDA(x) TORCH_CHECK(x.device().is_cuda(), #x " must be a CUDA tensor")
#define CHECK_CONTIGUOUS(x) TORCH_CHECK(x.is_contiguous(), #x " must be contiguous")
#define CHECK_INPUT(x) CHECK_CUDA(x); CHECK_CONTIGUOUS(x)
#define THREADS_PER_BLOCK 512

// -- basic --
#include <iostream>
#include <fstream>
#include <dirent.h>
#include <unistd.h> // For access()


// -- thrust --
#include <thrust/device_vector.h>
#include <thrust/host_vector.h>
#include <thrust/extrema.h>
#include <thrust/unique.h>
#include <thrust/sort.h>

// // -- opencv --
// #include <opencv2/opencv.hpp>

// -- local --
// #include "file_io.h"
#include "structs.h"
#include "init_utils.h"
#include "rgb2lab.h"
#include "bass.h"
#include "prop.h"
#include "shift_and_fill.h"
#include "seg_utils.h" // dev only
// #include "update_seg.h" // dev only
#include "split_disconnected.h"

// -- demo --
// #include "demo_utils.h"

// using namespace cv;
using namespace std;


torch::Tensor main_loop(torch::Tensor vid, torch::Tensor flows,
                        int niters, int sp_size, float sigma2_app,
                        float potts, float alpha, float split_alpha,
                        int target_nspix, bool video_mode, bool rgb2lab_b){

  // -- viz inputs --
  // printf("niters: %d, sp_size: %d, sigma2_app: %.3f, potts: %.3f, alpha: %.3f, split_alpha: %.3f, target_nspix: %d, video_mode: %s\n",niters, sp_size, sigma2_app, potts, alpha, split_alpha, target_nspix, video_mode ? "true" : "false");  

  // -- unpack shape --
  int nframes = vid.size(0);
  int height = vid.size(1);
  int width = vid.size(2);
  int nftrs = vid.size(3);
  int npix = height*width;
  int nbatch = 1;

  // -- legacy --
  int sm_start = 0;
  float sigma2_size = 0.0;

  // -- actually, not an input --
  int niters_seg = 4;
  // float split_alpha = 0.0;
  float merge_alpha = 0.0;

  // -- not controlled in python --
  float thresh_relabel = 1e-5;
  float thresh_new = 1e-2;

  // -- alloc options --
  auto options_f32 = torch::TensorOptions().dtype(torch::kFloat32)
    .layout(torch::kStrided).device(vid.device());
  auto options_i32 = torch::TensorOptions().dtype(torch::kInt32)
    .layout(torch::kStrided).device(vid.device());

  // -- allocate spix --
  torch::Tensor spix_th = torch::zeros({nframes, height, width}, options_i32);

  // -- init --
  // float* img_lab = (float*)easy_allocate(npix*3,sizeof(float));
  float* img_rgb;
  float* flow = nullptr;
  int* spix_prev = nullptr;
  SuperpixelParams* params_prev = nullptr;

  // -- start loop --
  for(int fidx=0; fidx < nframes; fidx++){
  

    // -- prepare images --
    img_rgb = vid[fidx].data_ptr<float>();
    // if (rgb2lab_b) {
    //   rgb2lab(img_rgb,img_lab,nbatch,npix); // convert image to LAB
    // }else {
    //   cudaMemcpy(img_lab,img_rgb,npix*3,cudaMemcpyDeviceToDevice);
    // }

    // -- unpack flow --
    if ((video_mode) and (fidx>0)){
      flow = flows[fidx-1].data_ptr<float>();
    }

    // -- init -- 
    int* spix = nullptr;
    bool* border = nullptr;
    int nspix = -1;
    SuperpixelParams* params = nullptr;

    if ((fidx == 0)||(video_mode == false)){
      // -- single image --
      auto out = run_bass(img_rgb, nbatch, height, width, nftrs,
                          niters, niters_seg, sm_start,
                          sp_size,sigma2_app,sigma2_size,
                          potts,alpha,split_alpha,target_nspix);
      spix = std::get<0>(out);
      border = std::get<1>(out);
      params = std::get<2>(out);

      // -- fill --
      torch::Tensor _spix_th = torch::from_blob(spix, {height, width}, options_i32);
      spix_th.index_put_({fidx}, _spix_th);

      // -- free data --
      // del _spix_th;
      cudaFree(border);

    }else{

      // -- shift & fill --
      auto out_saf = shift_and_fill(spix_prev,params_prev,flow,
                                    nbatch,height,width);
      int* filled_spix = std::get<0>(out_saf);
      int* shifted_spix = std::get<1>(out_saf);

      // -- count percentage invalid --
      int ninvalid = count_invalid(shifted_spix,npix);
      float iperc = ninvalid / (1.0*npix);
      if (iperc > 0.20){
        niters = 12;
      }else if(iperc < 0.01){
        niters = 4;
      }else {
        niters = 8;
      }

      // -- propogate --
      auto out = run_prop(img_rgb, nbatch, height, width, nftrs,
                          niters, niters_seg, sm_start,
                          sp_size,sigma2_app,sigma2_size,
                          potts,alpha,filled_spix,shifted_spix,params_prev,
                          thresh_relabel, thresh_new,
                          merge_alpha, split_alpha, target_nspix);
      spix = std::get<0>(out);
      border = std::get<1>(out);
      params = std::get<2>(out);

      // -- fill --
      torch::Tensor _spix_th = torch::from_blob(spix, {height, width}, options_i32);
      spix_th.index_put_({fidx}, _spix_th);

      // -- free data --
      // del _spix_th;
      cudaFree(border);

      // -- free --
      cudaFree(filled_spix);
      cudaFree(shifted_spix);

    }

    // -- [propogate info!] --
    if (fidx>0){
      cudaFree(spix_prev);
      free(params_prev);
    }
    if (fidx == (nframes-1)){
      cudaFree(spix);
      free(params);
    }else{
      spix_prev = spix;
      params_prev = params;
    }

  }
  // cudaFree(img_lab);



  return spix_th;
}


// std::tuple<torch::Tensor,torch::Tensor>
torch::Tensor
bist_forward_cuda(const torch::Tensor vid, const torch::Tensor flows,
                  int niters, int sp_size, float potts,
                  float sigma2_app, float alpha, float split_alpha,
                  bool video_mode, bool rgb2lab_b){

  // -- check --
  CHECK_INPUT(vid);
  CHECK_INPUT(flows);

  int target_nspix = 0;
  auto out = main_loop(vid, flows, niters,  sp_size,
                       sigma2_app, potts, alpha, split_alpha,
                       target_nspix, video_mode, rgb2lab_b);

  return out;
}





__global__ void GetImageOverlaid(float* filled, float* image, float* color,
                                 const bool* border, const int npix, const int xdim){
  int t = threadIdx.x + blockIdx.x * blockDim.x;  
  if (t>=npix) return;
  t = t + npix*blockIdx.y; // offset via batch

  if (border[t]){
    // -- for a nice grey --
    // filled[3*t] = 50;
    // filled[3*t+1] = 50;
    // filled[3*t+2] = 50;
    // -- for a sharp blue --
    // filled[3*t] = 0.0;
    // filled[3*t+1] = 0;
    // filled[3*t+2] = 1.0;
    filled[3*t] = color[0];
    filled[3*t+1] = color[1];
    filled[3*t+2] = color[2];
  }else{
    filled[3*t] = max(min(image[3*t],1.),0.0);
    filled[3*t+1] = max(min(image[3*t+1],1.),0.0);
    filled[3*t+2] = max(min(image[3*t+2],1.),0.0);
  }
    
}


__host__ void
CUDA_get_image_overlaid(float* filled, float* image, float* color,
                        const bool* border, const int npix,
                        const int xdim, const int nbatch){
  int num_block = ceil( double(npix) / double(THREADS_PER_BLOCK) ); 
  dim3 ThreadPerBlock(THREADS_PER_BLOCK,1);
  dim3 BlockPerGrid(num_block,nbatch);
  GetImageOverlaid<<<BlockPerGrid,ThreadPerBlock>>>(filled, image, color,
                                                    border, npix, xdim);
}


torch::Tensor get_marked_video(torch::Tensor vid,
                               torch::Tensor spix, torch::Tensor color){
  
  // -- check --
  CHECK_INPUT(vid);
  CHECK_INPUT(spix);
  CHECK_CONTIGUOUS(color);

  // -- unpack shape --
  int nframes = vid.size(0);
  int height = vid.size(1);
  int width = vid.size(2);
  int nftrs = vid.size(3);
  int npix = height*width;
  int nbatch = 1;

  // -- manage color input --
  long long ncolors = at::numel(color);
  assert(nftrs==ncolors);
  color = color.to(vid.device());

  // -- alloc border and marked image --
  auto options_f32 = torch::TensorOptions().dtype(torch::kFloat32)
    .layout(torch::kStrided).device(vid.device());
  torch::Tensor marked = torch::zeros({nframes, height, width, nftrs}, options_f32);

  // -- unpack pointers --
  float* _vid = vid.data_ptr<float>();
  int* _spix = spix.data_ptr<int>();
  float* _color = color.data_ptr<float>();
  float* _marked = marked.data_ptr<float>();

  // -- get the border --
  bool* border = (bool*)easy_allocate(nframes*npix,sizeof(bool));
  CudaFindBorderPixels_end(_spix, border, npix, nframes, width, height);

  // -- fill with marked values --
  CUDA_get_image_overlaid(_marked, _vid, _color, border, npix, width, nframes);

  // -- free memory --
  cudaFree(border);

  return marked;
}



void init_bist(py::module &m){
  m.def("bist_forward", &bist_forward_cuda,"BIST");
  m.def("get_marked_video", &get_marked_video,"get marked video");
  // m.def("bass_forward", &bass_forward_cuda,
  //       "BASS");
}

