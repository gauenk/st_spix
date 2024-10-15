

// -- project imports --
#include "pch.h"
#include "sp_video_pooling.h"

/********************************************


                Forward


********************************************/

__global__
void run_sp_video_downsample(float* img, int* seg,
                             float* downsampled, float* downcount,
                             int nspix, int nframes, int npix, int nftrs){
  
  // -- get pixel index --
  int pix_idx = threadIdx.x + blockIdx.x*blockDim.x;
  if (pix_idx>=npix) return;
  pix_idx = pix_idx + npix*(blockIdx.y*nframes+blockIdx.z);

  // -- get segmentation index --
  int seg_idx = seg[pix_idx];
  if (seg_idx < 0){ return; }

  // -- add to downsampled --
  float* imgF = img + pix_idx * nftrs;
  float* dsF = downsampled + seg_idx*nftrs + nspix*nftrs*blockIdx.y;
  float* dsC = downcount + seg_idx + nspix*blockIdx.y;
  for (int fidx = 0; fidx < nftrs; fidx++){
    atomicAdd(dsF+fidx,*(imgF+fidx));
  }
  atomicAdd(dsC,static_cast<float>(1));
}

__global__
void run_sp_video_pooling(float* pooled, int* seg, float* downsampled,
                          int nspix, int nframes, int npix, int nftrs){

  // -- get pixel index --
  int pix_idx = threadIdx.x + blockIdx.x*blockDim.x;
  if (pix_idx>=npix) return;
  pix_idx = pix_idx + npix*(blockIdx.y*nframes+blockIdx.z);

  // -- get segmentation index --
  int seg_idx = seg[pix_idx];
  if (seg_idx < 0){ return; }

  // -- write to pooled --
  float* dsF = downsampled + seg_idx * nftrs + nspix*nftrs*blockIdx.y;
  float* poolF = pooled + pix_idx * nftrs;
  for (int fidx = 0; fidx < nftrs; fidx++){
    *(poolF+fidx) = *(dsF+fidx);
  }

}

std::tuple<torch::Tensor,torch::Tensor,torch::Tensor>
sp_video_pooling(const torch::Tensor img, const torch::Tensor seg){


  // -- check --
  CHECK_INPUT(img);
  CHECK_INPUT(seg);

  // -- unpack --
  int nbatch = img.size(0);
  int nframes = img.size(1);
  int height = img.size(2);
  int width = img.size(3);
  int nftrs = img.size(4);
  int npix = height*width;

  // -- get max num of spix --
  int nspix = seg.max().item<int>()+1;

  // -- pointers --
  float* img_ptr = img.data<float>();
  int* seg_ptr = seg.data<int>();

  // -- alloc options --
  auto options_f32 = torch::TensorOptions().dtype(torch::kFloat32)
    .layout(torch::kStrided).device(img.device());
  auto options_i32 = torch::TensorOptions().dtype(torch::kFloat32)
    .layout(torch::kStrided).device(img.device());

  // -- init pooled --
  torch::Tensor pooled = torch::zeros({nbatch, nframes, height, width, nftrs},
                                      options_f32);
  float* pooled_ptr = pooled.data<float>();

  // -- init downsampled & counts --
  torch::Tensor downsampled = torch::zeros({nbatch, nspix, nftrs}, options_f32);
  float* downsampled_ptr = downsampled.data<float>();
  torch::Tensor counts = torch::zeros({nbatch, nspix}, options_f32);
  float* counts_ptr = counts.data<float>();

  // -- launch pooling --
  int num_block = ceil( double(npix) / double(THREADS_PER_BLOCK) ); 
  dim3 BlockPerGrid(num_block,nbatch,nframes);
  dim3 ThreadPerBlock(THREADS_PER_BLOCK,1);
  run_sp_video_downsample<<<BlockPerGrid,ThreadPerBlock>>>
    (img_ptr, seg_ptr, downsampled_ptr, counts_ptr, nspix, nframes, npix, nftrs);
  downsampled /= (counts.unsqueeze(2) + 1e-10); // normalize in-place
  run_sp_video_pooling<<<BlockPerGrid,ThreadPerBlock>>>
    (pooled_ptr, seg_ptr, downsampled_ptr, nspix, nframes, npix, nftrs);

  return std::make_tuple(pooled,downsampled,counts);
}

// /********************************************


//                 Backward


// ********************************************/


// std::tuple<torch::Tensor>
// sp_video_pooling_bwd(const torch::Tensor pgrad, const torch::Tensor seg){

//   // -- check --
//   CHECK_INPUT(pgrad);
//   CHECK_INPUT(seg);

//   // -- unpack --
//   int nbatch = pgrad.size(0);
//   int nframes = pgrad.size(1);
//   int height = pgrad.size(2);
//   int width = pgrad.size(3);
//   int nftrs = pgrad.size(4);
//   int npix = height*width;

//   // -- get max num of spix --
//   int nspix = seg.max().item<int>()+1;

//   // -- pointers --
//   float* img_ptr = img.data<float>();
//   int* seg_ptr = seg.data<int>();

//   // -- alloc options --
//   auto options_f32 = torch::TensorOptions().dtype(torch::kFloat32)
//     .layout(torch::kStrided).device(img.device());
//   auto options_i32 = torch::TensorOptions().dtype(torch::kFloat32)
//     .layout(torch::kStrided).device(img.device());

//   // -- init pooled --
//   torch::Tensor pooled = torch::zeros({nbatch, nframes, height, width, nftrs},
//                                       options_f32);
//   float* pooled_ptr = pooled.data<float>();

//   // -- init downsampled & counts --
//   torch::Tensor downsampled = torch::zeros({nbatch, nspix, nftrs}, options_f32);
//   float* downsampled_ptr = downsampled.data<float>();
//   torch::Tensor counts = torch::zeros({nbatch, nspix}, options_f32);
//   float* counts_ptr = counts.data<float>();

//   // -- launch pooling --
//   int num_block = ceil( double(npix) / double(THREADS_PER_BLOCK) ); 
//   dim3 BlockPerGrid(num_block,nbatch,nframes);
//   dim3 ThreadPerBlock(THREADS_PER_BLOCK,1);
//   run_sp_video_downsample<<<BlockPerGrid,ThreadPerBlock>>>
//     (img_ptr, seg_ptr, downsampled_ptr, counts_ptr, nspix, nframes, npix, nftrs);
//   downsampled /= (counts.unsqueeze(2) + 1e-10); // normalize in-place
//   run_sp_video_pooling<<<BlockPerGrid,ThreadPerBlock>>>
//     (pooled_ptr, seg_ptr, downsampled_ptr, nspix, nframes, npix, nftrs);

//   return std::make_tuple(pooled,downsampled,counts);

//   // // -- allocate --
//   // auto options_f32 = torch::TensorOptions().dtype(torch::kFloat32)
//   //   .layout(torch::kStrided).device(device);
//   // torch::Tensor img_grad = torch::zeros({nbatch, nframes, height, width, nftrs},
//   //                                       options_f32);
//   // float* img_grad_ptr = img_grad.data<float>();

//   // // // -- init downsampled & counts --
//   // // torch::Tensor downsampled = torch::zeros({nbatch, nspix, nftrs}, options_f32);
//   // // float* downsampled_ptr = downsampled.data<float>();
//   // // torch::Tensor counts = torch::zeros({nbatch, nspix, 1}, options_f32);
//   // // float* counts_ptr = counts.data<float>();

//   // // // -- launch pooling --
//   // // int num_block = ceil( double(npix) / double(THREADS_PER_BLOCK) ); 
//   // // dim3 BlockPerGrid(num_block);
//   // // dim3 ThreadPerBlock(THREADS_PER_BLOCK,1);
//   // // scatter_bwd<<<BlockPerGrid,ThreadPerBlock>>>
//   // //   (img_ptr, seg_ptr, downsampled_ptr, counts_ptr, npix, nftrs);


//   // // downsampled /= (counts + 1e-10); // normalize in-place
//   // // run_sp_pooling<<<BlockPerGrid,ThreadPerBlock>>>
//   // //   (pooled_ptr, seg_ptr, downsampled_ptr, npix, nftrs);

//   // return std::make_tuple(img_grad);
// }



/********************************************

           Upscale Pooled Features

********************************************/

torch::Tensor
downsampled_video_to_pooled(const torch::Tensor downsampled,
                            const torch::Tensor seg){

  // -- check --
  CHECK_INPUT(downsampled);
  CHECK_INPUT(seg);

  // -- unpack --
  int nbatch = seg.size(0);
  int nframes = seg.size(1);
  int height = seg.size(2);
  int width = seg.size(3);
  int nftrs = downsampled.size(2);
  int nspix = downsampled.size(1);
  int npix = height*width;

  // -- get max num of spix --
  int _nspix = seg.max().item<int>()+1;
  assert(_nspix <= nspix);

  // -- pointers --
  float* downsampled_ptr = downsampled.data<float>();
  int* seg_ptr = seg.data<int>();

  // -- alloc options --
  auto options_f32 = torch::TensorOptions().dtype(torch::kFloat32)
    .layout(torch::kStrided).device(seg.device());

  // -- init pooled --
  torch::Tensor pooled = torch::zeros({nbatch, nframes, height, width, nftrs},
                                      options_f32);
  float* pooled_ptr = pooled.data<float>();

  // -- launch pooling --
  int num_block = ceil( double(npix) / double(THREADS_PER_BLOCK) ); 
  dim3 BlockPerGrid(num_block,nbatch,nframes);
  dim3 ThreadPerBlock(THREADS_PER_BLOCK,1);
  run_sp_video_pooling<<<BlockPerGrid,ThreadPerBlock>>>
    (pooled_ptr, seg_ptr, downsampled_ptr, nspix, nframes, npix, nftrs);

  return pooled;
}


void init_sp_video_pooling(py::module &m){
  m.def("sp_video_pooling", &sp_video_pooling,"superpixel pooling");
  m.def("downsampled_video_to_pooled",
        &downsampled_video_to_pooled,"downsampled_video_to_pooled");
  // m.def("sp_video_pooling_bwd", &sp_video_pooling_bwd,"superpixel pooling bwd");
  // m.def("downsampled_to_pooled", &downsampled_to_pooled,
  //       "upscale from downsampled features");
  // m.def("sp_pooling_bwd", &sp_pooling_bwd,"superpixel pooling bwd");
}

