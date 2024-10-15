/******************

   Compute \sum_{s=1}^N P(S_i=s)P(S_j=s)

*******************/


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


#define CUDA_NUM_THREADS 512
#define CHECK_CUDA(x) TORCH_CHECK(x.device().is_cuda(), #x " must be a CUDA tensor")
#define CHECK_CONTIGUOUS(x) TORCH_CHECK(x.is_contiguous(), #x " must be contiguous")
#define CHECK_INPUT(x) CHECK_CUDA(x); CHECK_CONTIGUOUS(x)


inline __host__ __device__
int get_window_start(const int index, const int length, const int neigh_size){
  int new_index = max(index - neigh_size,0);
  new_index += ((index+neigh_size)>=length) * (length - index - neigh_size - 1);
  return new_index;
}

inline __host__ __device__
void check_valid(bool& valid, const int ti, const int hi, const int wi,
                 const int T, const int H, const int W){
  valid = (ti <= (T-1)) and (ti >= 0);
  valid = valid and (hi <= (H-1)) and (hi >= 0);
  valid = valid and (wi <= (W-1)) and (wi >= 0);
}


// template <typename scalar_t>
__global__ void sim_sum_fwd_kernel(
    torch::PackedTensorAccessor32<float,5,torch::RestrictPtrTraits> simsum,
    const torch::PackedTensorAccessor32<float,5,torch::RestrictPtrTraits> sims,
    const torch::PackedTensorAccessor32<int,4,torch::RestrictPtrTraits> seg,
    const torch::PackedTensorAccessor32<int,6,torch::RestrictPtrTraits> flows,
    int nframes, int height, int width, int wt, int kernel_size, int npix, int nsearch){

    // -- compute indices -- 
    int nspix = sims.size(4);
    int ksize_sq = kernel_size*kernel_size;
    int pix_idx = threadIdx.x + blockIdx.x * blockDim.x;
    int offset = threadIdx.y + blockIdx.y * blockDim.y;
    if (pix_idx >= npix){ return; }
    if (offset >= nsearch){ return; }
    int bi = blockIdx.z / nframes;
    int ti = blockIdx.z % nframes;
    int offset_s = nsearch%ksize_sq;
    int offset_t = nsearch/ksize_sq;

    // -- compute indices --
    int hi = pix_idx / width;
    int wi = pix_idx % width;

    // -- flow offsets --
    int flow_hi = 0;
    int flow_wi = 0;
    if (offset_t > 0){
      flow_hi = flows[bi][ti][offset_t-1][1][hi][wi];
      flow_wi = flows[bi][ti][offset_t-1][0][hi][wi];
    }

    // -- compute starts --
    int neigh_size = (kernel_size-1)/2;
    int t_start = get_window_start(ti, nframes, wt); // need to be updated...
    int h_start = get_window_start(hi+flow_hi, height, neigh_size);
    int w_start = get_window_start(wi+flow_wi, width, neigh_size);

    // -- accumulate --
    int offset_h = offset_s / kernel_size;
    int offset_w = offset_s % kernel_size;
    int v_ti = t_start+offset_t;
    int v_hi = h_start+offset_h;
    int v_wi = w_start+offset_w;

    // -- accumulate if valid --
    bool valid = true;
    check_valid(valid, v_ti, v_hi, v_wi, nframes, height, width);
    if (not valid){ return; }

    // -- \sum_{s} p(s_i=s)p(s_j=s) --
    auto sims_i = sims[bi][ti][hi][wi];
    auto sims_j = sims[bi][v_ti][v_hi][v_wi];
    for (int _sdx=0; _sdx < nspix; _sdx++){
      // atomicAdd(&(simsum[bi][ti][hi][wi][offset]),sims_i[_sdx]*1);
      atomicAdd(&(simsum[bi][ti][hi][wi][offset]),sims_j[_sdx]*1);
      // atomicAdd(&(simsum[bi][ti][hi][wi][offset]),sims_i[_sdx]*sims_j[_sdx]);
      // atomicAdd(&(simsum[bi][ti][hi][wi][offset]),1);
    }
}

torch::Tensor sim_sum_forward_cuda(const torch::Tensor sims,
                                   const torch::Tensor seg,
                                   const torch::Tensor flows,
                                   const int kernel_size, const int wt){

    // sims.shape = (nbatch,nframes,height,width,nspix)
    // seg.shape = (nbatch,nframes,height,width)
    // flows.shape = (nbatch.nframes,offsets,height,width,2)

    // -- check --
    CHECK_INPUT(sims);
    CHECK_INPUT(seg);
    CHECK_INPUT(flows);

    // -- unpack --
    int nbatch = seg.size(0);
    int nframes = seg.size(1);
    int height = seg.size(2);
    int width = seg.size(3);
    int npix = height*width;
    int nspix = sims.size(4);

    // -- num of searching --
    int nsearch_t = 2*wt+1;
    int nsearch_s = kernel_size*kernel_size;
    int nsearch = nsearch_t*nsearch_s;

    // -- allocate --
    auto options_f32 = torch::TensorOptions().dtype(torch::kFloat32)
      .layout(torch::kStrided).device(seg.device());
    torch::Tensor simsum = torch::zeros({nbatch, nframes, height, width, nsearch},
                                        options_f32);

    // -- launch --
    int nthreads_pix = 32;
    int nthreads_search = 16;
    int nblocks_pix = (npix-1)/nthreads_pix+1;
    int nblocks_search = (nsearch-1)/nthreads_search+1;
    dim3 nthreads(nthreads_pix,nthreads_search);
    dim3 nblock(nblocks_pix,nblocks_search,nbatch*nframes);
    sim_sum_fwd_kernel<<< nblock, nthreads >>>(
        simsum.packed_accessor32<float,5,torch::RestrictPtrTraits>(),
        sims.packed_accessor32<float,5,torch::RestrictPtrTraits>(),
        seg.packed_accessor32<int,4,torch::RestrictPtrTraits>(),
        flows.packed_accessor32<int,6,torch::RestrictPtrTraits>(),
        nframes,height,width,wt,kernel_size,npix,nsearch);

    return simsum;
}




/***********************************************


               Backward Kernel


 ***********************************************/

// template <typename scalar_t>
// __global__ void sna_reweight_backward_kernel(
//     torch::PackedTensorAccessor32<scalar_t,5,torch::RestrictPtrTraits> d_attn_in,
//     torch::PackedTensorAccessor32<scalar_t,5,torch::RestrictPtrTraits> d_sims,
//     const torch::PackedTensorAccessor32<scalar_t,6,torch::RestrictPtrTraits> d_attn_out,
//     const torch::PackedTensorAccessor32<scalar_t,6,torch::RestrictPtrTraits> attn_out,
//     const torch::PackedTensorAccessor32<scalar_t,5,torch::RestrictPtrTraits> attn_in,
//     const torch::PackedTensorAccessor32<scalar_t,5,torch::RestrictPtrTraits> sims,
//     const torch::PackedTensorAccessor32<long,3,torch::RestrictPtrTraits> sinds,
//     int kernel_size, int NUM_PER_THREAD){

//     // -- unpack --
//     int nbatch = d_attn_in.size(0);
//     int nheads = d_attn_in.size(1);
//     int height = d_attn_in.size(2);
//     int width = d_attn_in.size(3);
//     int ksize_sq = d_attn_in.size(4);
//     int num_pix = height*width;
//     int sH = sims.size(3);
//     int sW = sims.size(4);

//     // -- compute indices -- 
//     int ibatch = blockIdx.z / nheads;
//     int ihead = blockIdx.z - ibatch * nheads;
//     int hw_raster = blockIdx.x * blockDim.x + threadIdx.x;
//     // int attn_offset = blockIdx.y * blockDim.y + threadIdx.y;
//     int attn_offset_s = NUM_PER_THREAD*(blockIdx.y * blockDim.y + threadIdx.y);
//     int si = threadIdx.z;

//     // -- boundary --
//     if (hw_raster >= num_pix){ return; }

//     // -- derivative indices --
//     int neigh_size = (kernel_size-1)/2;

//     // -- compute indices --
//     int hi = hw_raster / width;
//     int wi = hw_raster - hi * width;
//     int s_hi = sinds[hi][wi][0]+(si / 3 - 1);
//     int s_wi = sinds[hi][wi][1]+(si % 3 - 1);
//     bool valid = true;
//     check_valid(valid,s_hi,s_wi,sH,sW);
//     int h_start = get_window_start(hi, height, neigh_size);
//     int w_start = get_window_start(wi, width, neigh_size);

//     // -- accumulate --
//     for (int _idx=0; _idx < NUM_PER_THREAD; _idx++){

//       // -- indices --
//       int attn_offset = attn_offset_s + _idx;
//       if (attn_offset >= ksize_sq){ return; }
//       int h_offset = attn_offset / kernel_size;
//       int w_offset = attn_offset % kernel_size;
//       int v_hi = h_start+h_offset;
//       int v_wi = w_start+w_offset;

//       // -- read sims --
//       scalar_t sim_prob = valid ? sims[ibatch][v_hi][v_wi][s_hi][s_wi] : 0;

//       // -- derivatives ("l,i" = pixel idx ,"f" = feature idx, "j" = attn map idx) --
//       scalar_t attn_val = attn_in[ibatch][ihead][hi][wi][attn_offset];
//       scalar_t d_attn_val = d_attn_out[ibatch][ihead][si][hi][wi][attn_offset];

//       // -- derivatives --
//       if (valid){
//         atomicAdd(&(d_sims[ibatch][v_hi][v_wi][s_hi][s_wi]),d_attn_val*attn_val);
//         atomicAdd(&(d_attn_in[ibatch][ihead][hi][wi][attn_offset]),d_attn_val*sim_prob);
//       }

//     }
// }

// void sna_reweight_backward_cuda(torch::Tensor d_attn_in,
//                                  torch::Tensor d_sims,
//                                  const torch::Tensor d_attn_out,
//                                  const torch::Tensor attn_out,
//                                  const torch::Tensor attn_in,
//                                  const torch::Tensor sims,
//                                  const torch::Tensor sinds){

//     // -- check --
//     CHECK_INPUT(d_attn_in);
//     CHECK_INPUT(d_sims);
//     CHECK_INPUT(d_attn_out);
//     CHECK_INPUT(attn_out);
//     CHECK_INPUT(attn_in);
//     CHECK_INPUT(sims);
//     CHECK_INPUT(sinds);

//     // -- unpack --
//     int nbatch = d_attn_in.size(0);
//     int nheads = d_attn_in.size(1);
//     int height = d_attn_in.size(2);
//     int width = d_attn_in.size(3);
//     int ksize_sq = d_attn_in.size(4);
//     int kernel_size = std::sqrt(ksize_sq);
//     int num_pix = height*width;
//     int nsuperpixels = 9;
//     int NUM_PER_THREAD = 4;

//     // -- block --
//     int nthreads_pix = 14;
//     int nthreads_ksize = 8;
//     // int nthreads_pix = 12;
//     // int nthreads_ksize = 8;
//     int nblocks_pix = (num_pix-1)/nthreads_pix+1;
//     int nblocks_ksize = (ksize_sq-1)/(NUM_PER_THREAD*nthreads_ksize)+1;
//     dim3 nthreads(nthreads_pix,nthreads_ksize,nsuperpixels);
//     dim3 nblock(nblocks_pix,nblocks_ksize,nbatch*nheads);
//     AT_DISPATCH_FLOATING_TYPES(d_attn_in.type(), "backward_kernel", ([&] {
//         sna_reweight_backward_kernel<scalar_t><<< nblock, nthreads >>>(
//             d_attn_in.packed_accessor32<scalar_t,5,torch::RestrictPtrTraits>(),
//             d_sims.packed_accessor32<scalar_t,5,torch::RestrictPtrTraits>(),
//             d_attn_out.packed_accessor32<scalar_t,6,torch::RestrictPtrTraits>(),
//             attn_out.packed_accessor32<scalar_t,6,torch::RestrictPtrTraits>(),
//             attn_in.packed_accessor32<scalar_t,5,torch::RestrictPtrTraits>(),
//             sims.packed_accessor32<scalar_t,5,torch::RestrictPtrTraits>(),
//             sinds.packed_accessor32<long,3,torch::RestrictPtrTraits>(),
//             kernel_size,NUM_PER_THREAD);
//         }));

// }


void init_sim_sum(py::module &m){
  m.def("sim_sum_fwd", &sim_sum_forward_cuda,
        "neighborhood superpixel atten forward");
  // m.def("sim_sum_bwd", &sim_sum_backward_cuda,
  //       "neighborhood superpixel atten backward");
}

