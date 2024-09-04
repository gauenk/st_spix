
#ifndef OUT_OF_BOUNDS_LABEL
#define OUT_OF_BOUNDS_LABEL -1
#endif

#ifndef BAD_TOPOLOGY_LABEL 
#define BAD_TOPOLOGY_LABEL -2
#endif

#ifndef NUM_OF_CHANNELS 
#define NUM_OF_CHANNELS 3
#endif


#ifndef USE_COUNTS
#define USE_COUNTS 1
#endif


#ifndef OUT_OF_BOUNDS_LABEL
#define OUT_OF_BOUNDS_LABEL -1
#endif

#define THREADS_PER_BLOCK 512


// #include "init_prop_seg.h"
// #ifndef MY_SP_SHARE_H
// #define MY_SP_SHARE_H
// #include "../bass/share/sp.h"
// #endif
// #include "../bass/core/Superpixels.h"
// // #include "../share/utils.h"

#include "../bass/relabel.h"
#include "split_disconnected.h"

#include <stdio.h>
#ifndef WIN32
#include <unistd.h>
#endif


__host__
void run_split_disconnected(float* img, int* seg,
                            int* missing, bool* border,
                            superpixel_params* sp_params, 
                            const int nPixels, const int nMissing,
                            int nbatch, int xdim, int ydim, int nftrs,
                            const float3 J_i, const float logdet_Sigma_i, 
                            float i_std, int s_std, int nInnerIters,
                            const int nSPs, int nSPs_buffer,
                            float beta_potts_term, int* debug_spix,
                            bool* debug_border, bool debug_fill){

    // -- init launch info --
    int num_block_sub = ceil( double(nPixels) / double(THREADS_PER_BLOCK) ); 
    dim3 BlockPerGridSub(num_block_sub,nbatch);
    int num_block = ceil( double(nMissing) / double(THREADS_PER_BLOCK) ); 
    dim3 BlockPerGrid(num_block,nbatch);
    dim3 ThreadPerBlock(THREADS_PER_BLOCK,1);
    const int sizeofint = sizeof(int);

    // -- allocate --
    int H = ydim;
    int W = xdim;
    int nspix = nSPs;
    torch::Device device(torch::kCUDA, 0); 
    auto options_i32 =torch::TensorOptions().dtype(torch::kInt32)
      .layout(torch::kStrided).device(device);
    torch::Tensor regions_th = torch::arange(npix).to(torch::kInt32);
    torch::Tensor locs_th = regions_th.clone();
    int* regions = regions_th.data<int>();
    int* locs = locs_th.data<int>();

    // -- allocate [change] --
    int changes = 1;
    int* changes_gpu;
    try {
      throw_on_cuda_error(cudaMalloc((void**)&changes_gpu,sizeofint));
      // throw_on_cuda_error(malloc((void*)num_neg_cpu,sizeofint));
    }
    catch (thrust::system_error& e) {
        std::cerr << "CUDA error after cudaMalloc: " << e.what() << std::endl;
    }

    // -- compute minimum within each connected region --
    while(changes){
      cudaMemcpy(locs,regions,npix*sizeofint,cudaMemcpyHostToHost);
      cudaMemset(changes_gpu, 0, sizeofint);
      find_spix_min<<<BlockPerGrid,ThreadPerBlock>>>\
        (seg,regions,locs,changes_gpu,npix,H,W);
      cudaMemcpy(&changes, changes_gpu, sizeofint, cudaMemcpyDeviceToHost);
    }
    cudaMemcpy(locs,regions,npix*sizeofint,cudaMemcpyHostToHost);

    // -- free --
    cudaFree(changes_gpu);

    // -- index superpixel labels at "min locations" --
    torch::Tensor spix_th = torch::tensor(spix,{nspix},options_i32);
    torch::Tensor spix_regions = spix_th.index(locs_th);

    // -- relabel "min locations" to "region labels" --
    auto unique_ids = std::get<0>(at::_unique(regions_th));
    int nregions = unique_ids.sizes()[0];
    relabel_spix<false><<<nblocks1,nthreads1>>>(regions, unique_ids.data<int>(),
                                                npix, nregions);

    // -- get size of each regions --
    torch::Tensor region_size = torch::zeros({nregions},options_i32);
    compute_region_size<<<nblocks1,nthreads1>>>(region_size.data<int>(),
                                                regions,npix, nregions);

    // -- count the number of regions for each superpixel --
    torch::Tensor spix_counts_th = torch::zeros({nspix},options_i32);
    spix_counts<<<nblocks1,nthreads1>>>(spix_regions.data<int>(),
                                        spix_counts_th.data<int>(), nspix);

    // -- ... --
    // assert(torch::all(spix_counts_th>0))
    spix_counts_th = spix_counts_th - 1; // lenght of superpixel
    int num_splits = torch::sum(spix_counts_th);
    // args = th.where(spix_conts_th>0)
    // spix_counts_th[args] = th.cumsum(spix_counts_th[args])
    // then, this gives the offsets from "nspix" for each
    // superpixel's new set of splitted superpixels...
    // for example, we can have many "0"s, then "2", then "1", and then "0"s
    // then we have "num_splits" be "0", "2", "3", and then "0" which can be used to
    // assign the new
    
    // -- count nsplits per spix --
    torch::Tensor nsplits = torch::zeros({nspix}, options_i32);
    // count_num_splits<<<nblocks1,nthreads1>>>(spix, regions, nsplits.data<int>(),
    //                                          npix, nspix, nregions);

    
    // -- compute average of minimum values --
    // region_summary(seg, regions, spix_regions, count_regions, npix, H, W);

}


__device__ inline
bool check_oob(hi,ix,wi,jx,H,W){
  bool oob = (hi+ix)<0 or (hi+ix)>(H-1);
  oob = oob or (wi+jx)<0 or (wi+jx)>(W-1);
  return oob;
}

__global__
void spix_counts(int* spix_regions, int* spix_counts, int nspix){
  // -- get index --
  int idx = threadIdx.x + blockIdx.x*blockDim.x;
  if (idx>=nregions)  return;
  spix = spix_regions[idx];
  atomicAdd(&spix_counts[spix],1);
}

// __global__
// void count_num_splits(int* spix, int* curr, int* nsplits,
//                       int npix, int nspix, int nregions){

// }

__global__
void compute_region_size(int* size, int* regions,
                         int npix, int nregions){
  int idx = threadIdx.x + blockIdx.x*blockDim.x;
  if (idx>=npix)  return;
  int region = regions[idx];
  // assert(region<nregions);
  atomicAdd(&size[region],1);
}

// __global__
// void region_summary(int* seg, int* regions,
//                     int * spix_regions,
//                     int * count_regions,
//                     int nregions, int npix, int H, int W){
//   // -- get index --
//   int idx = threadIdx.x + blockIdx.x*blockDim.x;
//   if (idx>=npix)  return;
//   atomicAdd(regions[ridx],1);
//   r = regions[idx];
// }

// idk proper pytorch c++ api; this is easier
__global__
void region_spix(int* seg, int* regions,
                 int * spix_regions,
                 int * count_regions,
                 int nregions, int npix, int H, int W){
  // -- get index --
  int idx = threadIdx.x + blockIdx.x*blockDim.x;
  if (idx>=npix)  return;
  atomicAdd(regions[ridx],1);
  r = regions[idx];
}

__global__
void find_spix_min(int* seg, int* curr, int* prev,
                   int* changes, int npix, int H, int W){
  
  // -- get index --
  int idx = threadIdx.x + blockIdx.x*blockDim.x;
  if (idx>=npix)  return;
  int hi = idx / width;
  int wi = idx % width;

  // -- only check locally --
  int neigh_idx;
  bool oob,match;
  #pragma unroll
  for (ix=-1;ix<=1;ix++){
  #pragma unroll
    for (jx=-1;jx<=1;jx++){
      oob = check_oob(hi,ix,wi,jx,H,W);
      if (oob){ continue; }
      neigh_idx = (hi+ix)*W + wi+jx;
      match = spix[idx] == spix[neigh_idx];
      if (match){  curr[idx] = min(curr[idx],prev[neigh_idx]); }
    }
  }

  // -- detect changes --
  atomicAdd(changes,curr[idx] != prev[hi,wi]);

}
