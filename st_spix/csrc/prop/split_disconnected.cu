
#define THREADS_PER_BLOCK 512


// #include "init_prop_seg.h"
// #ifndef MY_SP_SHARE_H
// #define MY_SP_SHARE_H
// #include "../bass/share/sp.h"
// #endif
// #include "../bass/core/Superpixels.h"
// // #include "../share/utils.h"
// #include <thrust/system_error.h>
// #include <thrust/system/cuda/error.h>

#include <assert.h>
#include <torch/torch.h>

#include "../bass/relabel.h"
#ifndef SPLIT_DISC
#define SPLIT_DISC
#include "split_disconnected.h"
#endif

using namespace torch::indexing;

void throw_on_cuda_error_split(cudaError_t code) // yes; I realize this is silly.
{
  if(code != cudaSuccess){
    throw thrust::system_error(code, thrust::cuda_category());
  }
}

// -- define --
#define CHECK_CUDA(x) TORCH_CHECK(x.device().is_cuda(), #x " must be a CUDA tensor")
#define CHECK_CONTIGUOUS(x) TORCH_CHECK(x.is_contiguous(), #x " must be contiguous")
#define CHECK_INPUT(x) CHECK_CUDA(x); CHECK_CONTIGUOUS(x)

// __host__
// torch::Tensor split_disconnected(int* seg, int nbatch, int height, int width,
//                                      int npix, int nspix){

__host__
std::tuple<torch::Tensor,torch::Tensor>
run_split_disconnected(int* seg, int nbatch, int height, int width, int nspix){

    // -- better names --
    int H = height;
    int W = width;
    int npix = H*W;

    // -- init launch info --
    int nblocks_for_npix = ceil( double(npix) / double(THREADS_PER_BLOCK) ); 
    dim3 BlocksPixels(nblocks_for_npix,nbatch);
    int nblocks_for_nspix = ceil( double(nspix) / double(THREADS_PER_BLOCK) ); 
    dim3 BlocksSuperPixels(nblocks_for_nspix,nbatch);
    dim3 ThreadPerBlock(THREADS_PER_BLOCK,1);
    dim3 NumThreads(THREADS_PER_BLOCK,1);
    const int sizeofint = sizeof(int);

    // -- allocate --
    torch::Device device(torch::kCUDA, 0); 
    auto options_i32 = torch::TensorOptions().dtype(torch::kInt32)
      .layout(torch::kStrided).device(device);

    // -- wrap superpixel segmentation within pytorch tensor --
    torch::Tensor spix_th = torch::from_blob(seg,{npix},options_i32);

    // -- get regions --
    auto [regions,region_ids] = get_regions(seg,nbatch,H,W,device);
    int nregions = region_ids.sizes()[0];
    torch::Tensor regions_spix = spix_th.index({region_ids}); // size = nregions

    // -- get size of each regions --
    torch::Tensor region_sizes = torch::zeros({nregions},options_i32);
    compute_region_sizes<<<BlocksPixels,NumThreads>>>(region_sizes.data<int>(),
                                                      regions.data<int>(),
                                                      npix,nregions);

    // -- count the number of regions for each superpixel --
    int blocksRegion = ceil( double(nregions) / double(THREADS_PER_BLOCK) ); 
    dim3 BlocksRegion(blocksRegion,nbatch);
    torch::Tensor spix_counts = torch::zeros({nspix},options_i32);
    get_spix_counts<<<BlocksRegion,NumThreads>>>(regions_spix.data<int>(),
                                                 spix_counts.data<int>(),
                                                 nregions);

    // -- get the starting index for each superpixel's new split region --
    auto [split_starts,max_nsplits] = get_split_starts(spix_counts);

    // -- get superpixel label offset from nspix for each disconnected region --
    // torch::Tensor region_offsets = -torch::ones({nspix,max_nsplits+1},options_i32);
    torch::Tensor region_offsets = -torch::ones({nregions},options_i32);
    compute_offsets<<<BlocksSuperPixels,NumThreads>>>(regions_spix.data<int>(),
                                                      region_sizes.data<int>(),
                                                      region_offsets.data<int>(),
                                                      nspix,nregions,max_nsplits+1);

    // -- allocate children -- 
    torch::Tensor children = -torch::ones({nspix,max_nsplits},options_i32);
    update_spix<<<BlocksPixels,NumThreads>>>(seg, regions.data<int>(),
                                             children.data<int>(),
                                             split_starts.data<int>(),
                                             region_offsets.data<int>(),
                                             npix, nspix, max_nsplits);

    // cudaMemset(seg, 100, sizeofint); // check
    return std::make_tuple(children,split_starts);
}


/**************************************************

   Smaller Functions to Keep Main Function Tidy

***************************************************/

__host__
std::tuple<torch::Tensor,torch::Tensor>
get_regions(int* seg, int nbatch, int H, int W, torch::Device device){

    // -- allocate --
    int npix = H*W;
    auto options_i32 = torch::TensorOptions().dtype(torch::kInt32)
      .layout(torch::kStrided).device(device);
    torch::Tensor regions = torch::arange(npix,torch::kInt32).to(device);
    torch::Tensor locs = regions.clone();
    int* regions_ptr = regions.data<int>();
    int* locs_ptr = locs.data<int>();
    const int sizeofint = sizeof(int);

    // -- blocks --
    int nblocks_for_npix = ceil( double(npix) / double(THREADS_PER_BLOCK) ); 
    dim3 BlocksPixels(nblocks_for_npix,nbatch);
    dim3 NumThreads(THREADS_PER_BLOCK,1);

    // -- allocate [change] --
    int changes = 1;
    int* changes_gpu;
    try {
      throw_on_cuda_error_split(cudaMalloc((void**)&changes_gpu,sizeofint));
    }
    catch (thrust::system_error& e) {
        std::cerr << "CUDA error after cudaMalloc: " << e.what() << std::endl;
    }

    // -- compute minimum within each connected region [locs=prev,regions=curr] --
    while(changes){

      cudaMemcpy(locs_ptr,regions_ptr,npix*sizeofint,cudaMemcpyDeviceToDevice);
      cudaMemset(changes_gpu, 0, sizeofint);
      cudaMemcpy(&changes, changes_gpu, sizeofint, cudaMemcpyDeviceToHost);
      find_spix_min<<<BlocksPixels,NumThreads>>>(seg,regions_ptr,locs_ptr,
                                                 changes_gpu,H,W,npix);
      cudaMemcpy(&changes,changes_gpu,sizeofint,cudaMemcpyDeviceToHost);
    }
    cudaFree(changes_gpu); // -- free --

    // -- relabel regions --
    auto region_ids = std::get<0>(at::_unique(regions));
    int nregions = region_ids.sizes()[0];
    relabel_spix<false><<<BlocksPixels,NumThreads>>>(regions.data<int>(),
                                                     region_ids.data<int>(),
                                                     npix, nregions);

    return std::make_tuple(regions,region_ids);
}

__host__
std::tuple<torch::Tensor,int>
get_split_starts(torch::Tensor spix_counts){

    // -- get the number of superpixel occurances per region --
    spix_counts = spix_counts - 1; // size()[0] = nspix
    auto args = torch::where(spix_counts <= 0).at(0);
    auto zero = torch::tensor({0}, torch::kInt32);
    spix_counts = spix_counts.index_put_({args}, zero);
    int num_splits = torch::sum(spix_counts.index({args})).item<int>();

    // -- put the starting additional spix num at the correct location --
    // auto split_starts = -torch::ones_like(spix_counts);
    auto split_starts = torch::zeros_like(spix_counts);
    args = torch::where(spix_counts > 0).at(0);
    auto cumsum_elements = torch::cumsum(spix_counts.index({args}),0).to(torch::kInt32);
    spix_counts = spix_counts.index_put_({args}, cumsum_elements);

    // base case; no split spix
    if (args.sizes()[0] == 0){
      return std::make_tuple(split_starts,0);
    }

    // -- take max number of splits for a single superpixel --
    int max_nsplits = torch::max(cumsum_elements).item<int>();

    // -- finally, get the starting index for each split --
    // args = th.where(spix_conts>0)
    // split_starts = -th.ones_like(spix_counts)
    // split_starts[args] = th.cumsum(spix_counts[args]) - spix_counts[args[0]]
    auto first_num = spix_counts.index({args[0]});
    assert(first_num.item<int>()>=1);//must be at least one.
    // fprintf(stdout,"first num: %d\n",first_num.item<int>());
    split_starts = split_starts.index_put_({args}, cumsum_elements) - first_num;

    return std::make_tuple(split_starts,max_nsplits);
}


/**************************************************

               Kernels Start Here

***************************************************/

__global__
void update_spix(int* spix, int* regions, int* children,
                 int* split_starts, int* region_offsets,
                 int npix, int nspix, int max_nchild){

  // -- get pixel index --
  int idx = threadIdx.x + blockIdx.x*blockDim.x;
  if (idx>=npix)  return;
  
  // -- get index --
  int* spix_idx = spix + idx;
  int region_idx = *(regions + idx);
  int split_start = *(split_starts + spix_idx[0]);
  int region_offset = *(region_offsets + region_idx);
  int* child_ptr = children + spix_idx[0]*max_nchild;
  // int region_offset = *(region_offsets + spix_idx[0]*(max_nchild+1));
  // int region_offset = *(region_offsets + spix_idx[0]*(max_nchild+1) + region_idx);

  // if (split_start >= 0){ // only non-negative if valid
  // // if (region_offset >= 0){
  //   assert(region_offset>=0);
  if (region_offset>0){
    assert(split_start >= 0);
    spix_idx[0] = nspix + split_start + region_offset;
    (child_ptr + region_offset-1)[0] = nspix + split_start + region_offset;
  }

}

__global__
void compute_offsets(int* regions_spix, int* region_sizes,
                     int* region_offsets, int nspix, int nregions, int max_nchild){

  // -- get spix index --
  int spix = threadIdx.x + blockIdx.x*blockDim.x;
  if (spix>=nspix)  return;
  // ? shouldn't this be "nregions" size?

  // -- the loop --
  // int r_max = -1;
  // int r_zero = -1;
  int curr_max = -1;
  int child_count = 0;
  int idx_max;
  int idx_zero;
  for (int r=0; r < nregions; r++){
    if (regions_spix[r] == spix){

      // -- get zero index --
      if (child_count == 0){
        idx_zero = r;
      }

      // -- get offset --
      region_offsets[r] = child_count;

      // -- get max size --
      int count = region_sizes[r];
      if (count > curr_max){
        idx_max = r;
        curr_max = count;
      }

      // -- update child count --
      child_count++;

    }
  }

  // -- swap largest region to zero offset --
  if ((idx_max != idx_zero) and (child_count > 0)){
    region_offsets[idx_zero] = region_offsets[idx_max];
    region_offsets[idx_max] = 0;
    // region_offsets[spix*max_nchild] = idx_max;
    // region_offsets[spix*max_nchild+idx_max] = 0;
    // region_offsets[spix*max_nchild+r_zero] = region_offsets[spix*max_nchild+r_max];
    // region_offsets[spix*max_nchild+r_max] = 0;
  }


}

__global__
void get_spix_counts(int* regions_spix, int* spix_counts, int nregions){
  /*************************
   
     "regions_spix" is the superpixel associated with each region; length "nregions"
     "spix_counts" is the number of counts associated with each spix; length "nspix"

  *************************/
  // -- get index --
  int idx = threadIdx.x + blockIdx.x*blockDim.x;
  if (idx>=nregions)  return;
  int spix = regions_spix[idx];
  atomicAdd(&spix_counts[spix],1);
}

__global__
void compute_region_sizes(int* size, int* regions, int npix, int nregions){
  int idx = threadIdx.x + blockIdx.x*blockDim.x;
  if (idx>=npix)  return;
  int region = regions[idx];
  assert(region<nregions);
  atomicAdd(&size[region],1);
}


__global__
void find_spix_min(int* seg, int* curr, int* prev,
                   int* changes, int H, int W, int npix){
  
  // -- get index --
  int idx = threadIdx.x + blockIdx.x*blockDim.x;
  // atomicAdd(&changes[0],1);
  if (idx>=npix)  return;
  int hi = idx / W;
  int wi = idx % W;

  // -- only check locally --
  int neigh_idx;
  bool oob,match;
  #pragma unroll
  for (int ix=-2;ix<=2;ix++){
  #pragma unroll
    for (int jx=-2;jx<=2;jx++){
      oob = check_oob(hi,ix,wi,jx,H,W);
      // if (oob){ continue; }
      if (not oob){
        neigh_idx = (hi+ix)*W + (wi+jx);
        match = seg[idx] == seg[neigh_idx];
        if (match){  curr[idx] = min(curr[idx],prev[neigh_idx]); }
      }
    }
  }

  // -- detect changes --
  atomicAdd(changes,curr[idx] != prev[idx]);
  // atomicAdd(changes,curr[idx] == prev[idx]);
  // atomicAdd(changes,1);

}

/**********************************************************

             -=-=-=-=- Python API  -=-=-=-=-=-

***********************************************************/

std::tuple<torch::Tensor,torch::Tensor,torch::Tensor>
split_disconnected(const torch::Tensor spix, int nspix){

  // -- check input --
  CHECK_INPUT(spix);

  // -- unpack --
  int nbatch = spix.size(0);
  int height = spix.size(1);
  int width = spix.size(2);
  assert(nbatch==1); // currently, must be batchsize of 1
  
  // -- main function --
  auto fxn_outs = run_split_disconnected(spix.data<int>(), nbatch,
                                         height, width, nspix);
  // -- return --
  // return std::make_tuple(spix,children);
  auto full_outs = std::tuple_cat(std::make_tuple(spix),fxn_outs);
  return full_outs;

}


void init_split_disconnected(py::module &m){
  m.def("split_disconnected", &split_disconnected,
        "assign unique ids to each connected regions and track original label");
}

