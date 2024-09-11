
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
#include <thrust/system_error.h>
#include <thrust/system/cuda/error.h>
#include <vector>
#include <torch/torch.h>

#include "../bass/relabel.h"
#include "split_disconnected.h"

#include <stdio.h>
#ifndef WIN32
#include <unistd.h>
#endif

using namespace torch::indexing;

void throw_on_cuda_error_split(cudaError_t code) // yes; I realize this is silly.
{
  if(code != cudaSuccess){
    throw thrust::system_error(code, thrust::cuda_category());
  }
}


// __host__
// torch::Tensor split_disconnected(int* seg, int nbatch, int height, int width,
//                                      int npix, int nspix){

__host__
std::tuple<torch::Tensor,torch::Tensor>
run_split_disconnected_dev(int* seg, int nbatch, int height, int width, int nspix){

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
    torch::Tensor region_offsets = torch::zeros({nspix,max_nsplits+1},options_i32);
    compute_offsets<<<BlocksSuperPixels,NumThreads>>>(regions_spix.data<int>(),
                                                      region_sizes.data<int>(),
                                                      region_offsets.data<int>(),
                                                      nspix,nregions,max_nsplits+1);

    // -- allocate children -- 
    torch::Tensor children = torch::zeros({nspix,max_nsplits},options_i32);
    update_spix<<<BlocksPixels,NumThreads>>>(seg, regions.data<int>(),
                                             children.data<int>(),
                                             split_starts.data<int>(),
                                             region_offsets.data<int>(),
                                             npix, nspix, max_nsplits+1);
    return std::make_tuple(children,split_starts);
}

void init_split_disconnected(py::module &m){
  m.def("split_disconnected", &split_disconnected,
        "assign unique ids to each connected regions and track original label");
}

