
/********************************************************************

      Run BASS using the propograted superpixel segs and params

********************************************************************/

// -- cpp imports --
#include <stdio.h>
#include "pch.h"

// -- "external" import --
#ifndef MY_SP_STRUCT
#define MY_SP_STRUCT
#include "../bass/share/my_sp_struct.h"
#endif

// -- utils --
#include "rgb2lab.h"
#include "sparams_io.h"
#include "seg_utils.h"
#include "init_utils.h"
#include "init_sparams.h"
#include "init_seg.h"
#include "../bass/relabel.h"

// -- primary functions --
#include "prop_bass.h"
#include "split_merge.h"
#include "update_params.h"
#include "update_seg.h"


/**********************************************************

             -=-=-=-=- Main Function -=-=-=-=-=-

***********************************************************/

__host__ int bass(float* img, int* seg,spix_params* sp_params,bool* border,
                  spix_helper* sp_helper,spix_helper_sm* sm_helper,
                  int* sm_seg1 ,int* sm_seg2, int* sm_pairs,
                  int niters, int niters_seg, int sm_start,
                  float sigma2_app,  float sigma2_size, int sp_size,
                  float potts, float alpha_hastings,
                  int nspix, int nbatch, int width, int height, int nftrs){

    // -- init --
    int count = 1;
    int npix = height * width;
    int nspix_buffer = nspix * 45;
    int max_spix = nspix;
    // float sigma2_app = sigma_app*sigma_app;
    // fprintf(stdout,"pix_var: %3.5f\n",pix_var);

    // -- Update Parameters --
    // set_border(seg, border, height, width);
    // update_params(img, seg, sp_params, sp_helper, sigma2_app,
    //               npix, sp_size, nspix_buffer, nbatch, width, nftrs);

    // -- Update Parameters --
    update_params(img, seg, sp_params, sp_helper, sigma2_app,
                  npix, sp_size, nspix_buffer, nbatch, width, nftrs);

    // -- Update Segmentation --
    update_seg(img, seg, border, sp_params,
               niters_seg, sigma2_app, potts,
               npix, nbatch, width, height, nftrs);

    for (int idx = 0; idx < niters; idx++) {


      // -- Update Parameters --
      update_params(img, seg, sp_params, sp_helper, sigma2_app,
                    npix, sp_size, nspix_buffer, nbatch, width, nftrs);

      // // -- Update Segmentation --
      // update_seg(img, seg, border, sp_params,
      //            niters_seg, sigma2_app, potts,
      //            npix, nbatch, width, height, nftrs);


      // -- Run Split/Merge --
      if (idx >= sm_start){
        if(idx%4 == 0){
          // count += 1;
          max_spix = run_split(img, seg, border, sp_params,
                               sp_helper, sm_helper, sm_seg1, sm_seg2, sm_pairs,
                               alpha_hastings, sigma2_app, sigma2_size, count,
                               idx, max_spix,sp_size,npix,nbatch,width,
                               height,nftrs,nspix_buffer);

          // -- Update Parameters --
          update_params(img, seg, sp_params, sp_helper, sigma2_app,
                        npix, sp_size, nspix_buffer, nbatch, width, nftrs);

        }
        if( idx%4 == 2){
          run_merge(img, seg, border, sp_params,
                    sp_helper, sm_helper, sm_seg1, sm_seg2, sm_pairs,
                    alpha_hastings, sigma2_app, sigma2_size, count, idx,
                    max_spix,sp_size,npix,nbatch,width,height,nftrs,nspix_buffer);

          // -- Update Parameters --
          update_params(img, seg, sp_params, sp_helper, sigma2_app,
                        npix, sp_size, nspix_buffer, nbatch, width, nftrs);


        }
      }

      // -- Update Segmentation --
      update_seg(img, seg, border, sp_params,
                 niters_seg, sigma2_app, potts,
                 npix, nbatch, width, height, nftrs);


      // -- dev only [ DELETE ME! ] --
      // auto options_i32 = torch::TensorOptions().dtype(torch::kInt32)
      //   .layout(torch::kStrided).device("cuda");
      // auto seg_th = at::from_blob(seg,{height,width},options_i32);
      // auto unique_ids = std::get<0>(at::_unique(seg_th));
      // auto ids = unique_ids.data<int>();
      // int num_ids = unique_ids.sizes()[0];
      // printf("num_ids: %d\n",num_ids);


    }

    CudaFindBorderPixels_end(seg, border, npix, nbatch, width, height);
    return max_spix;

}

__global__
void _view_prior_counts_kernel(spix_params* sp_params, int* ids, int nactive) {
    // -- filling superpixel params into image --
    int ix = threadIdx.x + blockIdx.x * blockDim.x;
    if (ix >= nactive) return;
    int spix_id = ids[ix];
    printf("[%d]: %f\n",spix_id,sp_params[spix_id].prior_count);
}


/**********************************************************

             -=-=-=-=- Python API  -=-=-=-=-=-

***********************************************************/

std::tuple<torch::Tensor,PySuperpixelParams>
run_bass(const torch::Tensor img, int niters,
         int niters_seg, int sm_start, int sp_size,
         float sigma2_app, float sigma2_size,
         float potts, float alpha_hastings){

    // -- check --
    CHECK_INPUT(img);

    // -- unpack --
    int nbatch = img.size(0);
    int height = img.size(1);
    int width = img.size(2);
    int nftrs = img.size(3);
    int npix = height*width;
    assert(nbatch==1);    

    // -- allocate filled spix --
    auto options_i32 = torch::TensorOptions().dtype(torch::kInt32)
      .layout(torch::kStrided).device(img.device());
    torch::Tensor spix = torch::zeros({nbatch,height,width},options_i32);

    // -- init superpixels --
    int nspix = init_seg(spix.data<int>(),sp_size,width,height,nbatch);

    // -- get min,max --
    int min_seg = at::min(spix).item<int>(); 
    int max_seg = at::max(spix).item<int>(); 
    auto _unique_ids = std::get<0>(at::_unique(spix));
    int nuniq = _unique_ids.size(0);
    // int nspix = nuniq;
    fprintf(stdout,"min_seg,max_seg,nuniq: %d,%d,%d\n",min_seg,max_seg,nuniq);

    // -- allocate memory --
    int nspix_buffer = nspix*50;
    const int sparam_size = sizeof(spix_params);
    const int helper_size = sizeof(spix_helper);
    bool* border = (bool*)easy_allocate(nbatch*npix,sizeof(bool));
    spix_params* sp_params=(spix_params*)easy_allocate(nspix_buffer,sparam_size);
    spix_helper* sp_helper=(spix_helper*)easy_allocate(nspix_buffer,helper_size);

    // -- compute pixel (inverse) covariance info --
    // float prior_sigma_app = float(sigma2_app/2) * float(sigma2_app/2);
    // float prior_sigma_app = sigma2_app;
    // float sigma2_app = 4*std::sqrt(1./pix_var.x);
    // float sigma2_app = std::sqrt(1./pix_var.x);
    // float sigma2_size = 0.1;

    // -- convert image color --
    // auto img_lab = img_rgb.clone();
    // if (run_rgb2lab){
    //   rgb2lab(img_rgb.data<float>(),img_lab.data<float>(),npix,nbatch);
    // }else{
    //   img_lab._copy(img_rgb);
    // }

    // -- Get pointers --
    float* img_ptr = img.data<float>();
    int* spix_ptr = spix.data<int>();

    // -- allocate larger memory for prior map --
    int* sm_seg1 = (int*)easy_allocate(npix,sizeof(int));
    int* sm_seg2 = (int*)easy_allocate(npix,sizeof(int));
    int* sm_pairs = (int*)easy_allocate(2*npix,sizeof(int));
    const int sm_helper_size = sizeof(spix_helper_sm);
    spix_helper_sm* sm_helper=(spix_helper_sm*)easy_allocate(nspix_buffer,sm_helper_size);

    // -=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-
    //
    //                 Run BASS
    //
    // -=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-

    // -- init spix_params --
    init_sp_params(sp_params,sigma2_app,img_ptr,spix_ptr,sp_helper,
                   npix,nspix,nspix_buffer,nbatch,width,nftrs,sp_size);
    mark_active_contiguous(sp_params,nspix,nspix_buffer,sp_size);
    //                  int npix, int nspix_buffer,ftrs);
    //              int nbatch, int width, int nftrs)
    // init_sp_params(sp_paramsimg,,nspix,nspix_buffer,npix);

    // -- run method --
    int max_spix = bass(img_ptr, spix_ptr, sp_params,
                        border, sp_helper, sm_helper, sm_seg1, sm_seg2, sm_pairs,
                        niters, niters_seg, sm_start, sigma2_app, sigma2_size,
                        sp_size, potts, alpha_hastings, nspix,
                        nbatch, width, height, nftrs);
    fprintf(stdout,"[before] max_spix: %d\n",max_spix);

    // -- view --
    // auto _unique_ids = std::get<0>(at::_unique(spix));
    // int nactive = _unique_ids.size(0);
    // auto _ids = _unique_ids.data<int>();
    // int _num_blocks = ceil( double(nactive) / double(THREADS_PER_BLOCK) ); 
    // dim3 _nblocks(_num_blocks);
    // dim3 _nthreads(THREADS_PER_BLOCK);
    // _view_prior_counts_kernel<<<_nblocks,_nthreads>>>(sp_params, _ids, nactive);

    // -- only keep superpixels which are alive --
    nspix = compactify_new_superpixels(spix,sp_params,0,max_spix,npix);

    // -- get spixel parameters as tensors --
    auto unique_ids = std::get<0>(at::_unique(spix));
    auto ids = unique_ids.data<int>();
    int num_ids = unique_ids.sizes()[0];

    // PySuperpixelParams params;
    PySuperpixelParams params = get_params_as_tensors(sp_params,ids,num_ids,nspix);
    run_update_prior(spix,params,0,false); // shift estimates to prior information @ spix

    // -- view --
    // _unique_ids = std::get<0>(at::_unique(spix));
    // nactive = _unique_ids.size(0);
    // _ids = _unique_ids.data<int>();
    // _num_blocks = ceil( double(nactive) / double(THREADS_PER_BLOCK) ); 
    // dim3 _nblocks0(_num_blocks);
    // dim3 _nthreads0(THREADS_PER_BLOCK);
    // _view_prior_counts_kernel<<<_nblocks0,_nthreads0>>>(sp_params, _ids, nactive);

    // PySuperpixelParams params = get_params_as_tensors(sp_params,ids,id_order,
    //                                                   nspix_post,nspix_post);
    // PySuperpixelParams params = get_params_as_tensors(sp_params,ids,nspix_post);

    // -- relabel spix --
    // int num_blocks1 = ceil( double(npix) / double(THREADS_PER_BLOCK) ); 
    // dim3 nthreads1(THREADS_PER_BLOCK);
    // dim3 nblocks1(num_blocks1);
    // relabel_spix<false><<<nblocks1,nthreads1>>>(spix.data<int>(),
    //                                             unique_ids.data<int>(),
    //                                             npix, num_ids);



    // -- free --
    cudaFree(sm_helper);
    cudaFree(sm_pairs);
    cudaFree(sm_seg2);
    cudaFree(sm_seg1);
    cudaFree(sp_helper);
    cudaFree(sp_params);
    cudaFree(border);


    return std::make_tuple(spix,params);
}

void init_bass(py::module &m){
  m.def("bass", &run_bass,"run bass");
}

