
/********************************************************************

      Run BASS using the propograted superpixel segs and params

********************************************************************/

// -- cpp imports --
#include <stdio.h>
#include <assert.h>

// -- thrust --
#include <thrust/device_vector.h>
#include <thrust/extrema.h>
#include <thrust/unique.h>
#include <thrust/sort.h>

// -- "external" import --
#include "structs.h"

// -- utils --
// #include "rgb2lab.h"
// #include "sparams_io.h"
#include "seg_utils.h"
#include "init_utils.h"
#include "init_seg.h"
#include "init_sparams.h"
#include "compact_spix.h"
#include "sparams_io.h"

// -- primary functions --
#include "split_merge_orig.h"
#include "update_params.h"
#include "update_seg.h"
// #include "compact_spix.h" // only for controllable bass

#define THREADS_PER_BLOCK 512


/**********************************************************

             -=-=-=-=- Main Function -=-=-=-=-=-

***********************************************************/

__host__ int bass(float* img, int* seg,spix_params* sp_params,bool* border,
                  spix_helper* sp_helper,spix_helper_sm* sm_helper,
                  int* sm_seg1 ,int* sm_seg2, int* sm_pairs,
                  int niters, int niters_seg, int sm_start,
                  float sigma2_app,  float sigma2_size, int sp_size,
                  float potts, float alpha_hastings, float split_alpha, int nspix,
                  int nspix_buffer, int nbatch, int width, int height, int nftrs,
                  int target_nspix){

    // // -- init --
    int count = 1;
    int npix = height * width;
    int max_spix = nspix-1;
    float merge_alpha = 0.0; // only for controlling # of spix

    // printf(".\n");
    // std::cout << "height, width: " << height << ", " << width << std::endl;
    // float ntarget_nspix = npix / (1.0 * sp_size * sp_size);
    int og_niters = niters;
    bool nspix_controlled = target_nspix>0;
    if (nspix_controlled){
      std::cout << "target_nspix: " << target_nspix << std::endl;
      niters = 5000;
    }

    for (int idx = 0; idx < niters; idx++) {


        // gpuErrchk( cudaPeekAtLastError() );
        // gpuErrchk( cudaDeviceSynchronize() );

      // -- control split/merge to yield a fixed # of spix --
      if (nspix_controlled){
        if ((idx % 2) == 0){
          thrust::device_vector<int> prop_ids = extract_unique_ids(seg, npix, 0);
          int nliving = prop_ids.size();
          if (nliving > 1.05*target_nspix){ 
            if (split_alpha > 0){
              split_alpha = 0;
              merge_alpha = 0;
            } // reset
            split_alpha += -1;
            merge_alpha += 1.0;
          }else if (nliving < 0.95*target_nspix){
            if (split_alpha < 0){
              split_alpha = 0;
              merge_alpha = 0;
            } // reset
            split_alpha += 1;
            merge_alpha += -1;
          }else{
            split_alpha = 0.;
            merge_alpha = 0.;
          }
          bool accept_cond = (nliving < 1.05*target_nspix);
          accept_cond = accept_cond and (nliving > 0.95*target_nspix);
          accept_cond = accept_cond and (idx >= og_niters);
          if (accept_cond){ break; }
        }
      }

      // -- Update Parameters --
      update_params(img, seg, sp_params, sp_helper, sigma2_app,
                    npix, sp_size, nspix_buffer, nbatch, width, nftrs);

      //   // gpuErrchk( cudaDeviceSynchronize() );
        // gpuErrchk( cudaPeekAtLastError() );
        // gpuErrchk( cudaDeviceSynchronize() );


      // -- Run Split/Merge --
      if (idx > sm_start){
        if(idx%4 == 0){
          // count = 2;
          max_spix = run_split_orig(img, seg, border, sp_params,
                               sp_helper, sm_helper, sm_seg1, sm_seg2, sm_pairs,
                               alpha_hastings, split_alpha,
                               sigma2_app, sigma2_size, count,
                               idx, max_spix,sp_size,npix,nbatch,width,
                               height,nftrs,nspix_buffer);
          // exit(1);
          // gpuErrchk( cudaPeekAtLastError() );
          // gpuErrchk( cudaDeviceSynchronize() );

          // -- Update Parameters --
          update_params(img, seg, sp_params, sp_helper, sigma2_app,
                        npix, sp_size, nspix_buffer, nbatch, width, nftrs);

        }
        if( idx%4 == 2){

          run_merge_orig(img, seg, border, sp_params,
                    sp_helper, sm_helper, sm_seg1, sm_seg2, sm_pairs,
                    alpha_hastings, merge_alpha, sigma2_app, sigma2_size, count, idx,
                    max_spix,sp_size,npix,nbatch,width,height,nftrs,nspix_buffer);
          // exit(1);
          // gpuErrchk( cudaPeekAtLastError() );
          // gpuErrchk( cudaDeviceSynchronize() );


          // -- Update Parameters --
          update_params(img, seg, sp_params, sp_helper, sigma2_app,
                        npix, sp_size, nspix_buffer, nbatch, width, nftrs);

        }
      }


      // -- Update Segmentation --
      update_seg(img, seg, border, sp_params,
                 niters_seg, sigma2_app, potts,
                 npix, nbatch, width, height, nftrs);

        // gpuErrchk( cudaPeekAtLastError() );
        // gpuErrchk( cudaDeviceSynchronize() );



    }

    update_params(img, seg, sp_params, sp_helper, sigma2_app,
                  npix, sp_size, nspix_buffer, nbatch, width, nftrs);
    store_sample_sigma_shape(sp_params,sp_helper,sp_size, nspix_buffer);

    CudaFindBorderPixels_end(seg, border, npix, nbatch, width, height);

    
    if (nspix_controlled){
      thrust::device_vector<int> prop_ids = extract_unique_ids(seg, npix, 0);
      int nliving = prop_ids.size(); // only when controlling # spix
      printf("nliving: %d\n",nliving);
    }

    return max_spix;

}

__global__
void _view_prior_counts_kernel(spix_params* sp_params, int* ids, int nactive) {
    // -- filling superpixel params into image --
    int ix = threadIdx.x + blockIdx.x * blockDim.x;
    if (ix >= nactive) return;
    int spix_id = ids[ix];
    float3 mu_app = sp_params[spix_id].mu_app;
    double2 mu_shape = sp_params[spix_id].mu_shape;
    double3 sigma_shape = sp_params[spix_id].sigma_shape;
    printf("[%d]: [%2.2f,%2.2f,%2.2f] [%2.2lf,%2.2lf] [%2.2lf,%2.2lf,%2.2lf]\n",
           spix_id,mu_app.x,mu_app.y,mu_app.z,mu_shape.x,mu_shape.y,
           sigma_shape.x,sigma_shape.y,sigma_shape.z);
}

void print_min_max(int* _spix, int npix){

    // -- init superpixels --
    thrust::device_ptr<int> _spix_ptr = thrust::device_pointer_cast(_spix);
    thrust::device_vector<int> spix(_spix_ptr, _spix_ptr + npix);

    auto min_iter = thrust::min_element(spix.begin(), spix.end());
    auto max_iter = thrust::max_element(spix.begin(), spix.end());
    int min_seg = *min_iter;
    int max_seg = *max_iter;
    std::cout << "Minimum element: " << min_seg << std::endl;
    std::cout << "Maximum element: " << max_seg << std::endl;

}


/**********************************************************

             -=-=-=-=- Python API  -=-=-=-=-=-

***********************************************************/

std::tuple<int*,bool*,SuperpixelParams*>
// std::tuple<int*,bool*>
run_bass(float* img, int nbatch, int height, int width, int nftrs,
         int niters, int niters_seg, int sm_start, int sp_size,
         float sigma2_app, float sigma2_size, float potts,
         float alpha_hastings, float split_alpha, int target_nspix){


    // -- unpack --
    int npix = height*width;
    assert(nbatch==1);    

    
    // -- allocate filled spix --
    int* _spix = (int*)easy_allocate(nbatch*npix,sizeof(int));
    thrust::device_ptr<int> _spix_ptr = thrust::device_pointer_cast(_spix);
    thrust::device_vector<int> spix(_spix_ptr, _spix_ptr + npix);

    // -- init superpixels --
    int nspix = init_seg(_spix,sp_size,width,height,nbatch);
    // printf("nspix: %d\n",nspix);

    // -- get min,max --
    // print_min_max(_spix, npix);

    // -- allocate memory --
    int nspix_buffer = nspix*30;
    const int sparam_size = sizeof(spix_params);
    const int helper_size = sizeof(spix_helper);
    bool* border = (bool*)easy_allocate(nbatch*npix,sizeof(bool));
    spix_params* sp_params=(spix_params*)easy_allocate(nspix_buffer,sparam_size);
    spix_helper* sp_helper=(spix_helper*)easy_allocate(nspix_buffer,helper_size);

    // -- INFO --
    thrust::device_vector<int> prop_ids0 = extract_unique_ids(_spix, npix, 0);
    nspix = compactify_new_superpixels(_spix,sp_params,prop_ids0,0,nspix,npix);
    // print_min_max(_spix, npix);
    // gpuErrchk( cudaPeekAtLastError() );
    // gpuErrchk( cudaDeviceSynchronize() );

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

    // gpuErrchk( cudaPeekAtLastError() );
    // gpuErrchk( cudaDeviceSynchronize() );
    // printf("hey\n");

    // -- init spix_params --
    mark_active_contiguous(sp_params,nspix,nspix_buffer,sp_size);
    // gpuErrchk( cudaPeekAtLastError() );
    // gpuErrchk( cudaDeviceSynchronize() );

    init_sp_params(sp_params,sigma2_app,img,_spix,sp_helper,
                   npix,nspix,nspix_buffer,nbatch,width,nftrs,sp_size);
    // gpuErrchk( cudaPeekAtLastError() );
    // gpuErrchk( cudaDeviceSynchronize() );

    CudaFindBorderPixels(_spix,border,npix,nbatch,width,height);

    // gpuErrchk( cudaPeekAtLastError() );
    // gpuErrchk( cudaDeviceSynchronize() );
    // printf("yay.\n");

    // -- run method --
    int max_spix = bass(img, _spix, sp_params,
                        border, sp_helper, sm_helper, sm_seg1, sm_seg2, sm_pairs,
                        niters, niters_seg, sm_start, sigma2_app, sigma2_size,
                        sp_size, potts, alpha_hastings, split_alpha, nspix, nspix_buffer,
                        nbatch, width, height, nftrs, target_nspix);
    // print_min_max(_spix, npix);

    // int max_spix = nspix-1;
    // // fprintf(stdout,"[before] max_spix: %d\n",max_spix);

    // // -- view --
    // thrust::device_vector<int> uniq_spix(_spix_ptr, _spix_ptr + npix);
    // thrust::sort(uniq_spix.begin(),uniq_spix.end());
    // auto uniq_end = thrust::unique(uniq_spix.begin(),uniq_spix.end());
    // uniq_spix.erase(uniq_end, uniq_spix.end());
    // uniq_spix.resize(uniq_end - uniq_spix.begin());
    // printf("delta: %d\n",uniq_end - uniq_spix.begin());
    // int nactive = uniq_spix.size();
    // int* _uniq_spix = thrust::raw_pointer_cast(uniq_spix.data());
    // printf("nactive: %d\n",nactive);
    // int _num_blocks = ceil( double(nactive) / double(THREADS_PER_BLOCK) ); 
    // dim3 _nblocks(_num_blocks);
    // dim3 _nthreads(THREADS_PER_BLOCK);
    // _view_prior_counts_kernel<<<_nblocks,_nthreads>>>(sp_params, _uniq_spix, nactive);

    // -- only keep superpixels which are alive --
    thrust::device_vector<int> prop_ids = extract_unique_ids(_spix, npix, 0);
    nspix = compactify_new_superpixels(_spix,sp_params,prop_ids,0,max_spix,npix);
    // print_min_max(_spix, npix);

    // -- get spixel parameters as tensors --
    thrust::copy(_spix_ptr,_spix_ptr+npix,spix.begin());
    thrust::device_vector<int> uniq_ids = get_unique(_spix,npix);
    int num_ids = uniq_ids.size();
    int* _uniq_ids = thrust::raw_pointer_cast(uniq_ids.data());
    SuperpixelParams* params = get_params_as_vectors(sp_params,_uniq_ids,num_ids,nspix);
    run_update_prior(params,_uniq_ids, npix, nspix, 0,false);
    // run_update_prior(params,_uniq_ids, npix, nspix, nspix_prev,false);
    CudaFindBorderPixels_end(_spix,border,npix,nbatch,width,height);


    // -- free --
    cudaFree(sm_helper);
    cudaFree(sm_pairs);
    cudaFree(sm_seg2);
    cudaFree(sm_seg1);
    cudaFree(sp_helper);
    cudaFree(sp_params);
    // cudaFree(border);

    // -- return! --
    return std::make_tuple(_spix,border,params);
    // return std::make_tuple(_spix,border);
}


