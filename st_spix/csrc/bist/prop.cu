/********************************************************************

      Run BASS using the propograted superpixel segs and params

********************************************************************/

// -- cpp imports --
#include <stdio.h>
#include <assert.h>
#include <tuple>

// -- thrust --
#include <thrust/host_vector.h>
#include <thrust/device_vector.h>
#include <thrust/extrema.h>
#include <thrust/unique.h>
#include <thrust/sort.h>
#include <thrust/host_vector.h> // for debugging

// -- "external" import --
#include "structs.h"

// -- utils --
// #include "rgb2lab.h"
// #include "sparams_io.h"
// #include "demo_utils.h"
#include "seg_utils.h"
#include "init_utils.h"
#include "init_seg.h"
#include "init_sparams.h"
#include "compact_spix.h"
#include "sparams_io.h"
#include "relabel.h"

// -- primary functions --
// #include "split_merge.h"
#include "split_merge_prop.h"
// #include "split_merge_prop_v2.h"
#include "update_params.h"
#include "update_seg.h"

#define THREADS_PER_BLOCK 512


/**********************************************************

             -=-=-=-=- Main Function -=-=-=-=-=-

***********************************************************/

// __host__ int bass(float* img, int* seg,spix_params* sp_params,bool* border,
//                   spix_helper* sp_helper,spix_helper_sm* sm_helper,
//                   int* sm_seg1 ,int* sm_seg2, int* sm_pairs,
//                   int niters, int niters_seg, int sm_start,
//                   float sigma2_app,  float sigma2_size, int sp_size,
//                   float potts, float alpha_hastings, int nspix,
//                   int nspix_buffer, int nbatch, int width, int height, int nftrs){

//     // // -- init --
//     int count = 1;
//     int npix = height * width;
//     // // int nspix_buffer = nspix * 45;
//     int max_spix = nspix;

//     std::cout << "height, width: " << height << ", " << width << std::endl;
    
//     for (int idx = 0; idx < niters; idx++) {


//         gpuErrchk( cudaPeekAtLastError() );
//         gpuErrchk( cudaDeviceSynchronize() );

//       // -- Update Parameters --
//       update_params(img, seg, sp_params, sp_helper, sigma2_app,
//                     npix, sp_size, nspix_buffer, nbatch, width, nftrs);

//       //   // gpuErrchk( cudaDeviceSynchronize() );
//         gpuErrchk( cudaPeekAtLastError() );
//         gpuErrchk( cudaDeviceSynchronize() );


//       // -- Run Split/Merge --
//       if (idx > sm_start){
//         if(idx%4 == 0){
//           // count = 2;
//           max_spix = run_split_orig(img, seg, border, sp_params,
//                                sp_helper, sm_helper, sm_seg1, sm_seg2, sm_pairs,
//                                alpha_hastings, sigma2_app, sigma2_size, count,
//                                idx, max_spix,sp_size,npix,nbatch,width,
//                                height,nftrs,nspix_buffer);
//           // exit(1);
//           gpuErrchk( cudaPeekAtLastError() );
//           gpuErrchk( cudaDeviceSynchronize() );

//           // -- Update Parameters --
//           update_params(img, seg, sp_params, sp_helper, sigma2_app,
//                         npix, sp_size, nspix_buffer, nbatch, width, nftrs);

//         }
//         if( idx%4 == 2){
//           run_merge_orig(img, seg, border, sp_params,
//                     sp_helper, sm_helper, sm_seg1, sm_seg2, sm_pairs,
//                     alpha_hastings, sigma2_app, sigma2_size, count, idx,
//                     max_spix,sp_size,npix,nbatch,width,height,nftrs,nspix_buffer);
//           // exit(1);
//           gpuErrchk( cudaPeekAtLastError() );
//           gpuErrchk( cudaDeviceSynchronize() );


//           // -- Update Parameters --
//           update_params(img, seg, sp_params, sp_helper, sigma2_app,
//                         npix, sp_size, nspix_buffer, nbatch, width, nftrs);

//         }
//       }


//       // -- Update Segmentation --
//       update_seg(img, seg, border, sp_params,
//                  niters_seg, sigma2_app, potts,
//                  npix, nbatch, width, height, nftrs);

//         gpuErrchk( cudaPeekAtLastError() );
//         gpuErrchk( cudaDeviceSynchronize() );



//     }

//     CudaFindBorderPixels_end(seg, border, npix, nbatch, width, height);
//     return max_spix;

// }

// __global__
// void _view_prior_counts_kernel(spix_params* sp_params, int* ids, int nactive) {
//     // -- filling superpixel params into image --
//     int ix = threadIdx.x + blockIdx.x * blockDim.x;
//     if (ix >= nactive) return;
//     int spix_id = ids[ix];
//     float3 mu_app = sp_params[spix_id].mu_app;
//     double2 mu_shape = sp_params[spix_id].mu_shape;
//     double3 sigma_shape = sp_params[spix_id].sigma_shape;
//     printf("[%d]: [%2.2f,%2.2f,%2.2f] [%2.2lf,%2.2lf] [%2.2lf,%2.2lf,%2.2lf]\n",
//            spix_id,mu_app.x,mu_app.y,mu_app.z,mu_shape.x,mu_shape.y,
//            sigma_shape.x,sigma_shape.y,sigma_shape.z);
// }

std::tuple<int,int> _get_min_max(int* _spix, int npix){
    // -- init superpixels --
    thrust::device_ptr<int> _spix_ptr = thrust::device_pointer_cast(_spix);
    thrust::device_vector<int> spix(_spix_ptr, _spix_ptr + npix);

    auto min_iter = thrust::min_element(spix.begin(), spix.end());
    auto max_iter = thrust::max_element(spix.begin(), spix.end());
    int min_seg = *min_iter;
    int max_seg = *max_iter;
    return std::make_tuple(min_seg,max_seg);
}
void _print_min_max(int* _spix, int npix){
    int min_seg, max_seg;
    std::tie(min_seg, max_seg) = _get_min_max(_spix,npix);
    std::cout << "Minimum element: " << min_seg << std::endl;
    std::cout << "Maximum element: " << max_seg << std::endl;
}


__host__ int prop_bass_v2(float* img, int* seg, int* shifted,
                          spix_params* sp_params,
                          bool* border,spix_helper* sp_helper,
                          spix_helper_sm_v2* sm_helper,
                          // spix_helper_sm* sm_helper,
                          int* sm_seg1 ,int* sm_seg2, int* sm_pairs,
                          int niters, int niters_seg, int sm_start,
                          float sigma2_app,  float sigma2_size, int sp_size,
                          float potts, float alpha_hastings,
                          int nspix, int nspix_buffer,
                          int nbatch, int width, int height, int nftrs,
                          SuperpixelParams* params_prev, int nspix_prev,
                          float thresh_relabel, float thresh_new, float merge_offset){


    // -- init --
    int count = 1;
    int npix = height * width;
    int max_spix = nspix-1;
    std::cout << "height, width: " << height << ", " << width << std::endl;
    
    for (int idx = 0; idx < niters; idx++) {


      // -- Update Parameters --
      update_params(img, seg, sp_params, sp_helper, sigma2_app,
                    npix, sp_size, nspix_buffer, nbatch, width, nftrs);

      // // -- [dev only] --
      // cv::String fname = "debug_v2_seg.csv";
      // save_spix_gpu(fname, seg, height, width);

      // -- Run Split/Merge --
      if (idx >= sm_start){
        if((idx%4 == 0) and (idx==0)){
          for (int zidx = 0; zidx<2; zidx++){
            count = rand() % 2; // random direction; seemingly little effect 
            // count = 2;
            max_spix = run_split_p(img, seg, shifted, border, sp_params,
                                   sp_helper, sm_helper, sm_seg1, sm_seg2, sm_pairs,
                                   alpha_hastings, alpha_hastings,
                                   sigma2_app, sigma2_size, count,
                                   idx, max_spix,sp_size,npix,nbatch,width,
                                   height,nftrs,nspix_buffer);
            // exit(1);
            update_params(img, seg, sp_params, sp_helper, sigma2_app,
                          npix, sp_size, nspix_buffer, nbatch, width, nftrs);


            max_spix = run_split_p(img, seg, shifted, border, sp_params,
                                   sp_helper, sm_helper, sm_seg1, sm_seg2, sm_pairs,
                                   alpha_hastings, alpha_hastings,
                                   sigma2_app, sigma2_size, count,
                                   idx, max_spix,sp_size,npix,nbatch,width,
                                   height,nftrs,nspix_buffer);

            update_params(img, seg, sp_params, sp_helper, sigma2_app,
                          npix, sp_size, nspix_buffer, nbatch, width, nftrs);
          }

        }

        if( idx == 6){
        // if( idx == -1){


          /*******************************************
             RELABEL different/new spix
          ********************************************/
          thrust::device_vector<int> prop_ids = extract_unique_ids(seg, npix, 0);
          max_spix = relabel_spix(seg,sp_params,params_prev,prop_ids,
                                  thresh_relabel,thresh_new,
                                  height,width,nspix_prev,max_spix);
          update_params(img, seg, sp_params, sp_helper, sigma2_app,
                        npix, sp_size, nspix_buffer, nbatch, width, nftrs);

          /*******************************************
             MERGE different/new spix
          ********************************************/
          // printf("[merge] nspix: %d\n",max_spix+1);
          for (int zidx = 0; zidx<2; zidx++){
              int init_count = rand() % 2;
              count = init_count; // random direction; seemingly little effect 
              run_merge_p(img, seg, border, sp_params,
                          sp_helper, sm_helper, sm_seg1, sm_seg2, sm_pairs,
                          merge_offset,alpha_hastings, sigma2_app, sigma2_size,count,idx,
                          max_spix,sp_size,npix,nbatch,width,height,nftrs,nspix_buffer);
    
              update_params(img, seg, sp_params, sp_helper, sigma2_app,
                            npix, sp_size, nspix_buffer, nbatch, width, nftrs);
    
              // count = rand() % 2; // random direction; seemingly little effect 
              run_merge_p(img, seg, border, sp_params,
                        sp_helper, sm_helper, sm_seg1, sm_seg2, sm_pairs,
                          merge_offset,alpha_hastings, sigma2_app, sigma2_size, count, idx,
                        max_spix,sp_size,npix,nbatch,width,height,nftrs,nspix_buffer);
    
              // count = rand() % 2; // random direction; seemingly little effect 
              // run_merge_p(img, seg, border, sp_params,
              //           sp_helper, sm_helper, sm_seg1, sm_seg2, sm_pairs,
              //             merge_offset,alpha_hastings, sigma2_app, sigma2_size, count, idx,
              //           max_spix,sp_size,npix,nbatch,width,height,nftrs,nspix_buffer);
    
              // -- Update Parameters --
              update_params(img, seg, sp_params, sp_helper, sigma2_app,
                            npix, sp_size, nspix_buffer, nbatch, width, nftrs);
          }

          // // -- merge second time --
          // count = init_count+1; // random direction; seemingly little effect 
          // run_merge_p(img, seg, border, sp_params,
          //           sp_helper, sm_helper, sm_seg1, sm_seg2, sm_pairs,
          //             merge_offset,alpha_hastings, sigma2_app, sigma2_size, count, idx,
          //           max_spix,sp_size,npix,nbatch,width,height,nftrs,nspix_buffer);

          // run_merge_p(img, seg, border, sp_params,
          //           sp_helper, sm_helper, sm_seg1, sm_seg2, sm_pairs,
          //             merge_offset,alpha_hastings, sigma2_app, sigma2_size, count, idx,
          //           max_spix,sp_size,npix,nbatch,width,height,nftrs,nspix_buffer);

          // // -- Update Parameters --
          // update_params(img, seg, sp_params, sp_helper, sigma2_app,
          //               npix, sp_size, nspix_buffer, nbatch, width, nftrs);

          // run_merge_p(img, seg, border, sp_params,
          //           sp_helper, sm_helper, sm_seg1, sm_seg2, sm_pairs,
          //           alpha_hastings, sigma2_app, sigma2_size, count, idx,
          //           max_spix,sp_size,npix,nbatch,width,height,nftrs,nspix_buffer);

          // // -- Update Parameters --
          // update_params(img, seg, sp_params, sp_helper, sigma2_app,
          //               npix, sp_size, nspix_buffer, nbatch, width, nftrs);

        }

      }


      // -- Update Segmentation --
      update_seg(img, seg, border, sp_params,
                 niters_seg, sigma2_app, potts,
                 npix, nbatch, width, height, nftrs);

    }


    //-- Update Parameters --
    update_params(img, seg, sp_params, sp_helper, sigma2_app,
                  npix, sp_size, nspix_buffer, nbatch, width, nftrs);
    store_sample_sigma_shape(sp_params,sp_helper,sp_size, nspix_buffer);


    CudaFindBorderPixels_end(seg, border, npix, nbatch, width, height);
    return max_spix;


}

__host__ int prop_bass_v3(float* img, int* seg, int* shifted,
                          spix_params* sp_params,
                          bool* border,spix_helper* sp_helper,
                          spix_helper_sm_v2* sm_helper,
                          // spix_helper_sm* sm_helper,
                          int* sm_seg1 ,int* sm_seg2, int* sm_pairs,
                          int niters, int niters_seg, int sm_start,
                          float sigma2_app,  float sigma2_size, int sp_size,
                          float potts, float alpha_hastings,
                          int nspix, int nspix_buffer,
                          int nbatch, int width, int height, int nftrs,
                          SuperpixelParams* params_prev, int nspix_prev,
                          float thresh_relabel, float thresh_new,
                          float merge_alpha, float split_alpha,
                          int target_nspix){

    // -- init --
    int count = 1;
    int npix = height * width;
    int max_spix = nspix-1;
    
    /*******************************************
             RELABEL different/new spix
    ********************************************/
    // thrust::device_vector<int> prop_ids = extract_unique_ids(seg, npix, 0);
    // max_spix = relabel_spix(seg,sp_params,params_prev,prop_ids,
    //                         thresh_relabel,thresh_new,
    //                         height,width,nspix_prev,max_spix);

    // -- controlled nspix --
    int og_niters = niters;
    bool nspix_controlled = target_nspix>0;
    if (nspix_controlled){
      std::cout << "target_nspix: " << target_nspix << std::endl;
      niters = 5000;
    }


    for (int idx = 0; idx < niters; idx++) {

      // -- Update Parameters --
      update_params(img, seg, sp_params, sp_helper, sigma2_app,
                    npix, sp_size, nspix_buffer, nbatch, width, nftrs);

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
          if ((idx > 2000) and (idx%200 == 0)){ // mark to new if  can't fix # spix
            thresh_new = 2*(thresh_new+1e-5);
          }
          bool accept_cond = (nliving < 1.05*target_nspix);
          accept_cond = accept_cond and (nliving > 0.95*target_nspix);
          accept_cond = accept_cond and (idx >= og_niters);
          if (accept_cond){ break; }
        }
      }

      // -- Run Split/Merge --
      if (idx > sm_start){
        if(idx%4 == 0){
          // count = 2;
          max_spix = run_split_p(img, seg, shifted, border, sp_params,
                                 sp_helper, sm_helper, sm_seg1, sm_seg2, sm_pairs,
                                 alpha_hastings, split_alpha,
                                 sigma2_app, sigma2_size,
                                 count, idx, max_spix,
                                 sp_size,npix,nbatch,width,
                                 height,nftrs,nspix_buffer);

          // exit(1);
          // gpuErrchk( cudaPeekAtLastError() );
          // gpuErrchk( cudaDeviceSynchronize() );

          // -- Update Parameters --
          update_params(img, seg, sp_params, sp_helper, sigma2_app,
                        npix, sp_size, nspix_buffer, nbatch, width, nftrs);

        }
        if( idx%4 == 2){
          thrust::device_vector<int> prop_ids = extract_unique_ids(seg, npix, 0);
          max_spix = relabel_spix(seg,sp_params,params_prev,prop_ids,
                                  thresh_relabel,thresh_new,
                                  height,width,nspix_prev,max_spix);

          run_merge_p(img, seg, border, sp_params,
                      sp_helper, sm_helper, sm_seg1, sm_seg2, sm_pairs,
                      merge_alpha,alpha_hastings, sigma2_app, sigma2_size,count,idx,
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
      
      // gpuErrchk( cudaPeekAtLastError() );
      // gpuErrchk( cudaDeviceSynchronize() );



    }

    update_params(img, seg, sp_params, sp_helper, sigma2_app,
                  npix, sp_size, nspix_buffer, nbatch, width, nftrs);
    store_sample_sigma_shape(sp_params,sp_helper,sp_size, nspix_buffer);

    if (nspix_controlled){
      thrust::device_vector<int> prop_ids = extract_unique_ids(seg, npix, 0);
      int nliving = prop_ids.size(); // only when controlling # spix
      printf("nliving: %d\n",nliving);
    }

    CudaFindBorderPixels_end(seg, border, npix, nbatch, width, height);
    return max_spix;

}


__host__ int prop_bass(float* img, int* seg,spix_params* sp_params,
                       bool* border,spix_helper* sp_helper,
                       spix_helper_sm* sm_helper,
                       int* sm_seg1 ,int* sm_seg2, int* sm_pairs,
                       int niters, int niters_seg, int sm_start,
                       float sigma2_app,  float sigma2_size, int sp_size,
                       float potts, float alpha_hastings,
                       int nspix, int nspix_buffer,
                       int nbatch, int width, int height, int nftrs){

    // -- init --
    int count = 1;
    int npix = height * width;
    // int count = 1;
    // int npix = height * width;
    // int max_spix = nspix;
    int max_spix = nspix-1;
    // float sigma2_app = sigma_app*sigma_app;
    // fprintf(stdout,"pix_var: %3.5f\n",pix_var);

    // printf("a.\n");
    // -- Update Parameters --
    // update_params(img, seg, sp_params, sp_helper, sigma2_app,
    //               npix, sp_size, nspix_buffer, nbatch, width, nftrs);

    // // -- Update Segmentation --
    // update_seg(img, seg, border, sp_params,
    //              niters_seg, sigma2_app, potts,
    //              npix, nbatch, width, height, nftrs);

    // cv::String fname = "init_seg.csv";
    // save_spix_gpu(fname, seg, height, width);

    set_border(seg, border, height, width);

    /*******************************************

             SPLIT only twice; run the rest

    ********************************************/

    for (int idx = 0; idx < niters; idx++) {


      //-- Update Parameters --
      update_params(img, seg, sp_params, sp_helper, sigma2_app,
                    npix, sp_size, nspix_buffer, nbatch, width, nftrs);

      // -- Run Split --
      // printf("idx: %d\n",idx);
      if(true){
      // if ((idx >= sm_start) and (idx <= (sm_start+4))){
      // if ((idx >= sm_start)){
        // if(idx%4 == 0){
        if(idx%2 == 0){
          // count += 1;
          // printf("d0.\n");

          // -- debug only --
          // if(true){
          thrust::device_vector<int> _uniq_ids = get_unique(seg,npix);
          thrust::host_vector<int> uniq_ids = _uniq_ids;
          thrust::device_vector<float> _pc = get_prior_counts(sp_params, nspix);
          thrust::host_vector<float> pcounts = _pc;
          // for(auto id : uniq_ids){ std::cout<< id << ", "; }
          std::cout << "before: " << _uniq_ids.back() << ", " \
                    << uniq_ids.size() << ", " << max_spix << std::endl;
          thrust::device_vector<int> _counts = get_spix_counts(seg,1,npix,max_spix+1);
          thrust::host_vector<int> counts = _counts;
          // for(auto count : counts){if(count > 5000){ std::cout<< count << ", "; } }
          // for(int _i=0;_i<(max_spix+1);_i++){
          //   // if(counts[_i] > 5000){std::cout<< _i << ": "<<counts[_i] << std::endl; }
          //   if(counts[_i] == 0){ continue; }
          //   if(counts[_i] < 200){std::cout<< _i<<": "<<counts[_i] << std::endl; }
          // }
          // }

          // max_spix = run_split_prop_v2(img, seg, border, sp_params,
          //                      sp_helper, sm_helper, sm_seg1, sm_seg2, sm_pairs,
          //                      alpha_hastings, sigma2_app, sigma2_size, count,
          //                      idx, max_spix,sp_size,npix,nbatch,width,
          //                      height,nftrs,nspix_buffer);

          // -- debug only --
          // if(true){
          // thrust::device_vector<int> _uniq_ids1 = get_unique(seg,npix);
          // thrust::host_vector<int> uniq_ids1 = _uniq_ids1;
          // // for(auto id : uniq_ids){ std::cout<< id << ", "; }
          // std::cout << "after: " << _uniq_ids1.back() << ", " \
          //           << uniq_ids1.size() << ", " << max_spix << std::endl;
          // thrust::device_vector<int> _counts1 = get_spix_counts(seg,1,npix,max_spix+1);
          // thrust::host_vector<int> counts1 = _counts1;
          // thrust::device_vector<float> _pc1 = get_prior_counts(sp_params, nspix);
          // thrust::host_vector<float> pcounts1 = _pc1;
          // // for(auto count : counts){ if (count > 5000){std::cout<< count << ", ";}}
          // for(int _i=0;_i<(max_spix+1);_i++){
          //   // if(counts[_i] > 5000){std::cout<< _i << ": " <<counts[_i]<< std::endl; }
          //   if(counts1[_i] == 0){ continue; }
          //   // if(counts1[_i] < 200){std::cout<< _i<<": "<<counts1[_i] << std::endl; }
          //   bool condA = counts[_i] == counts1[_i];
          //   bool condB = counts[_i] > 1312;
          //   // bool condA = counts[_i] > counts1[_i];
          //   // bool condB = counts[_i] < 500;
          //   if (condA and condB){std::cout<< _i<<": "<<counts[_i]<<", "<<\
          //       counts1[_i]<<", "<<pcounts[_i]<< ", " << pcounts1[_i] << std::endl; }
          // }
          // // }


          // -- count invalid --
          // int ninvalid =  count_invalid(seg,npix);
          // printf("ninvalid: %d\n",ninvalid);

          //-- Update Parameters --
          update_params(img, seg, sp_params, sp_helper, sigma2_app,
                        npix, sp_size, nspix_buffer, nbatch, width, nftrs);

          // printf("d1.\n");
          // cudaDeviceSynchronize();
        }
      }


      // -- Update Segmentation --
      update_seg(img, seg, border, sp_params,
                 niters_seg, sigma2_app, potts,
                 npix, nbatch, width, height, nftrs);

    }


    /*******************************************

             MERGE for only one update

    ********************************************/
    for (int idx = 0; idx < 4; idx++) {


      // -- Update Parameters --
      update_params(img, seg, sp_params, sp_helper, sigma2_app,
                    npix, sp_size, nspix_buffer, nbatch, width, nftrs);

      // -- Merge --

      if( idx%4 == 0){
          // printf("e0.\n");
          // run_merge_prop_v2(img, seg, border, sp_params,
          //             sp_helper, sm_helper, sm_seg1, sm_seg2, sm_pairs,
          //             alpha_hastings, sigma2_app, sigma2_size, count, idx,
          //             max_spix,sp_size,npix,nbatch,width,height,nftrs,nspix_buffer);
          // printf("e1.\n");

          //-- Update Parameters --
          update_params(img, seg, sp_params, sp_helper, sigma2_app,
                        npix, sp_size, nspix_buffer, nbatch, width, nftrs);

      }

      // -- Update Segmentation --
      update_seg(img, seg, border, sp_params,
                 niters_seg, sigma2_app, potts,
                 npix, nbatch, width, height, nftrs);

    }

    // -- dev only [ DELETE ME! ] --
    // auto options_i32 = torch::TensorOptions().dtype(torch::kInt32)
    //   .layout(torch::kStrided).device("cuda");
    // auto seg_th = at::from_blob(seg,{height,width},options_i32);
    // auto unique_ids = std::get<0>(at::_unique(seg_th));
    // auto ids = unique_ids.data<int>();
    // int num_ids = unique_ids.sizes()[0];
    // printf("num_ids: %d\n",num_ids);

    // fname = "post_seg.csv";
    // save_spix_gpu(fname, seg, height, width);

    //-- Update Parameters --
    update_params(img, seg, sp_params, sp_helper, sigma2_app,
                  npix, sp_size, nspix_buffer, nbatch, width, nftrs);
    store_sample_sigma_shape(sp_params,sp_helper,sp_size, nspix_buffer);

    CudaFindBorderPixels_end(seg, border, npix, nbatch, width, height);
    return max_spix;

}

int get_max_spix(int* _spix, int npix){
    // -- init superpixels --
    thrust::device_ptr<int> _spix_ptr = thrust::device_pointer_cast(_spix);
    thrust::device_vector<int> spix(_spix_ptr, _spix_ptr + npix);

    // auto min_iter = thrust::min_element(spix.begin(), spix.end());
    auto max_iter = thrust::max_element(spix.begin(), spix.end());
    int max_seg = *max_iter;
    return max_seg;
}

/**********************************************************

             -=-=-=-=- Python API  -=-=-=-=-=-

***********************************************************/

std::tuple<int*,bool*,SuperpixelParams*>
run_prop(float* img, int nbatch, int height, int width, int nftrs,
         int niters, int niters_seg, int sm_start, int sp_size,
         float sigma2_app, float sigma2_size,
         float potts, float alpha_hastings,
         int* spix_prev, int* shifted_spix, SuperpixelParams* params_prev,
         float thresh_relabel, float thresh_new,
         float merge_alpha, float split_alpha, int target_nspix){


    // -- unpack --
    int npix = height*width;
    assert(nbatch==1);    
    int nspix = params_prev->ids.size();
    // printf("[tag!] starting another prop.\n");

    
    // -- allocate filled spix --
    int* _spix = (int*)easy_allocate(nbatch*npix,sizeof(int));
    cudaMemcpy(_spix, spix_prev, npix*sizeof(int), cudaMemcpyDeviceToDevice);
    // thrust::device_ptr<int> _spix_ptr = thrust::device_pointer_cast(_spix);
    // thrust::device_vector<int> spix(_spix_ptr, _spix_ptr + npix);
    int nspix_prev = nspix;
    // printf("nspix_prev: %d\n",nspix_prev);

    // // -- allocate memory --
    int nspix_buffer = nspix*10;
    const int sparam_size = sizeof(spix_params);
    const int helper_size = sizeof(spix_helper);
    bool* border = (bool*)easy_allocate(nbatch*npix,sizeof(bool));
    // spix_params* sp_params=(spix_params*)easy_allocate(nspix_buffer,sparam_size);
    spix_helper* sp_helper=(spix_helper*)easy_allocate(nspix_buffer,helper_size);
    int* sm_seg1 = (int*)easy_allocate(npix,sizeof(int));
    int* sm_seg2 = (int*)easy_allocate(npix,sizeof(int));
    int* sm_pairs = (int*)easy_allocate(2*nspix_buffer,sizeof(int));
    // const int sm_helper_size = sizeof(spix_helper_sm);
    // spix_helper_sm* sm_helper=(spix_helper_sm*)easy_allocate(nspix_buffer,
    //                                                             sm_helper_size);
    const int sm_helper_size = sizeof(spix_helper_sm_v2);
    spix_helper_sm_v2* sm_helper=(spix_helper_sm_v2*)easy_allocate(nspix_buffer,
                                                                sm_helper_size);

    // -=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-
    //
    //             Run Propogated BASS
    //
    // -=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-

    // // -- init spix_params --
    // init_sp_params(sp_params,sigma2_app,img,_spix,sp_helper,
    //                npix,nspix,nspix_buffer,nbatch,width,nftrs,sp_size);
    // gpuErrchk( cudaPeekAtLastError() );
    // gpuErrchk( cudaDeviceSynchronize() );

          // gpuErrchk( cudaPeekAtLastError() );
          // gpuErrchk( cudaDeviceSynchronize() );

    // -- init params --
    spix_params* sp_params = get_vectors_as_params(params_prev,sp_size,
                                                   npix,nspix,nspix_buffer);
    // gpuErrchk( cudaPeekAtLastError() );
    // gpuErrchk( cudaDeviceSynchronize() );

    // -- mark active spix --
    thrust::device_vector<int> init_ids = get_unique(_spix,nbatch*npix);
    int* init_ids_ptr = thrust::raw_pointer_cast(init_ids.data());
    int nactive = init_ids.size();
    mark_active(sp_params, init_ids_ptr, nactive, nspix, nspix_buffer, sp_size);

    // gpuErrchk( cudaPeekAtLastError() );
    // gpuErrchk( cudaDeviceSynchronize() );

    // mark_active_contiguous(sp_params,nspix,nspix_buffer,sp_size);
    // gpuErrchk( cudaPeekAtLastError() );
    // gpuErrchk( cudaDeviceSynchronize() );

    // CudaFindBorderPixels(_spix,border,npix,nbatch,width,height);

    // gpuErrchk( cudaPeekAtLastError() );
    // gpuErrchk( cudaDeviceSynchronize() );

    // printf("before.\n");
    // view_invalid(sp_params,nspix);


    // -- run method --
    // int max_spix = prop_bass_v2(img,_spix,shifted_spix,sp_params,
    //                             border,sp_helper, sm_helper, sm_seg1,sm_seg2,sm_pairs,
    //                             niters, niters_seg, sm_start, sigma2_app, sigma2_size,
    //                             sp_size, potts, alpha_hastings, nspix, nspix_buffer,
    //                             nbatch, width, height, nftrs,
    //                             params_prev,nspix_prev,
    //                             thresh_relabel,thresh_new,merge_alpha);
    int max_spix = prop_bass_v3(img,_spix,shifted_spix,sp_params,
                                border,sp_helper, sm_helper, sm_seg1,sm_seg2,sm_pairs,
                                niters, niters_seg, sm_start, sigma2_app, sigma2_size,
                                sp_size, potts, alpha_hastings, nspix, nspix_buffer,
                                nbatch, width, height, nftrs,
                                params_prev,nspix_prev,
                                thresh_relabel,thresh_new,
                                merge_alpha,split_alpha,target_nspix);
    // printf("[after] max_spix: %d\n",max_spix);
    int prev_max_spix = max_spix;
    // printf("after.\n");
    // view_invalid(sp_params,nspix);

    // int max_spix = bass(img, _spix, sp_params,
    //                     border, sp_helper, sm_helper, sm_seg1, sm_seg2, sm_pairs,
    //                     niters, niters_seg, sm_start, sigma2_app, sigma2_size,
    //                     sp_size, potts, alpha_hastings, nspix, nspix_buffer,
    //                     nbatch, width, height, nftrs);
    // _print_min_max(_spix, npix);
    // // int max_spix = nspix-1;

    // gpuErrchk( cudaPeekAtLastError() );
    // gpuErrchk( cudaDeviceSynchronize() );
    // printf("0.\n");

    // -- get unique ids --
    // int* prop_ids_ptr = thrust::raw_pointer_cast(prop_ids.data());
    // gpuErrchk( cudaPeekAtLastError() );
    // gpuErrchk( cudaDeviceSynchronize() );
    // _print_min_max(prop_ids_ptr, prop_ids.size());//debug
    // int min_id = *thrust::min_element(prop_ids.begin(), prop_ids.end());
    // int max_id = *thrust::max_element(prop_ids.begin(), prop_ids.end());
    // std::cout << "Max/Min element: " << max_id << ", " << min_id << std::endl;
    // printf("1.\n");


    // -- relabel --
    thrust::device_vector<int> prop_ids = extract_unique_ids(_spix, npix, 0);
    // max_spix = relabel_spix(_spix,sp_params,params_prev,prop_ids,
    //                         thresh_relabel,thresh_new,
    //                         height,width,nspix_prev,max_spix);
    // update_params(img, _spix, sp_params, sp_helper, sigma2_app,
    //               npix, sp_size, nspix_buffer, nbatch, width, nftrs);
    int nalive = prop_ids.size();
    // printf("nliving: %d\n",nalive);
    // // printf("[after relabel_spix] max_spix: %d\n",max_spix);


    // printf("[after] nalive: %d\n",nalive);
    // int _min_spix, _max_spix;
    // std::tie(_min_spix, _max_spix) = _get_min_max(_spix,npix);
    // assert(_max_spix == max_spix);
    // printf("nalive,_min_spix,_max_spix,max_spix,prev_max_spix,nspix_prev: %d,%d,%d,%d,%d,%d\n",nalive,_min_spix,_max_spix,max_spix,prev_max_spix,nspix_prev);
    // _print_min_max(_spix, npix);
    // printf("2.\n");

    // -- only keep superpixels which are alive --
    // thrust::device_vector<int> new_ids = extract_unique_ids(_spix, npix, nspix_prev);
    thrust::device_vector<int> new_ids = remove_old_ids(prop_ids,nspix_prev);
    nspix = compactify_new_superpixels(_spix,sp_params,new_ids,
                                       nspix_prev,max_spix,npix);
    // printf("final nspix: %d\n",nspix);
    // exit(1);
    // print_min_max(_spix, npix);
    // printf("3.\n");

    // gpuErrchk( cudaPeekAtLastError() );
    // gpuErrchk( cudaDeviceSynchronize() );

    // -- get spixel parameters as tensors --
    // thrust::copy(_spix_ptr,_spix_ptr+npix,spix.begin());
    thrust::device_vector<int> uniq_ids = get_unique(_spix,npix);
    thrust::host_vector<int> uniq_ids_h = uniq_ids;
    // int num_ids = uniq_ids.size();
    int num_ids = nspix;
    int* _uniq_ids = thrust::raw_pointer_cast(uniq_ids.data());
    SuperpixelParams* params = get_params_as_vectors(sp_params,_uniq_ids,num_ids,nspix);
    run_update_prior(params,_uniq_ids, npix, nspix, nspix_prev,false);
    CudaFindBorderPixels_end(_spix,border,npix,nbatch,width,height);

    // -- [dev only] check nspix v.s. max --
    // printf("[@end] nspix,uniq_ids.back(),params->ids.size(): %d,%d,%d\n",
    //        nspix,uniq_ids_h.back(),params->ids.size());
    assert((nspix-1) >= uniq_ids_h.back());

    // // -- free --
    cudaFree(sm_helper);
    cudaFree(sm_pairs);
    cudaFree(sm_seg2);
    cudaFree(sm_seg1);
    cudaFree(sp_helper);
    cudaFree(sp_params);

    // -- return! --
    return std::make_tuple(_spix,border,params);
    // return std::make_tuple(_spix,border);
}

