
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
#ifndef MY_SP_SHARE_H
#define MY_SP_SHARE_H
#include "../bass/share/sp.h"
#endif
#include "../bass/core/Superpixels.h"
// #include "../share/utils.h"
// #include "update_prop_param.h"
#include "update_prop_seg.h"

#include <stdio.h>
#ifndef WIN32
#include <unistd.h>
#endif

__host__ void calc_prop_seg(float* image_gpu_double, int* seg_gpu,
                            int* seg_potts_label, bool* border_gpu,
                            superpixel_params* sp_params, 
                            superpixel_params* sp_params_prev, 
                            superpixel_GPU_helper* sp_gpu_helper,
                            const float3 J_i, const float logdet_Sigma_i, 
                            superpixel_options sp_options,
                            int nbatch, int nftrs, int dim_x, int dim_y, int nSPs,
                            bool use_transition){


    // -- init --
    int dim_i = nftrs; // RGB/BGR/LAB
    int dim_s = 2;

    int npixels = dim_x * dim_y;
    int prior_sigma_s = sp_options.area * sp_options.area;
    int prior_count = sp_options.area;
    const int sizeofint = sizeof(int);
    const int sizeoffloat = sizeof(float);
    bool cal_cov = sp_options.calc_cov;
    float i_std = sp_options.i_std;
    float alpha = sp_options.alpha_hasting;
    int s_std = sp_options.s_std;
    int nInnerIters = sp_options.nInnerIters;
    int split_merge_start = sp_options.split_merge_start;
    int nSPs_buffer = nSPs * 45 ;
    // fprintf(stdout,"split_merge_start: %d\n",split_merge_start);

    int count = 1;
    int count_split =0;
    // gpuErrchk( cudaPeekAtLastError() );
    // gpuErrchk( cudaDeviceSynchronize() );

    for (int i = 0; i < sp_options.nEMIters*1; i++) {
    // printf("%d \n",i);
    // for (int i = 0; i < 3; i++) {
        // "M step"

      update_param(image_gpu_double, seg_gpu, sp_params,
                   sp_gpu_helper, npixels, nSPs, nSPs_buffer,
                   nbatch, dim_x, dim_y, nftrs, prior_sigma_s, prior_count);
        // gpuErrchk( cudaPeekAtLastError() );
        // gpuErrchk( cudaDeviceSynchronize() );


        // fprintf(stdout,"idx,split_merge_start: %d,%d\n",i,split_merge_start);
        // if( (i<sp_options.nEMIters*20) && (i>-1) )
        // {
        //     // if(i>sp_options.nEMIters*split_merge_start){
        //     if(i>split_merge_start){
        //       if((i%4==0)&&(count<100)){
        //         count+=1;

        //         max_SP = CudaCalcSplitCandidate(image_gpu_double, split_merge_pairs,
        //            seg_gpu, border_gpu, sp_params ,sp_gpu_helper,sp_gpu_helper_sm,
        //            npixels,nbatch,dim_x,dim_y,nftrs,nSPs_buffer,seg_split1,seg_split2,
        //                seg_split3,max_SP, count, i_std, alpha);
        //         gpuErrchk( cudaPeekAtLastError() );
        //         gpuErrchk( cudaDeviceSynchronize() );


        //         update_param(image_gpu_double, seg_gpu, sp_params, sp_gpu_helper,
        //                      npixels, nSPs, nSPs_buffer,
        //                      nbatch, dim_x, dim_y, nftrs,
        //                      prior_sigma_s, prior_count);

        //         gpuErrchk( cudaPeekAtLastError() );
        //         gpuErrchk( cudaDeviceSynchronize() );

        //       }
        //       if((i%4==2)&&(count<100)){

        //         for(int j=0; j<1; j++){
        //           CudaCalcMergeCandidate(image_gpu_double, split_merge_pairs, seg_gpu,
        //                    border_gpu, sp_params ,sp_gpu_helper,sp_gpu_helper_sm,
        //                    npixels,nbatch,dim_x,dim_y,nftrs,
        //                    nSPs_buffer,count%2,i_std, alpha);
        //             gpuErrchk( cudaPeekAtLastError() );
        //             gpuErrchk( cudaDeviceSynchronize() );

        //             update_param(image_gpu_double, seg_gpu,
        //                          sp_params, sp_gpu_helper,
        //                          npixels, nSPs, nSPs_buffer,
        //                          nbatch, dim_x, dim_y, nftrs,
        //                          prior_sigma_s, prior_count);

        //             gpuErrchk( cudaPeekAtLastError() );
        //             gpuErrchk( cudaDeviceSynchronize() );

        //         }
        //       }
        //     }
        //     // else{
        //     //   // fprintf(stdout,"update.\n");
        //     //   update_param(image_gpu_double, seg_gpu,
        //     //                sp_params, sp_gpu_helper,
        //     //                npixels, nSPs, nSPs_buffer,
        //     //                nbatch, dim_x, dim_y, nftrs,
        //     //                prior_sigma_s, prior_count);
        //     //   gpuErrchk( cudaPeekAtLastError() );
        //     //   gpuErrchk( cudaDeviceSynchronize() );
        //     // }
        // }


        //"(Hard) E step" - find only the max value after potts term to get the best label
        update_prop_seg(image_gpu_double, seg_gpu, seg_potts_label, border_gpu,
                        sp_params, sp_params_prev, sp_gpu_helper,
                        J_i, logdet_Sigma_i,
                        cal_cov, i_std, s_std, nInnerIters,
                        npixels, nSPs, nSPs_buffer, nbatch,
                        dim_x, dim_y, nftrs, sp_options.beta_potts_term,
                        use_transition);
        // gpuErrchk( cudaPeekAtLastError() );
        // gpuErrchk( cudaDeviceSynchronize() );


      // cudaError_t err_t = cudaDeviceSynchronize();
      //   if (err_t) {
      //       std::cerr << "CUDA error after cudaDeviceSynchronize. " << err_t << std::endl;
      //       cudaError_t err = cudaGetLastError();
      //   }
    }
    CudaFindBorderPixels_end(seg_gpu, border_gpu, npixels, nbatch, dim_x, dim_y, 1);

}

