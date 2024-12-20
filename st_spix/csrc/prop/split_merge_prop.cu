
#include <string>
#include <iostream>
#include <fstream>
#include <sstream>
#include <float.h>
#include <math.h>


#ifndef OUT_OF_BOUNDS_LABEL
#define OUT_OF_BOUNDS_LABEL -1
#endif

#define THREADS_PER_BLOCK 512
// #define THREADS_PER_BLOCK 256

#include "split_merge_prop.h"

#include <stdio.h>
#ifndef WIN32
#include <unistd.h>
#endif

__host__
int run_split_p(const float* img, int* seg, bool* border,
                spix_params* sp_params, spix_helper* sp_helper,
                spix_helper_sm_v2* sm_helper,
                int* sm_seg1 ,int* sm_seg2, int* sm_pairs,
                float alpha_hastings, float sigma2_app, float sigma2_size,
                int& count, int idx, int max_nspix, 
                const int sp_size, 
                const int npix, const int nbatch,
                const int width, const int height,
                const int nftrs, const int nspix_buffer){

  // only the propogated spix can be split
  if(idx%4 == 0){
    count += 1;
    int direction = count%2+1;
    // -- run split --
    max_nspix = CudaCalcSplitCandidate_p(img, seg, border,
                                       sp_params, sp_helper, sm_helper,
                                       sm_seg1, sm_seg2, sm_pairs,
                                       sp_size,npix,nbatch,width,height,nftrs,
                                       nspix_buffer, max_nspix,
                                       direction, alpha_hastings,
                                       sigma2_app, sigma2_size);

  }
  return max_nspix;
}

__host__
void run_merge_p(const float* img, int* seg, bool* border,
               spix_params* sp_params, spix_helper* sp_helper,
               spix_helper_sm_v2* sm_helper,
               int* sm_seg1, int* sm_seg2, int* sm_pairs,
               float alpha_hastings,
               float sigma2_app, float sigma2_size,
               int& count, int idx, int max_nspix,
               const int sp_size, const int npix, const int nbatch,
               const int width, const int height,
               const int nftrs, const int nspix_buffer){

  if( idx%4 == 2){
    // -- run merge --
    int direction = count%2;
    // fprintf(stdout,"idx,count,direction: %d,%d,%d\n",idx,count,direction);
    CudaCalcMergeCandidate_p(img, seg, border,
                           sp_params, sp_helper, sm_helper, sm_pairs,
                           sp_size,npix,nbatch,width,height,nftrs,
                           nspix_buffer,direction, alpha_hastings,
                           sigma2_app, sigma2_size);

  }
}

__host__ void CudaCalcMergeCandidate_p(const float* img, int* seg, bool* border,
                                     spix_params* sp_params,spix_helper* sp_helper,
                                     spix_helper_sm_v2* sm_helper,int* sm_pairs,
                                     const int sp_size,
                                     const int npix, const int nbatch,
                                     const int width, const int height,
                                     const int nftrs, const int nspix_buffer,
                                     const int direction, float log_alpha,
                                     float sigma2_app, float sigma2_size){

    int num_block = ceil( double(npix) / double(THREADS_PER_BLOCK) ); 
    int num_block2 = ceil( double(nspix_buffer) / double(THREADS_PER_BLOCK) );
    dim3 BlockPerGrid2(num_block2,nbatch);
    dim3 ThreadPerBlock(THREADS_PER_BLOCK,1);
    dim3 BlockPerGrid(num_block,nbatch);
    // float alpha_hasting_ratio = alpha;
    // float a_0 = 1e6;
    // float b_0 = sigma2_app * (a_0) ;

    int nvalid_cpu;
    int* nvalid;
    cudaMalloc((void **)&nvalid, sizeof(int));
    cudaMemset(nvalid, 0,sizeof(int));

    int nmerges;
    int* nmerges_gpu;
    cudaMalloc((void **)&nmerges_gpu, sizeof(int));
    cudaMemset(nmerges_gpu, 0,sizeof(int));

    init_sm_p<<<BlockPerGrid2,ThreadPerBlock>>>(img,seg,sp_params,sm_helper,
                                                nspix_buffer, nbatch, height, width,
                                                nftrs, npix, sm_pairs, nvalid);
    // fprintf(stdout,"direction: %d\n",direction);
    calc_merge_candidate_p<<<BlockPerGrid,ThreadPerBlock>>>(seg,border, sm_pairs,
                                                          npix, nbatch, width,
                                                          height, direction); 
    sum_by_label_merge_p<<<BlockPerGrid,ThreadPerBlock>>>(img,seg,sp_params,sm_helper,
                                                        npix, nbatch, width,  nftrs);
    merge_marginal_likelihood_p<<<BlockPerGrid2,ThreadPerBlock>>>(sm_pairs,sp_params,
                                                                sm_helper,
                                                                sp_size, npix,
                                                                nbatch, width,
                                                                nspix_buffer,
                                                                sigma2_app,sigma2_size);
    merge_hastings_ratio_p<<<BlockPerGrid2,ThreadPerBlock>>>(img,  sm_pairs, sp_params,
                                                           sp_helper, sm_helper,
                                                           npix, nbatch, width,
                                                           nftrs, nspix_buffer,
                                                           log_alpha,nmerges_gpu);
    // -- count number of merges --
    cudaMemcpy(&nmerges,nmerges_gpu, sizeof(int), cudaMemcpyDeviceToHost);
    // printf("nmerges: %d\n",nmerges);
    cudaMemset(nmerges_gpu, 0,sizeof(int));

    cudaMemcpy(&nvalid_cpu, nvalid, sizeof(int), cudaMemcpyDeviceToHost);
    // printf("[merge] nvalid: %d\n",nvalid_cpu);

    
    // -- actually merge --
    remove_sp_p<<<BlockPerGrid2,ThreadPerBlock>>>(sm_pairs,sp_params,
                                                sm_helper,nspix_buffer);
    merge_sp_p<<<BlockPerGrid,ThreadPerBlock>>>(seg,border, sm_pairs, sp_params,
                                              sm_helper, npix, nbatch, width, height);  

    // -- free! --
    cudaFree(nvalid);
    cudaFree(nmerges_gpu);


}





__host__ int CudaCalcSplitCandidate_p(const float* img, int* seg, bool* border,
                                      spix_params* sp_params,
                                      spix_helper* sp_helper,
                                      spix_helper_sm_v2* sm_helper,
                                      int* sm_seg1, int* sm_seg2, int* sm_pairs,
                                      const int sp_size, const int npix,
                                      const int nbatch, const int width,
                                      const int height, const int nftrs,
                                      const int nspix_buffer, int max_nspix,
                                      int direction, float alpha,
                                      float sigma2_app, float sigma2_size){

    if (max_nspix>nspix_buffer/2){ return max_nspix; }
    int num_block = ceil( double(npix) / double(THREADS_PER_BLOCK) ); 
    int num_block2 = ceil( double(nspix_buffer) / double(THREADS_PER_BLOCK) );
    dim3 BlockPerGrid2(num_block2,1);
    dim3 ThreadPerBlock(THREADS_PER_BLOCK,1);
    dim3 BlockPerGrid(num_block,1);
    float alpha_hasting_ratio =  alpha;
    // float a_0 = 1e6;
    // float b_0 = sigma2_app * (a_0) ;
    // float b_0;
    int done = 1;
    int* done_gpu;
    int* max_sp;
    int nvalid_cpu;
    int* nvalid;
    cudaMalloc((void **)&nvalid, sizeof(int));
    cudaMemset(nvalid, 0,sizeof(int));
    cudaMalloc((void **)&max_sp, sizeof(int));
    cudaMalloc((void **)&done_gpu, sizeof(int)); 

    int distance = 1;
    cudaMemset(sm_seg1, 0, npix*sizeof(int));
    cudaMemset(sm_seg2, 0, npix*sizeof(int));

    init_sm_p<<<BlockPerGrid2,ThreadPerBlock>>>(img,seg,sp_params,
                                                sm_helper, nspix_buffer,
                                                nbatch, height, width,
                                                nftrs, npix, sm_pairs, nvalid);
    cudaMemcpy(&nvalid_cpu, nvalid, sizeof(int), cudaMemcpyDeviceToHost);
    // printf("[split] nvalid: %d\n",nvalid_cpu);
    cudaMemset(nvalid, 0,sizeof(int));

    init_split_p<<<BlockPerGrid2,ThreadPerBlock>>>(border,sm_seg1,sp_params,
                                                   sm_helper, nspix_buffer,
                                                   nbatch, width, height, direction,
                                                   seg, max_sp, max_nspix);
    init_split_p<<<BlockPerGrid2,ThreadPerBlock>>>(border,sm_seg2,sp_params,
                                                 sm_helper, nspix_buffer,
                                                 nbatch, width,height, -direction,
                                                 seg, max_sp, max_nspix);

    // idk what "split_sp" is doing here; init_sm clears the merge fields and
    // so the function returns immediately...
    split_sp_p<<<BlockPerGrid,ThreadPerBlock>>>(seg,sm_seg1,sm_pairs,
                                              sp_params, sm_helper, npix,
                                              nbatch, width, height, max_nspix);

    // gpuErrchk( cudaPeekAtLastError() );
    // gpuErrchk( cudaDeviceSynchronize() );
    while(done)
    {
        cudaMemset(done_gpu, 0, sizeof(int));
        cudaMemcpy(&done, done_gpu, sizeof(int), cudaMemcpyDeviceToHost);
        calc_split_candidate_p<<<BlockPerGrid,ThreadPerBlock>>>(\
                 sm_seg1,seg,border,distance, done_gpu, npix, nbatch, width, height); 
        distance++;
        cudaMemcpy(&done, done_gpu, sizeof(int), cudaMemcpyDeviceToHost);
        // gpuErrchk( cudaPeekAtLastError() );
        // gpuErrchk( cudaDeviceSynchronize() );
    }

    done = 1;
    distance = 1;
    while(done)
    {
		cudaMemset(done_gpu, 0, sizeof(int));
        cudaMemcpy(&done, done_gpu, sizeof(int), cudaMemcpyDeviceToHost);//?
        calc_split_candidate_p<<<BlockPerGrid,ThreadPerBlock>>>(\
                sm_seg2,seg,border,distance, done_gpu, npix, nbatch, width, height); 
        distance++;
        cudaMemcpy(&done, done_gpu, sizeof(int), cudaMemcpyDeviceToHost);
        // gpuErrchk( cudaPeekAtLastError() );
        // gpuErrchk( cudaDeviceSynchronize() );
    }

    // updates the segmentation to the two regions; split either left/right or up/down.
    calc_seg_split_p<<<BlockPerGrid,ThreadPerBlock>>>(sm_seg1,sm_seg2,
                                                    seg, npix,
                                                    nbatch, max_nspix);
    // std::string fname_split1_post = "split1_post";
    // write_tensor_to_file_v2(sm_seg1,height,width,fname_split1_post);

    // computes summaries stats for each split
    sum_by_label_split_p<<<BlockPerGrid,ThreadPerBlock>>>(img, sm_seg1, sp_params,
                                                          sm_helper, npix, nbatch,
                                                          height,width,nftrs,max_nspix);

    // gpuErrchk( cudaPeekAtLastError() );
    // gpuErrchk( cudaDeviceSynchronize() );
    // calc_bn_split<<<BlockPerGrid2,ThreadPerBlock>>>(sm_pairs, sp_params, sp_helper,
    //                                                 sm_helper, npix, nbatch, width,
    //                                                 nspix_buffer, b_0, max_nspix);

    // gpuErrchk( cudaPeekAtLastError() );
    // gpuErrchk( cudaDeviceSynchronize() );
    // split_likelihood<<<BlockPerGrid2,ThreadPerBlock>>>(img,  sm_pairs,
    //                                                    sp_params,  sp_helper,
    //                                                    sm_helper,
    //                                                    npix, nbatch, width, nftrs,
    //                                                    nspix_buffer, a_0,
    //                                                    b_0, max_nspix);

    sample_estimates_p<<<BlockPerGrid2,ThreadPerBlock>>>(\
        sp_params,sm_helper,sp_size,npix,nbatch,width,nspix_buffer,
        sigma2_app, sigma2_size, max_nspix);

    split_marginal_likelihood_p<<<BlockPerGrid2,ThreadPerBlock>>>(\
        sp_params,sm_helper,sp_size,npix,nbatch,width,nspix_buffer,
        sigma2_app, sigma2_size, max_nspix);

    // calc_marginal_likelihood<<<BlockPerGrid2,ThreadPerBlock>>>(\
    //     sp_params,sm_helper,npix,nbatch,width,nspix_buffer,
    //     sigma2_app, max_nspix);

    // gpuErrchk( cudaPeekAtLastError() );
    // gpuErrchk( cudaDeviceSynchronize() );

    // fprintf(stdout,"[s_m.cu] max_nspix: %d\n",max_nspix);
    split_hastings_ratio_p<<<BlockPerGrid2,ThreadPerBlock>>>(img, sm_pairs, sp_params,
                                                             sp_helper, sm_helper,
                                                             npix, nbatch, width, nftrs,
                                                             nspix_buffer,sp_size,
                                                             alpha_hasting_ratio,
                                                             max_nspix, max_sp);

    // -- do the split --
    split_sp_p<<<BlockPerGrid,ThreadPerBlock>>>(seg,sm_seg1,sm_pairs,
                                              sp_params, sm_helper, npix,
                                              nbatch, width, height, max_nspix);


    // gpuErrchk( cudaPeekAtLastError() );
    // gpuErrchk( cudaDeviceSynchronize() );
    // -- nvalid --
    int prev_max_sp = max_nspix;
    cudaMemcpy(&max_nspix, max_sp, sizeof(int), cudaMemcpyDeviceToHost);
    // printf("[split] nsplits: %d\n",max_nspix-prev_max_sp);

    // -- free --
    cudaFree(nvalid);
    cudaFree(max_sp);
    cudaFree(done_gpu);

    return max_nspix;
}



__global__ void init_sm_p(const float* img, const int* seg_gpu,
                          spix_params* sp_params,
                          spix_helper_sm_v2* sm_helper,
                          const int nspix_buffer, const int nbatch,
                          const int height, const int width,
                          const int nftrs, const int npix,
                          int* sm_pairs, int* nvalid) {
	int k = threadIdx.x + blockIdx.x * blockDim.x;  // the label
	if (k>=nspix_buffer) return;
	// if (sp_params[k].valid == 0) return;
    // atomicAdd(nvalid,1); // update valid

	if (sp_params[k].valid != 0) {
      atomicAdd(nvalid,1); // update valid
    }


    // sm_helper[k].b_n_app.x = 0;
    // sm_helper[k].b_n_app.y = 0;
    // sm_helper[k].b_n_app.z = 0;

	sm_helper[k].sq_sum_app.x = 0;
	sm_helper[k].sq_sum_app.y = 0;
	sm_helper[k].sq_sum_app.z = 0;
    sm_helper[k].sum_app.x = 0;
	sm_helper[k].sum_app.y = 0;
	sm_helper[k].sum_app.z = 0;

	sm_helper[k].sq_sum_shape.x = 0;
	sm_helper[k].sq_sum_shape.y = 0;
	sm_helper[k].sq_sum_shape.z = 0;
    sm_helper[k].sum_shape.x = 0;
	sm_helper[k].sum_shape.y = 0;

    sm_helper[k].count = 0;
    sm_helper[k].hasting = -999999;
    //sp_params[k].count = 0;

    sm_helper[k].merge = false;
    sm_helper[k].remove = false;

    // -- invalidate --
    sm_pairs[2*k] = -1;
    sm_pairs[2*k+1] = -1;
    // int k2 = 2*k;
    // if (k2 < 2*npix){
    //   sm_pairs[k2] = -1;
    // }
    // if (k2+1 < 2*npix){
    //   sm_pairs[k2+1] = -1;
    // }

}
__global__
void merge_marginal_likelihood_p(int* sm_pairs, spix_params* sp_params,
                               spix_helper_sm_v2* sm_helper,
                               const int sp_size,
                               const int npix, const int nbatch,
                               const int width, const int nspix_buffer,
                               float sigma2_app, float sigma2_size){

    /********************
           Init
    **********************/

    // -- init --
	int k = threadIdx.x + blockIdx.x * blockDim.x;  // the label
	if (k>=nspix_buffer) return;
	if (sp_params[k].valid == 0) return;
    int s = sm_pairs[2*k+1];
    if (s < 0){ return; }
    // float count_s = __ldg(&sp_params[s].count);
    // float count_k = __ldg(&sp_params[k].count);
    float count_s = __ldg(&sm_helper[s].count);
    float count_k = __ldg(&sm_helper[k].count);
    float count_f = count_s + count_k;

    if((count_f<1)||( count_k<1)||(count_s<1)) return;

    /********************
  
          Appearance
   
    **********************/

    double3 sum_s = sm_helper[s].sum_app;
    double3 sum_k = sm_helper[k].sum_app;
    double3 sum_f;
    sum_f.x = sum_s.x + sum_k.x;
    sum_f.y = sum_s.y + sum_k.y;
    sum_f.z = sum_s.z + sum_k.z;

    double3 sq_sum_s = sm_helper[s].sq_sum_app;
    double3 sq_sum_k = sm_helper[k].sq_sum_app;
    double3 sq_sum_f;
    sq_sum_f.x = sq_sum_s.x + sq_sum_k.x;
    sq_sum_f.y = sq_sum_s.y + sq_sum_k.y;
    sq_sum_f.z = sq_sum_s.z + sq_sum_k.z;

    // -- appearance --
    // double lprob_k = marginal_likelihood_app_p(sum_k,sq_sum_k,count_k,sigma2_app);
    // double lprob_s = marginal_likelihood_app_p(sum_s,sq_sum_s,count_s,sigma2_app);
    // double lprob_f = marginal_likelihood_app_p(sum_f,sq_sum_f,count_f,sigma2_app);
    double sigma2_prior_var = 1.;
    double lprob_k = appearance_variance_p(sum_k,sq_sum_k,count_k,sigma2_prior_var);
    double lprob_s = appearance_variance_p(sum_s,sq_sum_s,count_s,sigma2_prior_var);
    double lprob_f = appearance_variance_p(sum_f,sq_sum_f,count_f,sigma2_prior_var);


    // -- include size term --
    // int sp_size2 = sp_size*sp_size;
    lprob_k += size_likelihood_p(count_k,sp_size,sigma2_size);
    lprob_s += size_likelihood_p(count_s,sp_size,sigma2_size);
    lprob_f += size_likelihood_p(count_f,sp_size,sigma2_size);

    // -- include size term --
    // lprob_k += size_beta_likelihood_p(count_k,sp_size,sigma2_size,npix);
    // lprob_s += size_beta_likelihood_p(count_s,sp_size,sigma2_size,npix);
    // lprob_f += size_beta_likelihood_p(count_f,sp_size,sigma2_size,npix);

    // -- write --
    sm_helper[k].lprob_f_shape = lprob_f;
    sm_helper[s].lprob_s_cond_shape = lprob_s;
    sm_helper[k].lprob_k_cond_shape = lprob_k;


}

__global__ void merge_hastings_ratio_p(const float* img, int* sm_pairs,
                                    spix_params* sp_params,
                                    spix_helper* sp_helper,
                                    spix_helper_sm_v2* sm_helper,
                                    const int npix, const int nbatch, const int width,
                                    const int nftrs, const int nspix_buffer,
                                     float log_alpha, int* nmerges) {

	// getting the index of the pixel
	int k = threadIdx.x + blockIdx.x * blockDim.x;  // the label
	if (k>=nspix_buffer) return;
	if (sp_params[k].valid == 0) return;
    int s = sm_pairs[2*k+1];
    if(s<0) return;
    if (sp_params[k].prop || sp_params[s].prop) { return; }
	if (sp_params[s].valid == 0) return;
    // if(s<=0) return;

    // -- unpack --
    // float count_s = __ldg(&sp_params[s].count);
    // float count_k = __ldg(&sp_params[k].count);
    float count_k = __ldg(&sm_helper[k].count);
    float count_s = __ldg(&sm_helper[s].count);
    float count_f = count_s + count_k;
    if((count_f<1)||(count_k<1)||(count_s<1)) return;
    // float lprob_k = __ldg(&sm_helper[k].numerator_app);
    // float lprob_s = __ldg(&sm_helper[s].numerator_app);
    // float lprob_f = __ldg(&sm_helper[k].numerator_f_app);
    float lprob_k = 0;
    float lprob_s = 0;
    float lprob_f = 0;

    // -- compute hastings --
    double alpha = exp(log_alpha);
    // double log_const = lgammaf(count_f) + lgammaf(alpha) \
    //   + lgammaf(alpha / 2 + count_k) + lgammaf(alpha / 2 + count_s)\
    //   - lgammaf(count_s) - lgammaf(count_k) - lgammaf(alpha+count_f)-2*lgamma(alpha/2);
    double log_const = 0;
    double hastings = log_const + lprob_f - lprob_k - lprob_s - log_alpha;
    // double hastings = lprob_f - lprob_k - lprob_s - log_alpha;
    sm_helper[k].hasting = hastings;
    // sm_helper[k].merge = hastings > 0;
    // sm_helper[s].merge = hastings > 0;

    // printf("info[%d,%d] %f,%f,%f|%lf,%f,%f,%f,%lf|\n",k,s,
    //        count_s,count_k,count_f,
    //        log_const,lprob_f,lprob_k,lprob_s,hastings);

    // -- Check hastings and update --
    if(hastings > 0){

      // printf("info[%d,%d] %f,%f,%f|%lf,%f,%f,%f,%lf|\n",k,s,
      //        count_s,count_k,count_f,
      //        log_const,lprob_f,lprob_k,lprob_s,hastings);

      // printf("info[%d,%d] %lf,%f,%f,%f\n",k,s,log_const,lprob_f,lprob_k,lprob_s);
      int curr_max = atomicMax(&sm_pairs[2*s],k);
      if( curr_max == -1){
        atomicAdd(nmerges,1);
        sm_helper[k].merge = true;
      }else{
        sm_pairs[2*s] = curr_max;
      }
    }
    return;
}




__global__
void sample_estimates_p(spix_params* sp_params,
                        spix_helper_sm_v2* sm_helper,
                        const int sp_size,
                        const int npix, const int nbatch,
                        const int width, const int nspix_buffer,
                        float sigma2_app, float sigma2_size, int max_nspix){

    /********************
           Init
    **********************/

    // -- init --
	int k = threadIdx.x + blockIdx.x * blockDim.x;  // the label
	if (k>=nspix_buffer) return;
	if (sp_params[k].valid == 0) return;

    // -- split --
    int s = k + max_nspix;
	if (s>=nspix_buffer) return;
    // int count_f = __ldg(&sp_params[k].count);
    int count_k = __ldg(&sm_helper[k].count);
    int count_s = __ldg(&sm_helper[s].count);
    int count_f = count_k + count_s;
    if((count_f<1)||( count_k<1)||(count_s<1)) return;
    // count_f = count_k + count_s;

    // -- shape info --
    // note! using "prior_count/2" worked well.
    float prior_count = sp_params[k].prior_count;
    double3 sigma_k = compute_sigma_shape(sm_helper[k].sum_shape,
                                          sm_helper[k].sq_sum_shape,
                                          count_k,prior_count,sp_size);
    double3 sigma_s = compute_sigma_shape(sm_helper[s].sum_shape,
                                          sm_helper[s].sq_sum_shape,
                                          count_s,prior_count,sp_size);
    int2 sum_shape_f = get_sum_shape(sm_helper[s].sum_shape,sm_helper[k].sum_shape);
    longlong3 sq_sum_shape_f = get_sq_sum_shape(sm_helper[s].sq_sum_shape,
                                                sm_helper[k].sq_sum_shape);
    double3 sigma_f = compute_sigma_shape(sum_shape_f,sq_sum_shape_f,
                                          count_f,prior_count,sp_size);

    // -- save shape --
    sm_helper[k].sigma_k = sigma_k;
    sm_helper[k].sigma_s = sigma_s;
    sm_helper[k].sigma_f = sigma_f;

}

__global__
void split_marginal_likelihood_p(spix_params* sp_params,
                               spix_helper_sm_v2* sm_helper,
                               const int sp_size,
                               const int npix, const int nbatch,
                               const int width, const int nspix_buffer,
                               float sigma2_app, float sigma2_size, int max_nspix){

    /********************
           Init
    **********************/

    // -- init --
	int k = threadIdx.x + blockIdx.x * blockDim.x;  // the label
	if (k>=nspix_buffer) return;
	if (sp_params[k].valid == 0) return;

    // -- split --
    int s = k + max_nspix;
	if (s>=nspix_buffer) return;
    // int count_f = __ldg(&sp_params[k].count);
    int count_k = __ldg(&sm_helper[k].count);
    int count_s = __ldg(&sm_helper[s].count);
    int count_f = count_k + count_s;

    if((count_f<1)||( count_k<1)||(count_s<1)) return;


    /********************
  
          Appearance
   
    **********************/

    // float3 mu_pr_k = sp_params[k].prior_mu_app;
    // float3 mu_pr_f = mu_pr_k;
    // sp_params[s].prior_mu_app.x = 0;
    // sp_params[s].prior_mu_app.y = 0;
    // sp_params[s].prior_mu_app.z = 0;
    // float3 mu_pr_s = sp_params[s].prior_mu_app;

    // sp_params[s].prior_mu_app_count = 1;
    // int prior_mu_app_count_s = sp_params[s].prior_mu_app_count;
    // int prior_mu_app_count_k = sp_params[k].prior_mu_app_count;
    // int prior_mu_app_count_f = prior_mu_app_count_k;

    double3 sum_s = sm_helper[s].sum_app;
    double3 sum_k = sm_helper[k].sum_app;
    double3 sum_f;
    sum_f.x = sum_s.x + sum_k.x;
    sum_f.y = sum_s.y + sum_k.y;
    sum_f.z = sum_s.z + sum_k.z;

    // double3 sq_sum_s = sm_helper[s].sq_sum_app;
    // double3 sq_sum_k = sm_helper[k].sq_sum_app;
    // double3 sq_sum_f;
    // sq_sum_f.x = sq_sum_s.x + sq_sum_k.x;
    // sq_sum_f.y = sq_sum_s.y + sq_sum_k.y;
    // sq_sum_f.z = sq_sum_s.z + sq_sum_k.z;

    // -- prior --
    float prior_count = sp_params[k].prior_count;
    float3 prior_mu_app = sp_params[k].prior_mu_app;
    double3 prior_sigma_shape = sp_params[k].prior_sigma_shape;
    double3 ucond_prior_sigma;
    ucond_prior_sigma.x = 1./sp_size;
    ucond_prior_sigma.y = 0;
    ucond_prior_sigma.z = 1./sp_size;

    // -- shape info --
    double3 sigma_k = sm_helper[k].sigma_k;
    double3 sigma_s = sm_helper[k].sigma_s;
    double3 sigma_f = sm_helper[k].sigma_f;
    // printf("[%d] sigma_f.x,sigma_f.y,sigma_f.z: %lf %lf %lf\n",k,sigma_f.x,sigma_f.y,sigma_f.z);
    // double3 sigma_k = compute_sigma_shape(sm_helper[k].sum_shape,
    //                                       sm_helper[k].sq_sum_shape,
    //                                       count_k,prior_count,sp_size);
    // double3 sigma_s = compute_sigma_shape(sm_helper[s].sum_shape,
    //                                       sm_helper[s].sq_sum_shape,
    //                                       count_s,prior_count,sp_size);
    // int2 sum_shape_f = get_sum_shape(sm_helper[s].sum_shape,sm_helper[k].sum_shape);
    // longlong3 sq_sum_shape_f = get_sq_sum_shape(sm_helper[s].sq_sum_shape,
    //                                             sm_helper[k].sq_sum_shape);
    // double3 sigma_f = compute_sigma_shape(sum_shape_f,sq_sum_shape_f,
    //                                       count_f,prior_count,sp_size);

    // -- marginal likelihood --
    // double lprob_k = marginal_likelihood_app_p(sum_k,sq_sum_k,count_k,sigma2_app);
    // double lprob_s = marginal_likelihood_app_p(sum_s,sq_sum_s,count_s,sigma2_app);
    // double lprob_f = marginal_likelihood_app_p(sum_f,sq_sum_f,count_f,sigma2_app);
    // double sigma2_prior_var = 1.;
    // double lprob_k = appearance_variance_p(sum_k,sq_sum_k,count_k,sigma2_prior_var);
    // double lprob_s = appearance_variance_p(sum_s,sq_sum_s,count_s,sigma2_prior_var);
    // double lprob_f = appearance_variance_p(sum_f,sq_sum_f,count_f,sigma2_prior_var);
    // double sigma2_prior_var = 1.;
    // double lprob_app_k = compute_lprob_mu_app(sum_k,prior_mu_app,
    //                                           count_k,sigma2_prior_var);
    // double lprob_app_s = compute_lprob_mu_app(sum_s,prior_mu_app,
    //                                           count_s,sigma2_prior_var);
    // double lprob_app_f = compute_lprob_mu_app(sum_f,prior_mu_app,
    //                                           count_f,sigma2_prior_var);

    // -- shape --
    // double lprob_s_cond = marginal_likelihood_shape_p(sigma_s, prior_sigma_shape,
    //                                                   prior_count,count_s);
    // double lprob_s_ucond = marginal_likelihood_shape_p(sigma_s, ucond_prior_sigma,
    //                                                    prior_count,count_s);
    // double lprob_k_cond = marginal_likelihood_shape_p(sigma_k, prior_sigma_shape,
    //                                                   prior_count,count_k);
    // double lprob_k_ucond = marginal_likelihood_shape_p(sigma_k, ucond_prior_sigma,
    //                                                    prior_count,count_k);
    // double lprob_f = marginal_likelihood_shape_p(sigma_f, prior_sigma_shape,
    //                                              prior_count,count_f);

    double lprob_s_cond = compute_lprob_sigma_shape(sigma_s,prior_sigma_shape);
    double lprob_s_ucond = compute_lprob_sigma_shape(sigma_s,ucond_prior_sigma);
    double lprob_k_cond = compute_lprob_sigma_shape(sigma_k,prior_sigma_shape);
    double lprob_k_ucond = compute_lprob_sigma_shape(sigma_k,ucond_prior_sigma);
    double lprob_f = compute_lprob_sigma_shape(sigma_f,prior_sigma_shape);

    // -- prob --
    // double lprob_s_cond = 0;
    // double lprob_s_ucond = 0;
    // double lprob_k_cond = 0;
    // double lprob_k_ucond = 0;
    // double lprob_f = 0;

    // -- append appearance to cond --
    double sigma2_prior_var = 1.;
    lprob_s_cond += compute_l2norm_mu_app_p(sum_s,prior_mu_app,
                                            count_s,sigma2_prior_var);
    lprob_k_cond += compute_l2norm_mu_app_p(sum_k,prior_mu_app,
                                            count_k,sigma2_prior_var);
    lprob_f += compute_l2norm_mu_app_p(sum_f,prior_mu_app,
                                       count_f,sigma2_prior_var);

    // -- include size term --
    // float _sp_size = 1.*sp_size;
    float _sp_size_v0 = 1.*prior_count;
    float _sp_size_v1 = 1.*sp_size;
    lprob_f += size_likelihood_p(count_f,_sp_size_v0,sigma2_size);
    lprob_s_cond += size_likelihood_p(count_s,_sp_size_v0,sigma2_size);
    lprob_s_ucond += size_likelihood_p(count_s,_sp_size_v1,sigma2_size);
    lprob_k_cond += size_likelihood_p(count_k,_sp_size_v0,sigma2_size);
    lprob_k_ucond += size_likelihood_p(count_k,_sp_size_v1,sigma2_size);

    // -- include size term --
    // lprob_k += size_beta_likelihood_p(count_k,sp_size,sigma2_size,npix);
    // lprob_s += size_beta_likelihood_p(count_s,sp_size,sigma2_size,npix);
    // lprob_f += size_beta_likelihood_p(count_f,sp_size,sigma2_size,npix);

    // -- write --
    sm_helper[k].lprob_f_shape = lprob_f;
    sm_helper[k].lprob_s_cond_shape = lprob_s_cond;
    sm_helper[k].lprob_s_ucond_shape = lprob_s_ucond;
    sm_helper[k].lprob_k_cond_shape = lprob_k_cond;
    sm_helper[k].lprob_k_ucond_shape = lprob_k_ucond;

    // printf("[%d]: %lf %lf %lf | %lf %lf | %lf %lf %lf | %lf %lf %lf\n",
    //        k,lprob_f,lprob_s_cond,lprob_k_cond,
    //        lprob_s_ucond,lprob_k_ucond,
    //        sigma_f.x,sigma_f.y,sigma_f.z,
    //        prior_sigma_shape.x,prior_sigma_shape.y,prior_sigma_shape.z);

}

__device__ double size_likelihood_p(int curr_count, float tgt_count, double sigma2) {
  double delta = 1.*(sqrt(1.*curr_count) - tgt_count);
  double lprob = - log(2*M_PI*sigma2)/2. - delta*delta/(2*sigma2);
  return lprob;
}

__device__ double size_beta_likelihood_p(int _count, int _tgt_count,
                                       double alpha, const int _npix) {
  if (alpha < 0){ return 0; }
  double count = 1.*_count;
  double npix = 1.*_npix;
  double tgt_count = 1*_tgt_count*_tgt_count;
  double beta = alpha*(npix-tgt_count)/(tgt_count+1e-10); // just in case...
  // double beta = alpha;
  double lprob = (alpha-1)*log(count/npix) + (beta-1)*log(1-count/npix);
  // lprob += lgammaf(npix*alpha/tgt_count) - lgammaf(alpha) - lgammaf(beta);
  lprob += lgamma(alpha+beta) - lgamma(alpha) - lgamma(beta);
  return lprob;
}


__device__ double compute_lprob_sigma_shape(double3 sigma_est,
                                            double3 prior_sigma) {
  
  // double sxx = sigma_est.x;
  // double sxy = sigma_est.y;
  // double syy = sigma_est.y;
  // double det_est = abs(sxx*syy-sxy*sxy);


  // double pr_sxx = prior_sigma.x;
  // double pr_sxy = prior_sigma.y;
  // double pr_syy = prior_sigma.y;
  // double det_pr = abs(pr_sxx*pr_syy-pr_sxy*pr_sxy);

  // float df = prior_count;
  // double lprob = (df / 2.0) * log(det_pr) - ((df + nobs + 1) / 2.0) * log(det_sigma) - log(

  //   lgamma- lgamma(df/2.0) - lgamma((df-1)/2.0) - log(M_PI)/2.0 - df*log(2);

  double lprob = -wasserstein_p(sigma_est,prior_sigma);
  // double lprob = 0;
  return lprob;
}

__device__ double marginal_likelihood_shape_p(double3 sigma_est, double3 prior_sigma,
                                              float pr_count,int num_obs) {

  // -- compute det --
  double sxx = sigma_est.x;
  double sxy = sigma_est.y;
  double syy = sigma_est.z;
  double det_est = abs(sxx*syy-sxy*sxy);
  assert(det_est>0.001);

  double pr_sxx = prior_sigma.x;
  double pr_sxy = prior_sigma.y;
  double pr_syy = prior_sigma.z;
  double det_pr = abs(pr_sxx*pr_syy-pr_sxy*pr_sxy);
  assert(det_pr>0.001);

  // -- compute marginal --
  if (det_est < 0.001){ return -10000000; }
  double post_count = 1.*pr_count + num_obs;
  double gamma2_post = lgamma(post_count/2) + lgamma((post_count-1)/2);
  double gamma2_pr = lgamma(pr_count/2) + lgamma((pr_count-1)/2);
  double h_const = num_obs * log(M_PI) + log(pr_count) - log(post_count);
  // double lprob = pr_count*log(det_pr) - post_count*log(det_post) + gamma2_post - gamma2_pr + h_const;
  double lprob = pr_count*log(det_pr) - post_count*log(det_est);
  return lprob;
}


__device__ double wasserstein_p(double3 sigma_est,double3 sigma_prior) {

  // -- ... --
  // printf("s11,s12,s22 | %lf %lf %lf | %lf %lf %lf \n",
  //        sigma_est.x,sigma_est.y,sigma_est.z,
  //        sigma_prior.x,sigma_prior.y,sigma_prior.z);

  // Step 1: Compute eigenvalues for sigma_est
  double3 eigen_est = eigenvals_cov_p(sigma_est);
  double lambda1_est = eigen_est.x;
  double lambda2_est = eigen_est.y;

  // Step 2: Compute eigenvalues for sigma_prior
  double3 eigen_prior = eigenvals_cov_p(sigma_prior);
  double lambda1_prior = eigen_prior.x;
  double lambda2_prior = eigen_prior.y;

  // Step 3: Compute the trace term
  // double trace_term = sigma_est.x + sigma_est.z + sigma_prior.x + sigma_prior.z;
  double trace_term = eigen_est.z + eigen_prior.z;

  // Step 4: Compute the cross term using the square root of the products of eigenvalues
  double cross_term = 2.0 * (sqrt(lambda1_est * lambda1_prior) + \
                             sqrt(lambda2_est * lambda2_prior));

  // -- info --
  // printf("lambda1_est,lambda2_est,lambda1_prior,lambda2_prior: %lf %lf %lf %lf\n",
  //        lambda1_est,lambda2_est,lambda1_prior,lambda2_prior);

  // Step 5: Wasserstein squared distance
  double wasserstein_distance_squared = trace_term - cross_term;

  // Return the square root to get the actual Wasserstein distance
  return sqrt(wasserstein_distance_squared);

}

__device__ double3 eigenvals_cov_p(double3 icov) {

  // -- unpack --
  double s11 = icov.x;
  double s12 = icov.y;
  double s22 = icov.z;

  // Calculate the trace and determinant
  double determinant = 1./(s11 * s22 - s12 * s12); // inverse cov rather than cov
  double trace = (s11 + s22)*determinant;
  //printf("s11,s12,s22,det,trace: %lf %lf %lf %lf %lf\n",s11,s12,s22,determinant,trace);

  // // -- info --
  // printf("sxx,sxy,syy,pc: %lf %lf %lf %f %lf | %d %d | %lld %lld %lld\n",
  //        sigma.x,sigma.y,sigma.z,prior_count,count,
  //        sum.x,sum.y,sq_sum.x,sq_sum.y,sq_sum.z);
  // assert(determinant>0.0001);
  // printf("s11,s22,det,trace: %lf %lf %lf %lf\n",s11,s22,determinant,trace);

  // Calculate the square root term
  double tmp = (trace * trace)/4.0;
  double term;
  if (tmp > determinant){
    term = sqrt(tmp - determinant);
  }else{
    term = 0;
  }

  // Compute the two eigenvalues
  double lambda1 = (trace / 2) + term;
  double lambda2 = (trace / 2) - term;
  // printf("det,trace,term: %lf %lf %lf\n",determinant,trace,term);

  return make_double3(lambda1, lambda2, trace);
}



__device__ double compute_l2norm_mu_app_p(double3 sum_obs,float3 prior_mu,
                                          int _num_obs, double sigma2) {
  double num_obs = 1.*_num_obs;
  double delta_x = (sum_obs.x/num_obs - prior_mu.x);
  double delta_y = (sum_obs.y/num_obs - prior_mu.y);
  double delta_z = (sum_obs.z/num_obs - prior_mu.z);
  double l2norm = (delta_x*delta_x + delta_y*delta_y + delta_z*delta_z)/3.;
  return sqrt(l2norm);
}


// __device__ double compute_lprob_mu_app(double3 sum_obs,float3 prior_mu,
//                                       int _num_obs, double sigma2) {
//   double num_obs = 1.*_num_obs;
//   double delta_x = (sum_obs.x/num_obs - prior_mu.x);
//   double delta_y = (sum_obs.y/num_obs - prior_mu.y);
//   double delta_z = (sum_obs.z/num_obs - prior_mu.z);
//   double lprob = delta_x*delta_x + delta_y*delta_y + delta_z*delta_z;
//   lprob = -lprob/(2*sigma2);
//   return lprob;

// }

__device__ double appearance_variance_p(double3 sum_obs,double3 sq_sum_obs,
                                      int _num_obs, double sigma2) {
  double num_obs = 1.*_num_obs;
  double sample_var = (sq_sum_obs.x  - sum_obs.x*sum_obs.x);
  sample_var += (sq_sum_obs.y  - sum_obs.y*sum_obs.y);
  sample_var += (sq_sum_obs.z  - sum_obs.z*sum_obs.z);
  sample_var = sample_var/(3.*num_obs); // estimate sigma2
  // sample_var = sample_var/3.; // estimate sigma2
  double lprob = -sample_var/sigma2;
  return lprob;
}

__device__ double marginal_likelihood_app_p(double3 sum_obs,double3 sq_sum_obs,
                                          int _num_obs, double sigma2) {
  // ref: from https://www.cs.ubc.ca/~murphyk/Papers/bayesGauss.pdf
  // Equation 55 with modifications from Eq 57 where kappa = 1
  // -- silly; we should just replace forumla with tau2 -> infty limit --
  double tau2 = 1000.; // ~= mean has 95% prob to be within (-1,1)
  float num_obs = (float)_num_obs;

  // float3 mu_prior;

  double lprob_num = 1/2. * log(sigma2) - num_obs/2.0 * log(2*M_PI*sigma2) \
    - log(num_obs * tau2 + sigma2)/2.;
  double denom = 2*(num_obs*tau2+sigma2);
  double3 lprob;
  lprob.x = lprob_num - sq_sum_obs.x/(2*sigma2) \
    + tau2*sum_obs.x*sum_obs.x/(sigma2*denom);
  lprob.y = lprob_num - sq_sum_obs.y/(2*sigma2)
    + tau2*sum_obs.y*sum_obs.y/(sigma2*denom);
  lprob.z = lprob_num - sq_sum_obs.z/(2*sigma2)
    + tau2*sum_obs.z*sum_obs.z/(sigma2*denom);

  double _lprob;
  _lprob = lprob.x+lprob.y+lprob.z;
  return _lprob;
}




__global__  void calc_merge_candidate_p(int* seg, bool* border, int* sm_pairs,
                                      const int npix, const int nbatch,
                                      const int width, const int height,
                                      const int direction){
  // todo: add nbatch
    int idx = threadIdx.x + blockIdx.x * blockDim.x;  
    if (idx>=npix) return; 
    if(!border[idx]) return;
    int x = idx % width;
    int y = idx / width;

    int C = seg[idx]; // center 
    int W; // north, south, east,west            
    W = OUT_OF_BOUNDS_LABEL; // init 

    if(direction==1){
      if ((y>1) && (y< height-2))
        {
          W = __ldg(&seg[idx+width]);  // down
        }
    }else{
      if ((x>1) && (x< width-2))
        {
          W = __ldg(&seg[idx-1]);  // left
        }
    }
        
    // If the nbr is different from the central pixel and is not out-of-bounds,
    // then it is a border pixel.
    if (W>=0 && C!=W){
      atomicMax(&sm_pairs[C*2+1],W);
    }

    return;        
}

__global__
void calc_split_candidate_p(int* dists, int* spix, bool* border,
                          int distance, int* mutex, const int npix,
                          const int nbatch, const int width, const int height){
  
    // todo: add batch -- no nftrs
    int idx = threadIdx.x + blockIdx.x * blockDim.x;  
    if (idx>=npix) return; 
    int x = idx % width;
    int y = idx / width;
    int C = dists[idx]; // center 
    int spixC = spix[idx];
    // if (border[idx]) return; 

    if(C!=distance) return;

    if ((y>0)&&(idx-width>=0)){
      if((!dists[idx-width]) and (spix[idx-width] == spixC)){
        dists[idx-width] = distance+1;
        mutex[0] = 1;
      }
    }          
    if ((x>0)&&(idx-1>=0)){
      if((!dists[idx-1]) and (spix[idx-1] == spixC)){
        dists[idx-1] = distance+1;
        mutex[0] = 1;
      }
    }
    if ((y<height-1)&&(idx+width<npix)){
      if((!dists[idx+width]) and (spix[idx+width] == spixC)){
        dists[idx+width] = distance+1;
        mutex[0] = 1;
      }
    }   
    if ((x<width-1)&&(idx+1<npix)){
      if((!dists[idx+1]) and (spix[idx+1] == spixC)){
        dists[idx+1] = distance+1;
        mutex[0] = 1;
      }
    }
    
    return;        
}


__global__ void init_split_p(const bool* border, int* seg_gpu,
                           spix_params* sp_params,
                           spix_helper_sm_v2* sm_helper,
                           const int nspix_buffer,
                           const int nbatch, const int width,
                           const int height, const int direction,
                           const int* seg, int* max_sp, int max_nspix) {

    // todo: add batch -- no nftrs
	int k = threadIdx.x + blockIdx.x * blockDim.x;  // the label
    // *max_sp = max_nspix+1;
    *max_sp = max_nspix; // MAX number -> MAX label
	if (k>=nspix_buffer) return;
	if (sp_params[k].valid == 0) return;
    int x;
    int y;
    if((direction==1)||(direction==-1))
    {
        x = int(sp_params[k].mu_shape.x)+direction;
        y = int(sp_params[k].mu_shape.y);
    }
    else
    {
        x = int(sp_params[k].mu_shape.x);
        y = int(sp_params[k].mu_shape.y)+direction;
    }
    
    int ind = y*width+x;
    if((ind<0)||(ind>width*height-1)) return;
    
    // if(border[ind]) return;
    if (seg[ind]!=k) return;
    seg_gpu[ind] = 1;

}


__global__ void calc_seg_split_p(int* sm_seg1, int* sm_seg2, int* seg,
                               const int npix, int nbatch, int max_nspix) {
  // todo -- nbatch
    int t = threadIdx.x + blockIdx.x * blockDim.x;
	if (t>=npix) return;
    int seg_val = __ldg(&seg[t]);

    if(sm_seg1[t]>__ldg(&sm_seg2[t])) seg_val += max_nspix; 
    sm_seg1[t] = seg_val;

    return;
}

__global__ void sum_by_label_merge_p(const float* img, const int* seg_gpu,
                                   spix_params* sp_params,
                                   spix_helper_sm_v2* sm_helper,
                                   const int npix, const int nbatch,
                                   const int width, const int nftrs) {
  // todo: nbatch
	// getting the index of the pixel
    int t = threadIdx.x + blockIdx.x * blockDim.x;
	if (t>=npix) return;

	//get the label
	int k = __ldg(&seg_gpu[t]);
    float l = __ldg(& img[3*t]);
    float a = __ldg(& img[3*t+1]);
    float b = __ldg(& img[3*t+2]);
	//atomicAdd(&sp_params[k].count, 1); //TODO: Time it
	atomicAdd(&sm_helper[k].count, 1); 
	atomicAdd(&sm_helper[k].sq_sum_app.x, l*l);
	atomicAdd(&sm_helper[k].sq_sum_app.y, a*a);
	atomicAdd(&sm_helper[k].sq_sum_app.z,b*b);
    atomicAdd(&sm_helper[k].sum_app.x, l);
	atomicAdd(&sm_helper[k].sum_app.y, a);
	atomicAdd(&sm_helper[k].sum_app.z, b);
    
	int x = t % width;
	int y = t / width; 
    atomicAdd(&sm_helper[k].sum_shape.x, x);
    atomicAdd(&sm_helper[k].sum_shape.y, y);
    atomicAdd((unsigned long long *)&sm_helper[k].sq_sum_shape.x, x*x);
    atomicAdd((unsigned long long *)&sm_helper[k].sq_sum_shape.y, x*y);
    atomicAdd((unsigned long long *)&sm_helper[k].sq_sum_shape.z, y*y);


}

__global__ void sum_by_label_split_p(const float* img, const int* seg,
                                     spix_params* sp_params,
                                     spix_helper_sm_v2* sm_helper,
                                     const int npix, const int nbatch,
                                     const int height, const int width,
                                     const int nftrs, int max_nspix) {
  // todo: nbatch
	// getting the index of the pixel
    int t = threadIdx.x + blockIdx.x * blockDim.x;
	if (t>=npix) return;

	//get the label
    
	int k = __ldg(&seg[t]);
    float l = __ldg(& img[3*t]);
    float a = __ldg(& img[3*t+1]);
    float b = __ldg(& img[3*t+2]);
	atomicAdd(&sm_helper[k].count, 1); 
    atomicAdd(&sm_helper[k].sum_app.x, l);
	atomicAdd(&sm_helper[k].sum_app.y, a);
	atomicAdd(&sm_helper[k].sum_app.z, b);
	atomicAdd(&sm_helper[k].sq_sum_app.x, l*l);
	atomicAdd(&sm_helper[k].sq_sum_app.y, a*a);
	atomicAdd(&sm_helper[k].sq_sum_app.z,b*b);
    
	int x = t % width;
	int y = t / width; 
    atomicAdd(&sm_helper[k].sum_shape.x, x);
    atomicAdd(&sm_helper[k].sum_shape.y, y);
    atomicAdd((unsigned long long *)&sm_helper[k].sq_sum_shape.x, x*x);
    atomicAdd((unsigned long long *)&sm_helper[k].sq_sum_shape.y, x*y);
    atomicAdd((unsigned long long *)&sm_helper[k].sq_sum_shape.z, y*y);
    return;
}





__global__
void split_hastings_ratio_p(const float* img, int* sm_pairs,
                            spix_params* sp_params,
                            spix_helper* sp_helper,
                            spix_helper_sm_v2* sm_helper,
                            const int npix, const int nbatch,
                            const int width, const int nftrs,
                            const int nspix_buffer,
                            int sp_size, float log_alpha,
                            int max_nspix, int* max_sp) {
  // todo -- add nbatch and nftrs
	// getting the index of the pixel
	int k = threadIdx.x + blockIdx.x * blockDim.x;  // the label

	if (k>=nspix_buffer) return;
	if (sp_params[k].valid == 0) return;
	if (sp_params[k].prop == false) return;
    
    int s = k + max_nspix;
    if(s>=nspix_buffer) return;
    // float count_f = __ldg(&sp_params[k].count);
    float count_s = __ldg(&sm_helper[s].count);
    float count_k = __ldg(&sm_helper[k].count);
    float count_f = count_k + count_s;

    if((count_f<1)||(count_k<1)||(count_s<1)) return;

    // float lprob_k = __ldg(&sm_helper[k].numerator_app);
    // float lprob_s = __ldg(&sm_helper[s].numerator_app);
    // float lprob_f = __ldg(&sm_helper[k].numerator_f_app);

    // sm_helper[k].numerator_f_app = lprob_f;
    // sm_helper[k].denominator.x = lprob_s_cond;
    // sm_helper[k].denominator.y = lprob_s_ucond;
    // sm_helper[k].denominator_f.x = lprob_k_cond;
    // sm_helper[k].denominator_f.y = lprob_k_ucond;

    // -- unpack --
    float lprob_f = __ldg(&sm_helper[k].lprob_f_shape);
    float lprob_k_cond = __ldg(&sm_helper[k].lprob_k_cond_shape);
    float lprob_k_ucond = __ldg(&sm_helper[k].lprob_k_ucond_shape);
    float lprob_s_cond = __ldg(&sm_helper[k].lprob_s_cond_shape);
    float lprob_s_ucond = __ldg(&sm_helper[k].lprob_s_ucond_shape);

    // -- determine which is "conditioned" --
    bool select = lprob_k_cond > lprob_s_cond;
    float lprob_sel_cond = select ? lprob_k_cond : lprob_s_cond;
    float lprob_sel_ucond = select ? lprob_s_ucond : lprob_k_ucond;
    sm_helper[k].select = select; // pick "k" if true

    // -- compute hastings [old] --
    // float log_nominator = log_alpha\
    //   + lgammaf(count_k) +  lgammaf(count_s) + lprob_k + lprob_s;
    // float log_denominator = lgammaf(count_f) + lprob_f;
    // sm_helper[k].hasting = log_nominator - log_denominator;

    // -- compute hastings --
    double log_const = lgammaf(count_k) +  lgammaf(count_s) - lgammaf(count_f);
    // // double log_const = 0;
    // // double hastings = log_const + log_alpha + lprob_k + lprob_s - lprob_f;
    // double hastings = log_alpha + lprob_k + lprob_s - lprob_f;

    // -- determine if any splitting --

    // -- [looks good] --
    double hastings = log_alpha + lprob_sel_cond + lprob_sel_ucond - lprob_f;

    // -- [bad; too many long cuts] --
    // double hastings = log_alpha + lprob_sel_cond - lprob_f;

    sm_helper[k].hasting = hastings;
    sm_helper[k].merge = (sm_helper[k].hasting > 0);
    sm_helper[s].merge = (sm_helper[k].hasting > 0);
    // printf("info[%d,%d] %lf,%f,%f,%f,%f,%f,%lf\n",
    //        k,s,log_const,lprob_f,lprob_k_cond,lprob_s_cond,
    //        lprob_k_ucond,lprob_s_ucond,hastings);

    if((sm_helper[k].merge)) // split step
      {

      // printf("info[%d,%d] %lf,%f,%f,%f\n",k,s,log_const,lprob_f,lprob_k,lprob_s);
        // s = atomicAdd(max_sp,1) +1; //
        s = atomicAdd(max_sp,1)+1; // ? can't multiple splits happen at one time? yes :D
        sm_pairs[2*k] = s;

        // -- init new spix --
        float prior_count = max(sp_params[k].prior_count/2.0,8.0);
        sp_params[k].prior_count = prior_count;
        sp_params[s].prior_count = prior_count;

        // double3 prior_sigma_shape;
        // prior_sigma_shape.x = 1./sp_size;
        // prior_sigma_shape.y = 0;
        // prior_sigma_shape.z = 1./sp_size;
        // sp_params[s].prior_sigma_shape = prior_sigma_shape;
        
        // double2 prior_mu_shape;
        // prior_mu_shape.x = 0;
        // prior_mu_shape.y = 0;
        // sp_params[s].prior_mu_shape = prior_mu_shape;
        sp_params[s].prop = false;
        sp_params[s].valid = 1;

        // // -- [appearance] prior --
        // float3 prior_mu_app;
        // prior_mu_app.x = 0;
        // prior_mu_app.y = 0;
        // prior_mu_app.z = 0;
        // sp_params[s].prior_mu_app = prior_mu_app;
        // sp_params[s].prior_mu_app_count = 1;

        // // -- [shape] prior --
        // double2 prior_mu_shape;
        // prior_mu_shape.x = 0;
        // prior_mu_shape.y = 0;
        // sp_params[s].prior_mu_shape = prior_mu_shape;
        // sp_params[s].prior_mu_shape_count = 1;
        // double3 prior_sigma_shape;
        // prior_sigma_shape.x = prior_count;
        // prior_sigma_shape.y = 0;
        // prior_sigma_shape.z = prior_count;
        // sp_params[s].prior_sigma_shape = prior_sigma_shape;
        // sp_params[s].prior_sigma_shape_count = prior_count;
        // sp_params[s].prop = false;

      }

}

__global__ void merge_sp_p(int* seg, bool* border, int* sm_pairs,
                         spix_params* sp_params,
                         spix_helper_sm_v2* sm_helper,
                         const int npix, const int nbatch,
                         const int width, const int height){   
    int idx = threadIdx.x + blockIdx.x * blockDim.x;  
    if (idx>=npix) return; 
    int k = seg[idx]; // center 
    //if (sp_params[k].valid == 0) return;
    int f = sm_pairs[2*k+1];
    if(sm_helper[k].remove){
      seg[idx] =  f;
    }

    return;  
      
}

__global__ void split_sp_p(int* seg, int* sm_seg1, int* sm_pairs,
                         spix_params* sp_params,
                         spix_helper_sm_v2* sm_helper,
                         const int npix, const int nbatch,
                         const int width, const int height, int max_nspix){   

  // todo: add nbatch, no sftrs
    int idx = threadIdx.x + blockIdx.x * blockDim.x;  
    if (idx>=npix) return; 
    int k = seg[idx]; // center 
    int k2 = k + max_nspix;
    if (sp_params[k].valid == 0){ return; }
    if ((sm_helper[k].merge == false)||sm_helper[k2].merge == false){
      return;
    }

    int s = sm_pairs[2*k];
    if (s < 0){ return; }
    
    if (sm_helper[k].select){
      if(sm_seg1[idx]==k2) {
        seg[idx] = s;
      }
    }else{
      if(sm_seg1[idx]==k) {
        seg[idx] = s;
      }
    }

    return;  
}



__global__ void remove_sp_p(int* sm_pairs, spix_params* sp_params,
                          spix_helper_sm_v2* sm_helper,
                          const int nspix_buffer) {

	// -- getting the index of the pixel --
	int k = threadIdx.x + blockIdx.x * blockDim.x;  // the label
	if (k>=nspix_buffer) return;
    int s = sm_pairs[2*k+1];
    if(s<0) return;
    if ((sp_params[k].valid == 0)||(sp_params[s].valid == 0)) return;    
    // if ((sm_helper[k].merge == true) && (sm_helper[f].merge == false) && (split_merge_pairs[2*f]==k) )
    if ((sm_helper[k].merge==true)&&(sm_helper[s].merge==false)&&(sm_pairs[2*s]==k))
    // if ((sm_helper[k].merge == true) && (sm_helper[s].merge == false))
      {
        sm_helper[k].remove=true;
        sp_params[k].valid = 0;

        // -- update priors --
        sp_params[s].prior_count =sp_params[k].prior_count+sp_params[s].prior_count;
        // sp_params[s].prior_sigma_shape.x = 1.0/sp_params[s].prior_count;
        // sp_params[s].prior_sigma_shape.y = 0.;
        // sp_params[s].prior_sigma_shape.z = 1.0/sp_params[s].prior_count;

      }
    else
      {
        sm_helper[k].remove=false;
      }
    
    return;
    
}



__device__ int2 get_sum_shape(int2 sum_s, int2 sum_k){
  int2 sum_f;
  sum_f.x = sum_s.x+sum_k.x;
  sum_f.y = sum_s.y+sum_k.y;
  return sum_f;
}

__device__ longlong3 get_sq_sum_shape(longlong3 sq_sum_s, longlong3 sq_sum_k){
  longlong3 sq_sum_f;
  sq_sum_f.x = sq_sum_s.x+sq_sum_k.x;
  sq_sum_f.y = sq_sum_s.y+sq_sum_k.y;
  sq_sum_f.z = sq_sum_s.z+sq_sum_k.z;
  return sq_sum_f;
}


__device__ double3 compute_sigma_shape(int2 sum, longlong3 sq_sum,
                                       int _count, float prior_count, int sp_size) {

  // -- mean --
  double count = 1.0*_count;
  double2 mu;
  mu.x = sum.x/count;
  mu.y = sum.y/count;
  
  // -- sample covariance --
  double3 sigma;
  sigma.x = sq_sum.x - (mu.x * mu.x)*count;
  sigma.y = sq_sum.y - (mu.x * mu.y)*count;
  sigma.z = sq_sum.z - (mu.y * mu.y)*count;

  // // -- info --
  // printf("sxx,sxy,syy,pc: %lf %lf %lf %f %lf | %d %d | %lld %lld %lld\n",
  //        sigma.x,sigma.y,sigma.z,prior_count,count,
  //        sum.x,sum.y,sq_sum.x,sq_sum.y,sq_sum.z);

  // -- sample cov --
  double total_count = 1.*(count + prior_count);
  sigma.x = (prior_count*sp_size + sigma.x)/(total_count + 3.0);
  sigma.y = (sigma.y) / (total_count + 3.0);
  sigma.z = (prior_count*sp_size + sigma.z)/(total_count + 3.0);

  // -- determinant --
  double det = sigma.x*sigma.z - sigma.y*sigma.y;
  if (det < 0.0001){ det = 1.; }
  double tmp;
  double3 isigma;
  tmp = sigma.x;
  sigma.x = sigma.z/det;
  sigma.y = -sigma.y/det;
  sigma.z = tmp/det;
  // isigma.x = sigma.z/det;
  // isigma.y = -sigma.y/det;
  // isigma.z = sigma.x/det;
  // return isigma;

  return sigma;
}

