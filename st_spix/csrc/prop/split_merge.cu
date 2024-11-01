
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

#include "split_merge.h"

#include <stdio.h>
#ifndef WIN32
#include <unistd.h>
#endif

__host__
int run_split(const float* img, int* seg, bool* border,
              spix_params* sp_params, spix_helper* sp_helper,
              spix_helper_sm* sm_helper,
              int* sm_seg1 ,int* sm_seg2, int* sm_pairs,
              float alpha_hastings, float sigma2_app, float sigma2_size,
              int& count, int idx, int max_nspix, 
              const int sp_size, 
              const int npix, const int nbatch,
              const int width, const int height,
              const int nftrs, const int nspix_buffer){

  if(idx%4 == 0){
    count += 1;
    int direction = count%2+1;
    // -- run split --
    max_nspix = CudaCalcSplitCandidate(img, seg, border,
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
void run_merge(const float* img, int* seg, bool* border,
               spix_params* sp_params, spix_helper* sp_helper,
               spix_helper_sm* sm_helper,
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
    CudaCalcMergeCandidate(img, seg, border,
                           sp_params, sp_helper, sm_helper, sm_pairs,
                           sp_size,npix,nbatch,width,height,nftrs,
                           nspix_buffer,direction, alpha_hastings,
                           sigma2_app, sigma2_size);

  }
}

__host__ void CudaCalcMergeCandidate(const float* img, int* seg, bool* border,
                                     spix_params* sp_params,spix_helper* sp_helper,
                                     spix_helper_sm* sm_helper,int* sm_pairs,
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

    init_sm<<<BlockPerGrid2,ThreadPerBlock>>>(img,seg,sp_params,sm_helper,
                                              nspix_buffer, nbatch, width,
                                              nftrs, sm_pairs, nvalid);
    // fprintf(stdout,"direction: %d\n",direction);
    calc_merge_candidate<<<BlockPerGrid,ThreadPerBlock>>>(seg,border, sm_pairs,
                                                          npix, nbatch, width,
                                                          height, direction); 
    sum_by_label_merge<<<BlockPerGrid,ThreadPerBlock>>>(img,seg,sp_params,sm_helper,
                                                        npix, nbatch, width,  nftrs);
    merge_marginal_likelihood<<<BlockPerGrid2,ThreadPerBlock>>>(sm_pairs,sp_params,
                                                                sm_helper,
                                                                sp_size, npix,
                                                                nbatch, width,
                                                                nspix_buffer,
                                                                sigma2_app,sigma2_size);
    merge_hastings_ratio<<<BlockPerGrid2,ThreadPerBlock>>>(img,  sm_pairs, sp_params,
                                                           sp_helper, sm_helper,
                                                           npix, nbatch, width,
                                                           nftrs, nspix_buffer,
                                                           log_alpha,nmerges_gpu);
    // -- count number of merges --
    cudaMemcpy(&nmerges,nmerges_gpu, sizeof(int), cudaMemcpyDeviceToHost);
    printf("nmerges: %d\n",nmerges);
    cudaMemset(nmerges_gpu, 0,sizeof(int));

    cudaMemcpy(&nvalid_cpu, nvalid, sizeof(int), cudaMemcpyDeviceToHost);
    printf("[merge] nvalid: %d\n",nvalid_cpu);

    
    // -- actually merge --
    remove_sp<<<BlockPerGrid2,ThreadPerBlock>>>(sm_pairs,sp_params,
                                                sm_helper,nspix_buffer);
    merge_sp<<<BlockPerGrid,ThreadPerBlock>>>(seg,border, sm_pairs, sp_params,
                                              sm_helper, npix, nbatch, width, height);  

    // -- free! --
    cudaFree(nvalid);
    cudaFree(nmerges_gpu);


}





__host__ int CudaCalcSplitCandidate(const float* img, int* seg, bool* border,
                                    spix_params* sp_params,
                                    spix_helper* sp_helper,
                                    spix_helper_sm* sm_helper,
                                    int* sm_seg1, int* sm_seg2, int* sm_pairs,
                                    const int sp_size,
                                    const int npix, const int nbatch, const int width,
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

    init_sm<<<BlockPerGrid2,ThreadPerBlock>>>(img,seg,sp_params,
                                              sm_helper, nspix_buffer,
                                              nbatch, width, nftrs, sm_pairs, nvalid);
    cudaMemcpy(&nvalid_cpu, nvalid, sizeof(int), cudaMemcpyDeviceToHost);
    // printf("[split] nvalid: %d\n",nvalid_cpu);
    cudaMemset(nvalid, 0,sizeof(int));

    init_split<<<BlockPerGrid2,ThreadPerBlock>>>(border,sm_seg1,sp_params,
                                                 sm_helper, nspix_buffer,
                                                 nbatch, width, height, direction,
                                                 seg, max_sp, max_nspix);
    init_split<<<BlockPerGrid2,ThreadPerBlock>>>(border,sm_seg2,sp_params,
                                                 sm_helper, nspix_buffer,
                                                 nbatch, width,height, -direction,
                                                 seg, max_sp, max_nspix);

    // idk what "split_sp" is doing here; init_sm clears the merge fields and
    // so the function returns immediately...
    split_sp<<<BlockPerGrid,ThreadPerBlock>>>(seg,sm_seg1,sm_pairs,
                                              sp_params, sm_helper, npix,
                                              nbatch, width, height, max_nspix);

    // gpuErrchk( cudaPeekAtLastError() );
    // gpuErrchk( cudaDeviceSynchronize() );
    while(done)
    {
        cudaMemset(done_gpu, 0, sizeof(int));
        cudaMemcpy(&done, done_gpu, sizeof(int), cudaMemcpyDeviceToHost);
        calc_split_candidate<<<BlockPerGrid,ThreadPerBlock>>>(\
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
        calc_split_candidate<<<BlockPerGrid,ThreadPerBlock>>>(\
                sm_seg2,seg,border,distance, done_gpu, npix, nbatch, width, height); 
        distance++;
        cudaMemcpy(&done, done_gpu, sizeof(int), cudaMemcpyDeviceToHost);
        // gpuErrchk( cudaPeekAtLastError() );
        // gpuErrchk( cudaDeviceSynchronize() );
    }

    // updates the segmentation to the two regions; split either left/right or up/down.
    calc_seg_split<<<BlockPerGrid,ThreadPerBlock>>>(sm_seg1,sm_seg2,
                                                    seg, npix,
                                                    nbatch, max_nspix);
    // std::string fname_split1_post = "split1_post";
    // write_tensor_to_file_v2(sm_seg1,height,width,fname_split1_post);

    // computes summaries stats for each split
    sum_by_label_split<<<BlockPerGrid,ThreadPerBlock>>>(img, sm_seg1, sp_params,
                                                        sm_helper, npix, nbatch,
                                                        width,nftrs,max_nspix);
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

    split_marginal_likelihood<<<BlockPerGrid2,ThreadPerBlock>>>(\
        sp_params,sm_helper,sp_size,npix,nbatch,width,nspix_buffer,
        sigma2_app, sigma2_size, max_nspix);

    // calc_marginal_likelihood<<<BlockPerGrid2,ThreadPerBlock>>>(\
    //     sp_params,sm_helper,npix,nbatch,width,nspix_buffer,
    //     sigma2_app, max_nspix);

    // gpuErrchk( cudaPeekAtLastError() );
    // gpuErrchk( cudaDeviceSynchronize() );

    // fprintf(stdout,"[s_m.cu] max_nspix: %d\n",max_nspix);
    split_hastings_ratio<<<BlockPerGrid2,ThreadPerBlock>>>(img, sm_pairs, sp_params,
                                                           sp_helper, sm_helper,
                                                           npix, nbatch, width, nftrs,
                                                           nspix_buffer,
                                                           alpha_hasting_ratio,
                                                           max_nspix, max_sp);

    // -- do the split --
    split_sp<<<BlockPerGrid,ThreadPerBlock>>>(seg,sm_seg1,sm_pairs,
                                              sp_params, sm_helper, npix,
                                              nbatch, width, height, max_nspix);


    // gpuErrchk( cudaPeekAtLastError() );
    // gpuErrchk( cudaDeviceSynchronize() );
    // -- nvalid --
    int prev_max_sp = max_nspix;
    cudaMemcpy(&max_nspix, max_sp, sizeof(int), cudaMemcpyDeviceToHost);
    printf("[split] nsplits: %d\n",max_nspix-prev_max_sp);

    // -- free --
    cudaFree(nvalid);
    cudaFree(max_sp);
    cudaFree(done_gpu);

    return max_nspix;
}



__global__ void init_sm(const float* img, const int* seg_gpu,
                        spix_params* sp_params,
                        spix_helper_sm* sm_helper,
                        const int nspix_buffer, const int nbatch,
                        const int width,const int nftrs,
                        int* sm_pairs, int* nvalid) {
	int k = threadIdx.x + blockIdx.x * blockDim.x;  // the label
	if (k>=nspix_buffer) return;
	// if (sp_params[k].valid == 0) return;
    // atomicAdd(nvalid,1); // update valid

	if (sp_params[k].valid != 0) {
      atomicAdd(nvalid,1); // update valid
    }


    sm_helper[k].b_n_app.x = 0;
    sm_helper[k].b_n_app.y = 0;
    sm_helper[k].b_n_app.z = 0;

	sm_helper[k].sq_sum_app.x = 0;
	sm_helper[k].sq_sum_app.y = 0;
	sm_helper[k].sq_sum_app.z = 0;
    sm_helper[k].sum_app.x = 0;
	sm_helper[k].sum_app.y = 0;
	sm_helper[k].sum_app.z = 0;
    sm_helper[k].count_f = 0;
    sm_helper[k].count = 0;
    sm_helper[k].hasting = -999999;
    //sp_params[k].count = 0;

    sm_helper[k].merge = false;
    sm_helper[k].remove = false;
    sm_pairs[k*2+1] = -1;
    sm_pairs[k*2] = -1;
   

}
__global__
void merge_marginal_likelihood(int* sm_pairs, spix_params* sp_params,
                               spix_helper_sm* sm_helper,
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
    float count_s = __ldg(&sp_params[s].count);
    float count_k = __ldg(&sp_params[k].count);
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
    // double lprob_k = marginal_likelihood_app(sum_k,sq_sum_k,count_k,sigma2_app);
    // double lprob_s = marginal_likelihood_app(sum_s,sq_sum_s,count_s,sigma2_app);
    // double lprob_f = marginal_likelihood_app(sum_f,sq_sum_f,count_f,sigma2_app);
    double sigma2_prior_var = 1.;
    double lprob_k = appearance_variance(sum_k,sq_sum_k,count_k,sigma2_prior_var);
    double lprob_s = appearance_variance(sum_s,sq_sum_s,count_s,sigma2_prior_var);
    double lprob_f = appearance_variance(sum_f,sq_sum_f,count_f,sigma2_prior_var);


    // -- include size term --
    // int sp_size2 = sp_size*sp_size;
    // lprob_k += size_likelihood(count_k,sp_size,sigma2_size);
    // lprob_s += size_likelihood(count_s,sp_size,sigma2_size);
    // lprob_f += size_likelihood(count_f,sp_size,sigma2_size);

    // -- include size term --
    lprob_k += size_beta_likelihood(count_k,sp_size,sigma2_size,npix);
    lprob_s += size_beta_likelihood(count_s,sp_size,sigma2_size,npix);
    lprob_f += size_beta_likelihood(count_f,sp_size,sigma2_size,npix);

    // -- write --
    sm_helper[k].numerator_app = lprob_k;
    sm_helper[s].numerator_app = lprob_s;
    sm_helper[k].numerator_f_app = lprob_f;


}

__global__ void merge_hastings_ratio(const float* img, int* sm_pairs,
                                    spix_params* sp_params,
                                    spix_helper* sp_helper,
                                    spix_helper_sm* sm_helper,
                                    const int npix, const int nbatch, const int width,
                                    const int nftrs, const int nspix_buffer,
                                     float log_alpha, int* nmerges) {

	// getting the index of the pixel
	int k = threadIdx.x + blockIdx.x * blockDim.x;  // the label
	if (k>=nspix_buffer) return;
	if (sp_params[k].valid == 0) return;
    int s = sm_pairs[2*k+1];
    if(s<0) return;
	if (sp_params[s].valid == 0) return;
    // if(s<=0) return;

    // -- unpack --
    float count_s = __ldg(&sp_params[s].count);
    float count_k = __ldg(&sp_params[k].count);
    float count_f = count_s + count_k;
    if((count_f<1)||(count_k<1)||(count_s<1)) return;
    float lprob_k = __ldg(&sm_helper[k].numerator_app);
    float lprob_s = __ldg(&sm_helper[s].numerator_app);
    float lprob_f = __ldg(&sm_helper[k].numerator_f_app);

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
void split_marginal_likelihood(spix_params* sp_params,
                               spix_helper_sm* sm_helper,
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
    int count_f = __ldg(&sp_params[k].count);
    int count_k = __ldg(&sm_helper[k].count);
    int count_s = __ldg(&sm_helper[s].count);

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

    double3 sq_sum_s = sm_helper[s].sq_sum_app;
    double3 sq_sum_k = sm_helper[k].sq_sum_app;
    double3 sq_sum_f;
    sq_sum_f.x = sq_sum_s.x + sq_sum_k.x;
    sq_sum_f.y = sq_sum_s.y + sq_sum_k.y;
    sq_sum_f.z = sq_sum_s.z + sq_sum_k.z;

    // -- marginal likelihood --
    // double lprob_k = marginal_likelihood_app(sum_k,sq_sum_k,count_k,sigma2_app);
    // double lprob_s = marginal_likelihood_app(sum_s,sq_sum_s,count_s,sigma2_app);
    // double lprob_f = marginal_likelihood_app(sum_f,sq_sum_f,count_f,sigma2_app);
    double sigma2_prior_var = 1.;
    double lprob_k = appearance_variance(sum_k,sq_sum_k,count_k,sigma2_prior_var);
    double lprob_s = appearance_variance(sum_s,sq_sum_s,count_s,sigma2_prior_var);
    double lprob_f = appearance_variance(sum_f,sq_sum_f,count_f,sigma2_prior_var);


    // -- include size term --
    // int sp_size2 = sp_size*sp_size;
    // lprob_k += size_likelihood(count_k,sp_size,sigma2_size);
    // lprob_s += size_likelihood(count_s,sp_size,sigma2_size);
    // lprob_f += size_likelihood(count_f,sp_size,sigma2_size);

    // -- include size term --
    lprob_k += size_beta_likelihood(count_k,sp_size,sigma2_size,npix);
    lprob_s += size_beta_likelihood(count_s,sp_size,sigma2_size,npix);
    lprob_f += size_beta_likelihood(count_f,sp_size,sigma2_size,npix);

    // -- write --
    sm_helper[k].numerator_app = lprob_k;
    sm_helper[s].numerator_app = lprob_s;
    sm_helper[k].numerator_f_app = lprob_f;




}

__device__ double size_likelihood(int curr_count, int tgt_count, double sigma2) {
  double delta = 1.*(sqrt(1.*curr_count) - tgt_count);
  double lprob = - log(2*M_PI*sigma2)/2. - delta*delta/(2*sigma2);
  return lprob;
}

__device__ double size_beta_likelihood(int _count, int _tgt_count,
                                       double alpha, const int _npix) {
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

__device__ double appearance_variance(double3 sum_obs,double3 sq_sum_obs,
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

__device__ double marginal_likelihood_app(double3 sum_obs,double3 sq_sum_obs,
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




__global__  void calc_merge_candidate(int* seg, bool* border, int* sm_pairs,
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
void calc_split_candidate(int* dists, int* spix, bool* border,
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


__global__ void init_split(const bool* border, int* seg_gpu,
                           spix_params* sp_params,
                           spix_helper_sm* sm_helper,
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


__global__ void calc_seg_split(int* sm_seg1, int* sm_seg2, int* seg,
                               const int npix, int nbatch, int max_nspix) {
  // todo -- nbatch
    int t = threadIdx.x + blockIdx.x * blockDim.x;
	if (t>=npix) return;
    int seg_val = __ldg(&seg[t]);

    if(sm_seg1[t]>__ldg(&sm_seg2[t])) seg_val += max_nspix; 
    sm_seg1[t] = seg_val;

    return;
}

__global__ void sum_by_label_merge(const float* img, const int* seg_gpu,
                                   spix_params* sp_params,
                                   spix_helper_sm* sm_helper,
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
	// atomicAdd(&sm_helper[k].count, 1); 
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

}

__global__ void sum_by_label_split(const float* img, const int* seg,
                                   spix_params* sp_params,
                                   spix_helper_sm* sm_helper,
                                   const int npix, const int nbatch,
                                   const int width, const int nftrs, int max_nspix) {
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
    return;
}





__global__
void split_hastings_ratio(const float* img, int* sm_pairs,
                          spix_params* sp_params,
                          spix_helper* sp_helper,
                          spix_helper_sm* sm_helper,
                          const int npix, const int nbatch,
                          const int width, const int nftrs,
                          const int nspix_buffer,
                          float log_alpha,
                          int max_nspix, int* max_sp) {
  // todo -- add nbatch and nftrs
	// getting the index of the pixel
	int k = threadIdx.x + blockIdx.x * blockDim.x;  // the label

	if (k>=nspix_buffer) return;
	if (sp_params[k].valid == 0) return;
    
    int s = k + max_nspix;
    if(s>=nspix_buffer) return;
    float count_f = __ldg(&sp_params[k].count);
    float count_k = __ldg(&sm_helper[k].count);
    float count_s = __ldg(&sm_helper[s].count);
    if((count_f<1)||(count_k<1)||(count_s<1)) return;

    float lprob_k = __ldg(&sm_helper[k].numerator_app);
    float lprob_s = __ldg(&sm_helper[s].numerator_app);
    float lprob_f = __ldg(&sm_helper[k].numerator_f_app);

    // -- compute hastings [old] --
    // float log_nominator = log_alpha\
    //   + lgammaf(count_k) +  lgammaf(count_s) + lprob_k + lprob_s;
    // float log_denominator = lgammaf(count_f) + lprob_f;
    // sm_helper[k].hasting = log_nominator - log_denominator;

    // -- compute hastings --
    double log_const = lgammaf(count_k) +  lgammaf(count_s) - lgammaf(count_f);
    // double log_const = 0;
    // double hastings = log_const + log_alpha + lprob_k + lprob_s - lprob_f;
    double hastings = log_alpha + lprob_k + lprob_s - lprob_f;
    sm_helper[k].hasting = hastings;
    sm_helper[k].merge = (sm_helper[k].hasting > -2);
    sm_helper[s].merge = (sm_helper[k].hasting > -2);
    // printf("info[%d,%d] %lf,%f,%f,%f,%lf\n",
    //        k,s,log_const,lprob_f,lprob_k,lprob_s,hastings);

    if((sm_helper[k].merge)) // split step
      {

      // printf("info[%d,%d] %lf,%f,%f,%f\n",k,s,log_const,lprob_f,lprob_k,lprob_s);
        s = atomicAdd(max_sp,1) +1; // ? can't multiple splits happen at one time? yes :D
        sm_pairs[2*k] = s;
        // -- update shape prior --

        float prior_count = max(sp_params[k].prior_count/2.0,8.0);
        sp_params[k].prior_count = prior_count;

        // sp_params[k].prior_sigma_shape.x/=2;
        // sp_params[k].prior_sigma_shape.y/=2;
        // sp_params[k].prior_sigma_shape.z/=2;

        sp_params[s].prior_sigma_shape.x = prior_count;
        sp_params[s].prior_sigma_shape.y = 0;
        sp_params[s].prior_sigma_shape.z = prior_count;

        sp_params[k].prior_sigma_shape.x = prior_count;
        sp_params[k].prior_sigma_shape.y = 0;
        sp_params[k].prior_sigma_shape.z = prior_count;
        
        double2 prior_mu_shape;
        prior_mu_shape.x = 0;
        prior_mu_shape.y = 0;
        sp_params[s].prior_mu_shape = prior_mu_shape;
        sp_params[s].prior_mu_shape_count = 1;
        sp_params[s].prior_count =  sp_params[k].prior_count; 
        sp_params[s].prior_sigma_shape = sp_params[k].prior_sigma_shape;
        

      }

}

__global__ void merge_sp(int* seg, bool* border, int* sm_pairs,
                         spix_params* sp_params,
                         spix_helper_sm* sm_helper,
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

__global__ void split_sp(int* seg, int* sm_seg1, int* sm_pairs,
                         spix_params* sp_params,
                         spix_helper_sm* sm_helper,
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
    if(sm_seg1[idx]==k2) seg[idx] = s;
    //seg[idx] = sm_seg1[idx];
    //printf("Add the following: %d - %d'\n", k,sm_pairs[2*k]);
    sp_params[s].valid = 1;
    // sp_params[sm_pairs[2*k]].prior_count = sp_params[sm_pairs[2*k]].prior_count;
    // sp_params[k].prior_sigma_shape.x = count*count;
    // sp_params[k].prior_sigma_shape.z = count*count;

    // ?

    return;  
}



__global__ void remove_sp(int* sm_pairs, spix_params* sp_params,
                          spix_helper_sm* sm_helper,
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
        sp_params[s].prior_sigma_shape.x+= sp_params[k].prior_sigma_shape.x;
        sp_params[s].prior_sigma_shape.y+= sp_params[k].prior_sigma_shape.y;
        sp_params[s].prior_sigma_shape.z+= sp_params[k].prior_sigma_shape.z;

      }
    else
      {
        sm_helper[k].remove=false;
      }
    
    return;
    
}


