
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
              float alpha_hastings, float sigma2_app,
              int& count, int idx, int max_nspix,
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
                                       npix,nbatch,width,height,nftrs,
                                       nspix_buffer, max_nspix,
                                       direction, alpha_hastings, sigma2_app);

  }
  return max_nspix;
}

__host__
void run_merge(const float* img, int* seg, bool* border,
               spix_params* sp_params, spix_helper* sp_helper,
               spix_helper_sm* sm_helper,
               int* sm_seg1, int* sm_seg2, int* sm_pairs,
               float alpha_hastings, float sigma2_app,
               int& count, int idx, int max_nspix,
               const int npix, const int nbatch,
               const int width, const int height,
               const int nftrs, const int nspix_buffer){

  if( idx%4 == 2){
    // fprintf(stdout,"idx,count: %d,%d\n",idx,count);
    // -- run merge --
    int direction = count%2;
    CudaCalcMergeCandidate(img, seg, border,
                           sp_params, sp_helper, sm_helper, sm_pairs,
                           npix,nbatch,width,height,nftrs,
                           nspix_buffer,direction, alpha_hastings, sigma2_app);

  }
}

__host__ void CudaCalcMergeCandidate(const float* img, int* seg, bool* border,
                                     spix_params* sp_params,spix_helper* sp_helper,
                                     spix_helper_sm* sm_helper,int* sm_pairs,
                                     const int npix, const int nbatch,
                                     const int width, const int height,
                                     const int nftrs, const int nspix_buffer,
                                     const int direction, float alpha, float sigma2_app){

    int num_block = ceil( double(npix) / double(THREADS_PER_BLOCK) ); 
    int num_block2 = ceil( double(nspix_buffer) / double(THREADS_PER_BLOCK) );
    dim3 BlockPerGrid2(num_block2,nbatch);
    dim3 ThreadPerBlock(THREADS_PER_BLOCK,1);
    dim3 BlockPerGrid(num_block,nbatch);
    float alpha_hasting_ratio = alpha;
    float a_0 = 10000;
    float b_0 = sigma2_app * (a_0) ;

    init_sm<<<BlockPerGrid2,ThreadPerBlock>>>(img,seg,sp_params,sm_helper,
                                              nspix_buffer, nbatch, width,
                                              nftrs, sm_pairs);
    // fprintf(stdout,"direction: %d\n",direction);
    calc_merge_candidate<<<BlockPerGrid,ThreadPerBlock>>>(seg,border, sm_pairs,
                                                          npix, nbatch, width,
                                                          height, direction); 
    sum_by_label_merge<<<BlockPerGrid,ThreadPerBlock>>>(img,seg,sp_params,sm_helper,
                                                        npix, nbatch, width,  nftrs);
    calc_bn_merge<<<BlockPerGrid2,ThreadPerBlock>>>(seg, sm_pairs, sp_params,
                                                    sp_helper, sm_helper,
                                                    npix, nbatch, width,
                                                    nspix_buffer, b_0);
    merge_likelihood<<<BlockPerGrid2,ThreadPerBlock>>>(img,  sm_pairs,  sp_params,
                                                       sp_helper, sm_helper,
                                                       npix, nbatch, width, nftrs,
                                                       nspix_buffer, a_0, b_0);
    calc_hasting_ratio<<<BlockPerGrid2,ThreadPerBlock>>>(img,  sm_pairs, sp_params,
                                                         sp_helper, sm_helper,
                                                         npix, nbatch, width,
                                                         nftrs, nspix_buffer,
                                                         alpha_hasting_ratio);
    calc_hasting_ratio2<<<BlockPerGrid2,ThreadPerBlock>>>(img,  sm_pairs, sp_params,
                                                          sp_helper, sm_helper,
                                                          npix, nbatch, width,
                                                          nftrs, nspix_buffer,
                                                          alpha_hasting_ratio);
    remove_sp<<<BlockPerGrid2,ThreadPerBlock>>>(sm_pairs,sp_params,
                                                sm_helper,nspix_buffer);
    merge_sp<<<BlockPerGrid,ThreadPerBlock>>>(seg,border, sm_pairs, sp_params,
                                              sm_helper, npix, nbatch, width, height);  

}





__host__ int CudaCalcSplitCandidate(const float* img, int* seg, bool* border,
                                    spix_params* sp_params,
                                    spix_helper* sp_helper,
                                    spix_helper_sm* sm_helper,
                                    int* sm_seg1, int* sm_seg2, int* sm_pairs,
                                    const int npix, const int nbatch, const int width,
                                    const int height, const int nftrs,
                                    const int nspix_buffer, int max_nspix,
                                    int direction, float alpha, float sigma2_app){

    int num_block = ceil( double(npix) / double(THREADS_PER_BLOCK) ); 
    int num_block2 = ceil( double(nspix_buffer) / double(THREADS_PER_BLOCK) );
    dim3 BlockPerGrid2(num_block2,1);
    dim3 ThreadPerBlock(THREADS_PER_BLOCK,1);
    dim3 BlockPerGrid(num_block,1);
    float alpha_hasting_ratio =  alpha;
    float a_0 = 10000;
    float b_0 = sigma2_app * (a_0) ;
    // float b_0;
    int done = 1;
    int* done_gpu;
    int* max_sp;
    cudaMalloc((void **)&max_sp, sizeof(int));
    cudaMalloc((void **)&done_gpu, sizeof(int)); 

    int distance = 1;
    cudaMemset(sm_seg1, 0, npix*sizeof(int));
    cudaMemset(sm_seg2, 0, npix*sizeof(int));
    init_sm<<<BlockPerGrid2,ThreadPerBlock>>>(img,seg,sp_params,
                                              sm_helper, nspix_buffer,
                                              nbatch, width, nftrs, sm_pairs);
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
        sp_params,sm_helper,npix,nbatch,width,nspix_buffer,
        sigma2_app, max_nspix);

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

    cudaMemcpy(&max_nspix, max_sp, sizeof(int), cudaMemcpyDeviceToHost);
    cudaFree(max_sp);
    cudaFree(done_gpu);

    return max_nspix;
}



__global__ void init_sm(const float* img, const int* seg_gpu,
                        spix_params* sp_params,
                        spix_helper_sm* sm_helper,
                        const int nspix_buffer, const int nbatch,
                        const int width,const int nftrs,int* sm_pairs) {
	int k = threadIdx.x + blockIdx.x * blockDim.x;  // the label
	if (k>=nspix_buffer) return;
	//if (sp_params[k].valid == 0) return;
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
    sm_pairs[k*2+1] = 0;
    sm_pairs[k*2] = 0;
   

}

__global__
void split_marginal_likelihood(spix_params* sp_params,
                               spix_helper_sm* sm_helper,
                               const int npix, const int nbatch,
                               const int width, const int nspix_buffer,
                               float sigma2_app, int max_nspix){

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

    float3 mu_pr_k = sp_params[k].prior_mu_app;
    float3 mu_pr_f = mu_pr_k;
    sp_params[s].prior_mu_app.x = 0;
    sp_params[s].prior_mu_app.y = 0;
    sp_params[s].prior_mu_app.z = 0;
    float3 mu_pr_s = sp_params[s].prior_mu_app;

    sp_params[s].prior_mu_app_count = 1;
    int prior_mu_app_count_s = sp_params[s].prior_mu_app_count;
    int prior_mu_app_count_k = sp_params[k].prior_mu_app_count;
    int prior_mu_app_count_f = prior_mu_app_count_k;

    double3 sum_s = sm_helper[s].sum_app;
    double3 sum_k = sm_helper[k].sum_app;
    double3 sum_f;
    sum_f.x = sum_s.x + sum_k.x;
    sum_f.y = sum_s.y + sum_k.y;
    sum_f.z = sum_s.z + sum_k.z;

    double3 sq_sum_s = sm_helper[s].sum_app;
    double3 sq_sum_k = sm_helper[k].sum_app;
    double3 sq_sum_f;
    sq_sum_f.x = sq_sum_s.x + sq_sum_k.x;
    sq_sum_f.y = sq_sum_s.y + sq_sum_k.y;
    sq_sum_f.z = sq_sum_s.z + sq_sum_k.z;

    double lprob_k = marginal_likelihood_app(sum_k,sq_sum_k,count_k,sigma2_app);
    double lprob_s = marginal_likelihood_app(sum_s,sq_sum_s,count_s,sigma2_app);
    double lprob_f = marginal_likelihood_app(sum_f,sq_sum_f,count_f,sigma2_app);

    // -- write --
    sm_helper[k].numerator_app = lprob_k;
    sm_helper[s].numerator_app = lprob_s;
    sm_helper[k].numerator_f_app = lprob_f;


}

__device__ double marginal_likelihood_app(double3 sum_obs,double3 sq_sum_obs,
                                          int _num_obs, double sigma2) {
  // ref: from https://www.cs.ubc.ca/~murphyk/Papers/bayesGauss.pdf
  // Equation 55 with modifications from Eq 57 where kappa = 1
  double tau2 = 2; // ~= mean has 95% prob to be within (-1,1)
  float num_obs = (float)_num_obs;

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
    if (W>0 && C!=W){
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
    *max_sp = max_nspix+1;
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
	atomicAdd(&sm_helper[k].sq_sum_app.x, l*l);
	atomicAdd(&sm_helper[k].sq_sum_app.y, a*a);
	atomicAdd(&sm_helper[k].sq_sum_app.z,b*b);
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

__global__ void calc_bn_merge(int* seg, int* sm_pairs,
                              spix_params* sp_params,
                              spix_helper* sp_helper,
                              spix_helper_sm* sm_helper,
                              const int npix, const int nbatch,
                              const int width, const int nspix_buffer, float b_0) {

    // todo -- add nbatch
	// getting the index of the pixel
	int k = threadIdx.x + blockIdx.x * blockDim.x;  // the label
	if (k>=nspix_buffer) return;
	if (sp_params[k].valid == 0) return;

    // TODO: check if there is no neigh
    //get the label of neigh
    int f = sm_pairs[2*k+1];
	//if (sp_params[f].valid == 0) return;
    //if (f<=0) return;

    float count_f = __ldg(&sp_params[f].count);
    float count_k = __ldg(&sp_params[k].count);

    // float squares_f_x = __ldg(&sm_helper[f].sq_sum_app.x);
    // float squares_f_y = __ldg(&sm_helper[f].sq_sum_app.y);
    // float squares_f_z = __ldg(&sm_helper[f].sq_sum_app.z);
   
    // float squares_k_x = __ldg(&sm_helper[k].sq_sum_app.x);
    // float squares_k_y = __ldg(&sm_helper[k].sq_sum_app.y);
    // float squares_k_z = __ldg(&sm_helper[k].sq_sum_app.z);
   
    float squares_f_x = __ldg(&sp_helper[f].sq_sum_app.x);
    float squares_f_y = __ldg(&sp_helper[f].sq_sum_app.y);
    float squares_f_z = __ldg(&sp_helper[f].sq_sum_app.z);
   
    float squares_k_x = __ldg(&sp_helper[k].sq_sum_app.x);
    float squares_k_y = __ldg(&sp_helper[k].sq_sum_app.y);
    float squares_k_z = __ldg(&sp_helper[k].sq_sum_app.z);

    float mu_f_x = __ldg(&sp_helper[f].sum_app.x);
    float mu_f_y = __ldg(&sp_helper[f].sum_app.y);
    float mu_f_z = __ldg(&sp_helper[f].sum_app.z);
   
    float mu_k_x = __ldg(&sp_helper[k].sum_app.x);
    float mu_k_y = __ldg(&sp_helper[k].sum_app.y);
    float mu_k_z = __ldg(&sp_helper[k].sum_app.z);
    //if ((k==105)||(k==42)) printf("Merger:  %d, %d ,sq_x: %f , sq_y: %f , sq_z: %f\n", k, f,squares_k_x, squares_k_y, squares_k_z) ;   


    int count_fk = count_f + count_k;
    sm_helper[k].count_f = count_fk;
    sm_helper[k].b_n_app.x = b_0 + 0.5 * ((squares_k_x) -( mu_k_x*mu_k_x/count_k));
    sm_helper[k].b_n_f_app.x = b_0 + 0.5 *( (squares_k_x+squares_f_x) -
                                        ( (mu_f_x + mu_k_x ) * (mu_f_x + mu_k_x ) /
                                          (count_fk)));
    sm_helper[k].b_n_app.y = b_0 + 0.5 * ((squares_k_y) -( mu_k_y*mu_k_y/count_k));
    sm_helper[k].b_n_f_app.y = b_0 + 0.5 *( (squares_k_y+squares_f_y) -
                                ( (mu_f_y + mu_k_y ) * (mu_f_y + mu_k_y ) /
                                (count_fk)));
    sm_helper[k].b_n_app.z = b_0 + 0.5 * ((squares_k_z) -( mu_k_z*mu_k_z/count_k));
    sm_helper[k].b_n_f_app.z = b_0 + 0.5 *( (squares_k_z+squares_f_z) -
                                        ( (mu_f_z + mu_k_z ) * (mu_f_z + mu_k_z ) /
                                          (count_fk)));

    if(  sm_helper[k].b_n_app.x<0)   sm_helper[k].b_n_app.x = 0.1;
    if(  sm_helper[k].b_n_app.y<0)   sm_helper[k].b_n_app.y = 0.1;
    if(  sm_helper[k].b_n_app.z<0)   sm_helper[k].b_n_app.z = 0.1;

    if(  sm_helper[k].b_n_f_app.x<0)   sm_helper[k].b_n_f_app.x = 0.1;
    if(  sm_helper[k].b_n_f_app.y<0)   sm_helper[k].b_n_f_app.y = 0.1;
    if(  sm_helper[k].b_n_f_app.z<0)   sm_helper[k].b_n_f_app.z = 0.1;

}

__global__ void calc_bn_split(int* sm_pairs,
                              spix_params* sp_params,
                              spix_helper* sp_helper,
                              spix_helper_sm* sm_helper,
                              const int npix, const int nbatch,
                              const int width, const int nspix_buffer,
                              float b_0, int max_nspix) {
  // todo; -- add nbatch
	// getting the index of the pixel
	int k = threadIdx.x + blockIdx.x * blockDim.x;  // the label
	if (k>=nspix_buffer) return;
	if (sp_params[k].valid == 0) return;
    // TODO: check if there is no neigh
    //get the label of neigh
    int s = k + max_nspix;
	if (s>=nspix_buffer) return;
    float count_f = __ldg(&sp_params[k].count);
    float count_k = __ldg(&sm_helper[k].count);
    float count_s = __ldg(&sm_helper[s].count);
    if((count_f<1)||( count_k<1)||(count_s<1)) return;

    float squares_s_x = __ldg(&sm_helper[s].sq_sum_app.x);
    float squares_s_y = __ldg(&sm_helper[s].sq_sum_app.y);
    float squares_s_z = __ldg(&sm_helper[s].sq_sum_app.z);
   
    float squares_k_x = __ldg(&sm_helper[k].sq_sum_app.x);
    float squares_k_y = __ldg(&sm_helper[k].sq_sum_app.y);
    float squares_k_z = __ldg(&sm_helper[k].sq_sum_app.z);
   
    float mu_s_x = __ldg(&sm_helper[s].sum_app.x);
    float mu_s_y = __ldg(&sm_helper[s].sum_app.y);
    float mu_s_z = __ldg(&sm_helper[s].sum_app.z);

    float mu_k_x = __ldg(&sm_helper[k].sum_app.x);
    float mu_k_y = __ldg(&sm_helper[k].sum_app.y);
    float mu_k_z = __ldg(&sm_helper[k].sum_app.z);

    // -- this is correct; its the "helper" associated with "sp_params" --
    float mu_f_x =__ldg(&sp_helper[k].sum_app.x);
    float mu_f_y = __ldg(&sp_helper[k].sum_app.y);
    float mu_f_z = __ldg(&sp_helper[k].sum_app.z);

    // -- update b_n = b_0 + ... in Supp. --
    sm_helper[k].b_n_app.x = b_0 + 0.5 * ((squares_k_x) - ( mu_k_x*mu_k_x/ count_k));
    sm_helper[k].b_n_app.y = b_0 + 0.5 * ((squares_k_y) - ( mu_k_y*mu_k_y/ count_k));
    sm_helper[k].b_n_app.z = b_0 + 0.5 * ((squares_k_z) - ( mu_k_z*mu_k_z/ count_k));
    sm_helper[s].b_n_app.x = b_0 + 0.5 * ((squares_s_x) - ( mu_s_x*mu_s_x/ count_s));
    sm_helper[s].b_n_app.y = b_0 + 0.5 * ((squares_s_y) - ( mu_s_y*mu_s_y/ count_s));
    sm_helper[s].b_n_app.z = b_0 + 0.5 * ((squares_s_z) - ( mu_s_z*mu_s_z/ count_s));
    sm_helper[k].b_n_f_app.x=b_0+0.5*((squares_k_x+squares_s_x)-(mu_f_x*mu_f_x/count_f)); 
    sm_helper[k].b_n_f_app.y =b_0+0.5*((squares_k_y+squares_s_y)-(mu_f_y*mu_f_y/count_f)); 
    sm_helper[k].b_n_f_app.z =b_0+0.5 * ((squares_k_z+squares_s_z) -
                                ( mu_f_z*mu_f_z/ count_f));

}




__global__
void split_likelihood(const float* img, int* sm_pairs,
                      spix_params* sp_params,
                      spix_helper* sp_helper,
                      spix_helper_sm* sm_helper,
                      const int npix, const int nbatch,
                      const int width, const int nftrs,
                      const int nspix_buffer,
                      float a_0, float b_0, int max_nspix) {
  // todo -- add nbatch
	// getting the index of the pixel
	int k = threadIdx.x + blockIdx.x * blockDim.x;  // the label
    if (k>=nspix_buffer) return;
	if (sp_params[k].valid == 0) return;


    int s = k + max_nspix;
    if (s>=nspix_buffer) return;
    float count_f = __ldg(&sp_params[k].count);
    float count_k = __ldg(&sm_helper[k].count);
    float count_s = __ldg(&sm_helper[s].count);

    if((count_f<1)||( count_k<1)||(count_s<1)) return;
    if (count_f!=count_k+count_s) return;
    // TODO: check if there is no neigh
    // TODO: check if num is the same
	//get the label
    //a_0 = 1100*(count_f);

    float a_n_k = a_0+float(count_k)/2;
    float a_n_s = a_0+float(count_s)/2;
    float a_n_f = a_0+float(count_f)/2;

    float v_n_k = 1/float(count_k);
    float v_n_s = 1/float(count_s);
    float v_n_f = 1/float(count_f);

    float b_n_k_x = __ldg(&sm_helper[k].b_n_app.x);
    float b_n_k_y = __ldg(&sm_helper[k].b_n_app.y);
    float b_n_k_z = __ldg(&sm_helper[k].b_n_app.z);

    float b_n_s_x = __ldg(&sm_helper[s].b_n_app.x);
    float b_n_s_y = __ldg(&sm_helper[s].b_n_app.y);
    float b_n_s_z = __ldg(&sm_helper[s].b_n_app.z);

    float b_n_f_x = __ldg(&sm_helper[k].b_n_f_app.x);
    float b_n_f_y = __ldg(&sm_helper[k].b_n_f_app.y);
    float b_n_f_z = __ldg(&sm_helper[k].b_n_f_app.z);

    // why use this as a_0? This seems wrong.
    a_0 = a_n_k;
    // sm_helper[k].numerator.x = a_0 * __logf(b_0) + lgammaf(a_n_k)+ 0.5*__logf(v_n_k);
    //sm_helper[k].numerator_app=a_0 * __logf(b_0) + lgammaf(a_n_k)+ 0.5*__logf(count_k);
    sm_helper[k].numerator_app = a_0 * __logf(b_0) + lgammaf(a_n_k)+ 0.5*__logf(v_n_k);

    sm_helper[k].denominator.x = a_n_k * __logf (b_n_k_x) + \
      0.5 * count_k * __logf (M_PI) + count_k * __logf (2) + lgammaf(a_0);

    sm_helper[k].denominator.y = a_n_k * __logf (b_n_k_y) + \
      0.5 * count_k * __logf (M_PI) + count_k * __logf (2) + lgammaf(a_0);

    sm_helper[k].denominator.z = a_n_k * __logf (b_n_k_z) + \
      0.5 * count_k * __logf (M_PI) + count_k * __logf (2) + lgammaf(a_0);


    a_0 = a_n_s;
    // sm_helper[s].numerator.x = a_0 * __logf(b_0) + lgammaf(a_n_s)+0.5*__logf(v_n_s);
    // sm_helper[s].numerator_app=a_0 * __logf(b_0) + lgammaf(a_n_s)+0.5*__logf(count_s);
    sm_helper[s].numerator_app=a_0 * __logf(b_0) + lgammaf(a_n_s)+0.5*__logf(v_n_s);
    sm_helper[s].denominator.x = a_n_s * __logf (b_n_s_x) + \
      0.5 * count_s * __logf (M_PI) + count_s * __logf (2) + lgammaf(a_0);

    sm_helper[s].denominator.y = a_n_s * __logf (b_n_s_y) + \
      0.5 * count_s * __logf (M_PI) + count_s * __logf (2) + lgammaf(a_0);

    sm_helper[s].denominator.z = a_n_s * __logf (b_n_s_z) + \
      0.5 * count_s * __logf (M_PI) + count_s * __logf (2) + lgammaf(a_0);      


    a_0 =a_n_f;
    // sm_helper[k].numerator_f_app.x =a_0*__logf(b_0)+lgammaf(a_n_f)+0.5*__logf(v_n_f);
    // sm_helper[k].numerator_f_app =a_0*__logf(b_0)+lgammaf(a_n_f)+0.5*__logf(count_f);
    sm_helper[s].numerator_f_app=a_0 * __logf(b_0) + lgammaf(a_n_f)+0.5*__logf(v_n_f);
    sm_helper[k].denominator_f.x = a_n_f * __logf (b_n_f_x) + \
      0.5 * count_f * __logf (M_PI) + count_f * __logf (2) + lgammaf(a_0);

    sm_helper[k].denominator_f.y = a_n_f * __logf (b_n_f_y) + \
      0.5 * count_f * __logf (M_PI) + count_f * __logf (2) + lgammaf(a_0);

    sm_helper[k].denominator_f.z = a_n_f * __logf (b_n_f_z) + \
      0.5 * count_f * __logf (M_PI) + count_f * __logf (2) + lgammaf(a_0);        

}   


__global__
void merge_likelihood(const float* img, int* sm_pairs,
                      spix_params* sp_params,
                      spix_helper* sp_helper,
                      spix_helper_sm* sm_helper,
                      const int npix, const int nbatch,
                      const int width, const int nftrs,
                      const int nspix_buffer,
                      float a_0, float b_0) {

	// -- getting the index of the pixel --
	int k = threadIdx.x + blockIdx.x * blockDim.x;  // the label
    if (k>=nspix_buffer) return;
	if (sp_params[k].valid == 0) return;

    // -- counts --
    float count_k = __ldg(&sp_params[k].count);
    float count_f = __ldg(&sm_helper[k].count_f);

    // -- counts --
    float a_n = a_0 + float(count_k) / 2;
    float a_n_f = a_0+ float(count_f) / 2;
    // float v_n = 1 / float(num_pixels_in_sp);
    float v_n = 1/float(count_k);
    float v_n_f = 1/float(count_f);

    // -- update numer/denom --
    a_0 = a_n;
    sm_helper[k].numerator_app = a_0 * __logf(b_0) + lgammaf(a_n)+0.5*__logf(v_n);
    sm_helper[k].denominator.x = a_n* __logf ( __ldg(&sm_helper[k].b_n_app.x)) + 0.5 * count_k * __logf (M_PI) + count_k * __logf (2) + lgammaf(a_0);
    sm_helper[k].denominator.y = a_n* __logf ( __ldg(&sm_helper[k].b_n_app.y)) + 0.5 * count_k * __logf (M_PI) + count_k * __logf (2) + lgamma(a_0);
    sm_helper[k].denominator.z = a_n* __logf(__ldg(&sm_helper[k].b_n_app.z)) + 0.5 * count_k * __logf (M_PI) + count_k * __logf (2) + lgammaf(a_0);
    
    // -- update numer/denom --
    a_0 = a_n_f;
    sm_helper[k].numerator_f_app = a_0 * __logf (b_0) + lgammaf(a_n_f)+0.5*__logf(v_n_f);
    sm_helper[k].denominator_f.x = a_n_f* __logf (__ldg(&sm_helper[k].b_n_f_app.x)) + 0.5 * count_f * __logf (M_PI) + count_f * __logf (2) + lgammaf(a_0);
    sm_helper[k].denominator_f.y = a_n_f* __logf (__ldg(&sm_helper[k].b_n_f_app.y)) + 0.5 * count_f * __logf (M_PI) + count_f * __logf (2) + lgammaf(a_0);
    sm_helper[k].denominator_f.z = a_n_f* __logf (__ldg(&sm_helper[k].b_n_f_app.z)) + 0.5 * count_f* __logf (M_PI) + count_f * __logf (2) + lgammaf(a_0);         

}   


__global__ void calc_hasting_ratio(const float* img, int* sm_pairs,
                                   spix_params* sp_params,
                                   spix_helper* sp_helper,
                                   spix_helper_sm* sm_helper,
                                   const int npix, const int nbatch, const int width,
                                   const int nftrs, const int nspix_buffer,
                                   float log_alpha_hasting_ratio) {

	// getting the index of the pixel
	int k = threadIdx.x + blockIdx.x * blockDim.x;  // the label

	if (k>=nspix_buffer) return;
	if (sp_params[k].valid == 0) return;

    int f = sm_pairs[2*k+1];
	if (sp_params[f].valid == 0) return;
    if(f<=0) return;

    float count_k = __ldg(&sp_params[k].count);
    float count_f = __ldg(&sm_helper[k].count_f);
    
    if ((count_k<1)||(count_f<1)) return;

    sm_helper[k].merge = false;
    float num_k = __ldg(&sm_helper[k].numerator_app);

    float total_marginal_1 = (num_k - __ldg(&sm_helper[k].denominator.x)) + 
      (num_k - __ldg(&sm_helper[k].denominator.y)) +
      (num_k - __ldg(&sm_helper[k].denominator.z)); 

    float num_f = __ldg(&sm_helper[f].numerator_app);

    float total_marginal_2 = (num_f - __ldg(&sm_helper[f].denominator.x)) + 
      (num_f - __ldg(&sm_helper[f].denominator.y)) +
      (num_f - __ldg(&sm_helper[f].denominator.z));

    float num_kf = __ldg(&sm_helper[k].numerator_f_app);

    float total_marginal_f = (num_kf - __ldg(&sm_helper[k].denominator_f.x)) +   
      (num_kf - __ldg(&sm_helper[k].denominator_f.y)) + 
      (num_kf - __ldg(&sm_helper[k].denominator_f.z));

    
    double alpha_hasting_ratio = exp(log_alpha_hasting_ratio);
    float log_nominator = lgammaf(count_f) + total_marginal_f +
      lgammaf(alpha_hasting_ratio) + lgammaf(alpha_hasting_ratio / 2 + count_k) +
      lgammaf(alpha_hasting_ratio / 2 + count_f -  count_k);

   float log_denominator = __logf(alpha_hasting_ratio) + lgammaf(count_k) +
     lgammaf(count_f -  count_k) + total_marginal_1 + 
     total_marginal_2 + lgammaf(alpha_hasting_ratio + count_f) +
     lgammaf(alpha_hasting_ratio / 2) + lgammaf(alpha_hasting_ratio / 2);

    log_denominator = __logf(alpha_hasting_ratio) + total_marginal_1 + total_marginal_2;
    log_nominator = total_marginal_f ;

    sm_helper[k].hasting = log_nominator - log_denominator;

    return;
}


__global__ void calc_hasting_ratio2(const float* img, int* sm_pairs,
                                    spix_params* sp_params,
                                    spix_helper* sp_helper,
                                    spix_helper_sm* sm_helper,
                                    const int npix, const int nbatch, const int width,
                                    const int nftrs, const int nspix_buffer,
                                    float alpha_hasting_ratio) {
  // todo -- add nbatch and sftrs
	// getting the index of the pixel
	int k = threadIdx.x + blockIdx.x * blockDim.x;  // the label

	if (k>=nspix_buffer) return;
	if (sp_params[k].valid == 0) return;

    int f = sm_pairs[2*k+1];
	if (sp_params[f].valid == 0) return;
    if(f<=0) return;
    if((sm_helper[k].hasting ) > -2)
    // if((sm_helper[k].hasting ) > 0)
    {
      //printf("Want to merge k: %d, f: %d, splitmerge k %d, splitmerge  f %d, %d\n", k, f, sm_pairs[2*k], sm_pairs[2*f], sm_pairs[2*f+1] );
      int curr_max = atomicMax(&sm_pairs[2*f],k);
      if( curr_max == 0){
        sm_helper[k].merge = true;
      }else{
        sm_pairs[2*f] = curr_max;
      }
    }
         
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
                          float log_alpha_hasting_ratio,
                          int max_nspix, int* max_sp ) {
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

    // float num_k = __ldg(&sm_helper[k].numerator_app);
    // float num_s = __ldg(&sm_helper[s].numerator_app);
    // float num_f = __ldg(&sm_helper[k].numerator_f_app);
    
    // float total_marginal_k = (num_k - __ldg(&sm_helper[k].denominator.x)) +  
    //                      (num_k - __ldg(&sm_helper[k].denominator.y)) + 
    //                      (num_k - __ldg(&sm_helper[k].denominator.z)); 

    // float total_marginal_s = (num_s - __ldg(&sm_helper[s].denominator.x)) +  
    //                      (num_s - __ldg(&sm_helper[s].denominator.y)) + 
    //                      (num_s - __ldg(&sm_helper[s].denominator.z)); 

    // float total_marginal_f = (num_f - __ldg(&sm_helper[k].denominator_f.x)) +  
    //                      (num_f - __ldg(&sm_helper[k].denominator_f.y)) + 
    //                      (num_f - __ldg(&sm_helper[k].denominator_f.z)); 

 
     //printf("hasating:x k: %d, count: %f, den: %f, %f, %f, b_n: %f, %f, %f, num: %f \n",k, count_k,  sm_helper[k].denominator.x, sm_helper[k].denominator.y,  sm_helper[k].denominator.z,   __logf (sm_helper[k].b_n_app.x) ,  __logf (sm_helper[k].b_n_app.y),   __logf (sm_helper[k].b_n_app.z), sm_helper[k].numerator.x);

    // float log_nominator = __logf(alpha_hasting_ratio)+ lgammaf(count_k)\
    //   + total_marginal_k + lgammaf(count_s) + total_marginal_s;
    // log_nominator = total_marginal_k + total_marginal_s;
    // float log_denominator = lgammaf(count_f) + total_marginal_f;
    // log_denominator =total_marginal_f;
    // sm_helper[k].hasting = log_nominator - log_denominator;


    float lprob_k = __ldg(&sm_helper[k].numerator_app);
    float lprob_s = __ldg(&sm_helper[s].numerator_app);
    float lprob_f = __ldg(&sm_helper[k].numerator_f_app);

    float log_nominator = log_alpha_hasting_ratio\
      + lgammaf(count_k) +  lgammaf(count_s) + lprob_k + lprob_s;
    float log_denominator = lgammaf(count_f) + lprob_f;
    sm_helper[k].hasting = log_nominator - log_denominator;

    // ".merge" is merely a bool variable; nothing about merging here. only splitting
    // sm_helper[k].merge = (sm_helper[k].hasting > -2); // why "-2"?
    // sm_helper[s].merge = (sm_helper[k].hasting > -2);

    sm_helper[k].merge = (sm_helper[k].hasting > 0); // why "-2"?
    sm_helper[s].merge = (sm_helper[k].hasting > 0);


    if((sm_helper[k].merge)) // split step
      {

        s = atomicAdd(max_sp,1) +1; // ? can't multiple splits happen at one time? yes :D
        sm_pairs[2*k] = s;
        // -- update shape prior --

        int prior_count = sp_params[k].prior_count/2;
        sp_params[k].prior_count = prior_count;

        // sp_params[k].prior_sigma_shape.x/=2;
        // sp_params[k].prior_sigma_shape.y/=2;
        // sp_params[k].prior_sigma_shape.z/=2;

        sp_params[k].prior_sigma_shape.x = prior_count*prior_count;
        sp_params[k].prior_sigma_shape.y = 0;
        sp_params[k].prior_sigma_shape.z = prior_count*prior_count;

        
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
    if(sm_helper[k].remove)
    seg[idx] =  f;

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
    if ((sm_helper[k].merge == false)||sm_helper[k2].merge == false){
      return;
    }

    if(sm_seg1[idx]==k2) seg[idx] = sm_pairs[2*k];
    //seg[idx] = sm_seg1[idx];
    //printf("Add the following: %d - %d'\n", k,sm_pairs[2*k]);
    sp_params[sm_pairs[2*k]].valid = 1;
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
    int f = sm_pairs[2*k+1];
    if ((sp_params[k].valid == 0)||(sp_params[f].valid == 0)) return;    
    if(f<=0) return;
    // if ((sm_helper[k].merge == true) && (sm_helper[f].merge == false) && (sm_pairs[2*f]==k) )
    if ((sm_helper[k].merge == true) && (sm_helper[f].merge == false))
      {
        sm_helper[k].remove=true;
        sp_params[k].valid =0;

        // -- update priors --
        sp_params[f].prior_count =sp_params[k].prior_count+sp_params[f].prior_count;
        sp_params[f].prior_sigma_shape.x+= sp_params[k].prior_sigma_shape.x;
        sp_params[f].prior_sigma_shape.y+= sp_params[k].prior_sigma_shape.y;
        sp_params[f].prior_sigma_shape.z+= sp_params[k].prior_sigma_shape.z;
        // sp_params[k].prior_sigma_shape.x/=2;
        // sp_params[k].prior_sigma_shape.y/=2;
        // sp_params[k].prior_sigma_shape.z/=2;

      }
    else
      {
        sm_helper[k].remove=false;
      }
    
    return;
    
}


