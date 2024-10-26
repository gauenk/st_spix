
/******************************************************


       -> Merging is Okay for Two Unconditioned Nodes


******************************************************/

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

#include "split_merge_prop.h"
// #include "update_params.h"

#include <stdio.h>
#ifndef WIN32
#include <unistd.h>
#endif

__host__
int run_split_prop(const float* img, int* seg, bool* border,
                   spix_params* sp_params, spix_helper* sp_helper,
                   spix_helper_sm* sm_helper,
                   int* sm_seg1 ,int* sm_seg2, int* sm_pairs,
                   float alpha_hastings, float sigma2_app,
                   int& count, int idx, int max_nspix,
                   const int npix, const int nbatch,
                   const int width, const int height,
                   const int nftrs, const int nspix_buffer){

  // -- (left/right,up/down,new/old) --
  count += 1;
  int split_direction = count%2+1;
  int oldnew_choice = count%4;

  // -- run split --
  max_nspix = CudaCalcSplitCandidate_p(img, seg, border,
                                       sp_params, sp_helper, sm_helper,
                                       sm_seg1, sm_seg2, sm_pairs,
                                       npix,nbatch,width,height,nftrs,
                                       nspix_buffer, max_nspix,
                                       split_direction, oldnew_choice,
                                       alpha_hastings, sigma2_app);

  return max_nspix;
}

__host__
void run_merge_prop(const float* img, int* seg, bool* border,
               spix_params* sp_params, spix_helper* sp_helper,
               spix_helper_sm* sm_helper,
               int* sm_seg1, int* sm_seg2, int* sm_pairs,
               float alpha_hastings, float sigma2_app,
                    int& count, int idx, int max_nspix, // int nspix,
               const int npix, const int nbatch,
               const int width, const int height,
               const int nftrs, const int nspix_buffer){

  if( idx%4 == 2){
    // fprintf(stdout,"idx,count: %d,%d\n",idx,count);
    // -- run merge --
    int direction = count%2;
    CudaCalcMergeCandidate_p(img, seg, border,
                           sp_params, sp_helper, sm_helper, sm_pairs,
                             npix,nbatch,width,height,nftrs, 
                           nspix_buffer,// nspix, 
                             direction, alpha_hastings, sigma2_app);

  }
}

__host__ void CudaCalcMergeCandidate_p(const float* img, int* seg, bool* border,
                                       spix_params* sp_params,spix_helper* sp_helper,
                                       spix_helper_sm* sm_helper,int* sm_pairs,
                                       const int npix, const int nbatch,
                                       const int width, const int height,
                                       const int nftrs, const int nspix_buffer,
                                       // const int nspix,
                                       const int direction,float alpha, float sigma2_app){

    int num_block = ceil( double(npix) / double(THREADS_PER_BLOCK) ); 
    int num_block2 = ceil( double(nspix_buffer) / double(THREADS_PER_BLOCK) );
    dim3 BlockPerGrid2(num_block2,nbatch);
    dim3 ThreadPerBlock(THREADS_PER_BLOCK,1);
    dim3 BlockPerGrid(num_block,nbatch);
    float alpha_hasting_ratio = alpha;
    float a_0 = 10000;
    float b_0 = sigma2_app * (a_0) ;

// __global__ void init_sm_p(const float* img, const int* seg_gpu,
//                           spix_params* sp_params,
//                           spix_helper_sm* sm_helper, bool* split_dir,
//                           const int nspix_buffer, const int nbatch,
//                           const int width,const int nftrs,int* sm_pairs) {

    bool* split_dir;
    cudaMalloc((void **)&split_dir, nspix_buffer*sizeof(bool)); 

    init_sm_p<<<BlockPerGrid2,ThreadPerBlock>>>(img,seg,sp_params,sm_helper,
                                                split_dir,nspix_buffer, nbatch, width,
                                                nftrs, sm_pairs);

    // fprintf(stdout,"direction: %d\n",direction);
    calc_merge_candidate_p<<<BlockPerGrid,ThreadPerBlock>>>(seg,border, sm_pairs,
                                                            npix, nbatch, width,
                                                            height, direction);// nspix,
    sum_by_label_merge_p<<<BlockPerGrid,ThreadPerBlock>>>(img,seg,sp_params,sm_helper,
                                                        npix, nbatch, width,  nftrs);
    calc_bn_merge_p<<<BlockPerGrid2,ThreadPerBlock>>>(seg, sm_pairs, sp_params,
                                                      sp_helper, sm_helper,
                                                      npix, nbatch, width,
                                                      nspix_buffer, b_0);
    merge_likelihood_p<<<BlockPerGrid2,ThreadPerBlock>>>(img,  sm_pairs,  sp_params,
                                                       sp_helper, sm_helper,
                                                       npix, nbatch, width, nftrs,
                                                       nspix_buffer, a_0, b_0);
    calc_hasting_ratio_p<<<BlockPerGrid2,ThreadPerBlock>>>(img,  sm_pairs, sp_params,
                                                         sp_helper, sm_helper,
                                                         npix, nbatch, width,
                                                         nftrs, nspix_buffer,
                                                         alpha_hasting_ratio);
    calc_hasting_ratio2_p<<<BlockPerGrid2,ThreadPerBlock>>>(img,  sm_pairs, sp_params,
                                                          sp_helper, sm_helper,
                                                          npix, nbatch, width,
                                                          nftrs, nspix_buffer,
                                                          alpha_hasting_ratio);
    remove_sp_p<<<BlockPerGrid2,ThreadPerBlock>>>(sm_pairs,sp_params,
                                                sm_helper,nspix_buffer);
    merge_sp_p<<<BlockPerGrid,ThreadPerBlock>>>(seg,border, sm_pairs, sp_params,
                                              sm_helper, npix, nbatch, width, height);  

    cudaFree(split_dir);

}





__host__ int CudaCalcSplitCandidate_p(const float* img, int* seg, bool* border,
                                      spix_params* sp_params, spix_helper* sp_helper,
                                      spix_helper_sm* sm_helper,
                                      int* sm_seg1, int* sm_seg2, int* sm_pairs,
                                      const int npix, const int nbatch, const int width,
                                      const int height, const int nftrs,
                                      const int nspix_buffer, int max_nspix,
                                      int direction, int oldnew_choice,
                                      float alpha, float sigma2_app){

    int num_block = ceil( double(npix) / double(THREADS_PER_BLOCK) ); 
    int num_block2 = ceil( double(nspix_buffer) / double(THREADS_PER_BLOCK) );
    dim3 BlockPerGrid2(num_block2,1);
    dim3 ThreadPerBlock(THREADS_PER_BLOCK,1);
    dim3 BlockPerGrid(num_block,1);
    float alpha_hasting_ratio =  alpha;
    // float a_0 = 10000;
    // float b_0 = sigma2_app * (a_0) ;
    // float b_0;
    int done = 1;
    int* done_gpu;
    int* max_sp;
    cudaMalloc((void **)&max_sp, sizeof(int));
    cudaMalloc((void **)&done_gpu, sizeof(int)); 
    bool* split_dir;
    cudaMalloc((void **)&split_dir, nspix_buffer*sizeof(bool)); 
    int distance = 1;
    cudaMemset(sm_seg1, 0, npix*sizeof(int));
    cudaMemset(sm_seg2, 0, npix*sizeof(int));


    direction = 1;
    // -- splits --
    init_sm_p<<<BlockPerGrid2,ThreadPerBlock>>>(img,seg,sp_params,sm_helper,
                                                split_dir,nspix_buffer,
                                                nbatch, width, nftrs, sm_pairs);
    init_split_p<<<BlockPerGrid2,ThreadPerBlock>>>(border,sm_seg1,sp_params,
                                                   sm_helper, split_dir, nspix_buffer,
                                                   nbatch, width, height, direction,
                                                   seg, max_sp, max_nspix);
    init_split_p<<<BlockPerGrid2,ThreadPerBlock>>>(border,sm_seg2,sp_params,
                                                   sm_helper, split_dir, nspix_buffer,
                                                   nbatch, width,height, -direction,
                                                   seg, max_sp, max_nspix);

    // idk what "split_sp" is doing here; init_sm clears the merge fields and
    // so the function returns immediately...
    // split_sp_p<<<BlockPerGrid,ThreadPerBlock>>>(seg,sm_seg1,sm_pairs,
    //                                             sp_params, sm_helper, npix,
    //                                             nbatch, width, height, max_nspix);

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
    // previous it's a sum of distances from the center; now its a new label
    calc_seg_split_p<<<BlockPerGrid,ThreadPerBlock>>>(sm_seg1,sm_seg2, seg,
                                                      oldnew_choice, npix,
                                                      nbatch, max_nspix);
    // std::string fname_split1_post = "split1_post";
    // write_tensor_to_file_v2(sm_seg1,height,width,fname_split1_post);

    // computes summaries stats for each split
    sum_by_label_split_p<<<BlockPerGrid,ThreadPerBlock>>>(img, sm_seg1, sp_params,
                                                          sm_helper, npix, nbatch,
                                                          width,nftrs,max_nspix);
    split_marginal_likelihood_p<<<BlockPerGrid2,ThreadPerBlock>>>(\
        sp_params,sm_helper,npix,nbatch,width,nspix_buffer,
        sigma2_app, max_nspix);

    // calc_bn_split_p<<<BlockPerGrid2,ThreadPerBlock>>>(sm_pairs, sp_params, sp_helper,
    //                                                   sm_helper, oldnew_choice,
    //                                                   npix, nbatch, width,
    //                                                   nspix_buffer, b_0,
    //                                                   sigma2_app, max_nspix);

    // gpuErrchk( cudaPeekAtLastError() );
    // gpuErrchk( cudaDeviceSynchronize() );
    // split_likelihood_p<<<BlockPerGrid2,ThreadPerBlock>>>(\
    //                          img,  sm_pairs,
    //                          sp_params,  sp_helper, sm_helper,
    //                          npix, nbatch, width, nftrs,
    //                          nspix_buffer, a_0, b_0, max_nspix);

    // gpuErrchk( cudaPeekAtLastError() );
    // gpuErrchk( cudaDeviceSynchronize() );

    // fprintf(stdout,"[s_m.cu] max_nspix: %d\n",max_nspix);
    split_hastings_ratio_p<<<BlockPerGrid2,ThreadPerBlock>>>(img, sm_pairs, sp_params,
                                                           sp_helper, sm_helper,
                                                             split_dir,
                                                           npix, nbatch, width, nftrs,
                                                           nspix_buffer,
                                                           alpha_hasting_ratio,
                                                           max_nspix, max_sp);

    // -- do the split --
    split_sp_p<<<BlockPerGrid,ThreadPerBlock>>>(seg,sm_seg1,sm_pairs,
                                                sp_params, sm_helper, split_dir, npix,
                                              nbatch, width, height, max_nspix);


    // gpuErrchk( cudaPeekAtLastError() );
    // gpuErrchk( cudaDeviceSynchronize() );

    cudaMemcpy(&max_nspix, max_sp, sizeof(int), cudaMemcpyDeviceToHost);
    cudaFree(split_dir);
    cudaFree(max_sp);
    cudaFree(done_gpu);

    return max_nspix;
}



__global__ void init_sm_p(const float* img, const int* seg_gpu,
                          spix_params* sp_params,
                          spix_helper_sm* sm_helper, bool* split_dir,
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

	sm_helper[k].sq_sum_app.x = 0;
	sm_helper[k].sq_sum_shape.y = 0;
    sm_helper[k].sum_app.x = 0;
	sm_helper[k].sum_shape.y = 0;

    sm_helper[k].count_f = 0;
    sm_helper[k].count = 0;
    sm_helper[k].hasting = -999999;
    //sp_params[k].count = 0;

    sm_helper[k].merge = false;
    sm_helper[k].remove = false;
    sm_pairs[k*2+1] = 0;
    sm_pairs[k*2] = 0;
   
    
    // "UpDown" if sY > sX -> inv_sX < inv_sY; 
    double inv_sX = sp_params[k].sigma_shape.x;
    double inv_sY = sp_params[k].sigma_shape.z;
    split_dir[k] = inv_sY > inv_sX; // bool for "UpDown"; o.w. "LeftRight"

}

__global__  void calc_merge_candidate_p(int* seg, bool* border, int* sm_pairs,
                                        const int npix, const int nbatch,
                                        const int width, const int height,
                                        const int direction){ // const int nspix
  

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
void calc_split_candidate_p(int* dists, int* spix, bool* border,
                          int distance, int* mutex, const int npix,
                          const int nbatch, const int width, const int height){

    // this is where the new "split_direction" lives...
  
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
                             spix_helper_sm* sm_helper,
                             bool* split_dir,
                             const int nspix_buffer,
                             const int nbatch, const int width,
                             const int height, const int direction,
                             const int* seg, int* max_sp, int max_nspix) {

    // todo: add batch -- no nftrs
	int k = threadIdx.x + blockIdx.x * blockDim.x;  // the label
	if (k>=nspix_buffer) return;
	if (sp_params[k].valid == 0) return;
    // *max_sp = max_nspix+1;
    *max_sp = max_nspix;
    int x;
    int y;
    bool UpDown = split_dir[k];

    // if((direction==1)||(direction==-1))
    if (not(UpDown))
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
                                 // sp_params* sparams,
                                 int oldnew_choice, const int npix,
                                 int nbatch, int max_nspix) {

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

__global__ void sum_by_label_split_p(const float* img, const int* seg,
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
    atomicAdd((unsigned long long *)&sm_helper[k].sq_sum_shape.x, x*x);
	atomicAdd((unsigned long long *)&sm_helper[k].sq_sum_shape.y, x*y);
	atomicAdd((unsigned long long *)&sm_helper[k].sq_sum_shape.z, y*y);

    return;
}

__global__
void split_marginal_likelihood_p(spix_params* sp_params,
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
    mu_pr_s.x = 0;
    mu_pr_s.y = 0;
    mu_pr_s.z = 0;

    int prior_count = sp_params[s].prior_count;
    int prior_count_div2 = prior_count/2;
    // sp_params[s].prior_mu_app_count = prior_count/2;
    // int prior_mu_app_count_s = sp_params[s].prior_count/2;

    // int prior_mu_app_count_s = sp_params[s].prior_mu_app_count;
    // int prior_mu_app_count_k = sp_params[k].prior_mu_app_count;
    // int prior_mu_app_count_f = prior_mu_app_count_k;

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

    // -- marginal likelihoods --
    double lprob_k_cond = marginal_likelihood_app_sm(sum_k,sq_sum_k,mu_pr_k,count_k,
                                                     prior_count_div2,sigma2_app);
    double lprob_s_ucond = marginal_likelihood_app_sm(sum_s,sq_sum_s,mu_pr_s,count_s,
                                                      prior_count_div2,sigma2_app);
    double lprob_k_ucond = marginal_likelihood_app_sm(sum_k,sq_sum_k,mu_pr_s,count_k,
                                                      prior_count_div2,sigma2_app);
    double lprob_s_cond = marginal_likelihood_app_sm(sum_s,sq_sum_s,mu_pr_k,count_s,
                                                     prior_count_div2,sigma2_app);
    double lprob_f = marginal_likelihood_app_sm(sum_f,sq_sum_f,mu_pr_f,count_f,
                                                prior_count,sigma2_app);

    // -- write --
    sm_helper[k].numerator_app = lprob_k_cond;
    sm_helper[k].b_n_shape_det = lprob_k_ucond;
    sm_helper[s].numerator_app = lprob_s_cond;
    sm_helper[s].b_n_shape_det = lprob_s_ucond;
    sm_helper[k].numerator_f_app = lprob_f;

    // sm_helper[k].numerator_app = 0.;
    // sm_helper[s].numerator_app = 0.;
    // sm_helper[k].numerator_f_app = 0.;


}



__global__
void split_likelihood_p(const float* img, int* sm_pairs,
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
    float pr_count = __ldg(&sp_params[k].prior_count);
    float count_f = __ldg(&sp_params[k].count);
    float count_k= __ldg(&sm_helper[k].count);
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

    // float v_n_k = 1/float(count_k);
    // float v_n_s = 1/float(count_s);
    // float v_n_f = 1/float(count_f);
    float v_n_k = 1 + pr_count/count_k;
    float v_n_s = 1 + pr_count/count_s;
    float v_n_f = 1 + pr_count/count_f;

    float b_n_k_x = __ldg(&sm_helper[k].b_n_app.x);
    float b_n_k_y = __ldg(&sm_helper[k].b_n_app.y);
    float b_n_k_z = __ldg(&sm_helper[k].b_n_app.z);

    float b_n_s_x = __ldg(&sm_helper[s].b_n_app.x);
    float b_n_s_y = __ldg(&sm_helper[s].b_n_app.y);
    float b_n_s_z = __ldg(&sm_helper[s].b_n_app.z);

    float b_n_f_x = __ldg(&sm_helper[k].b_n_f_app.x);
    float b_n_f_y = __ldg(&sm_helper[k].b_n_f_app.y);
    float b_n_f_z = __ldg(&sm_helper[k].b_n_f_app.z);

    /********************
  
          Appearance
   
    **********************/

    a_0 = a_n_k;
    // sm_helper[k].numerator_app = a_0*__logf(b_0) + lgammaf(a_n_k)+ 0.5*__logf(v_n_k);

    // sm_helper[k].denominator.x = a_n_k * __logf (b_n_k_x) + \
    //   0.5 * count_k * __logf (M_PI) + count_k * __logf (2) + lgammaf(a_0);
    // sm_helper[k].denominator.y = a_n_k * __logf (b_n_k_y) + \
    //   0.5 * count_k * __logf (M_PI) + count_k * __logf (2) + lgammaf(a_0);
    // sm_helper[k].denominator.z = a_n_k * __logf (b_n_k_z) + \
    //   0.5 * count_k * __logf (M_PI) + count_k * __logf (2) + lgammaf(a_0);

    a_0 = a_n_s;
    // sm_helper[s].numerator_app = a_0*__logf(b_0) + lgammaf(a_n_s)+0.5*__logf(v_n_s);

    // sm_helper[s].denominator.x = a_n_s * __logf (b_n_s_x) + \
    //   0.5 * count_s * __logf (M_PI) + count_s * __logf (2) + lgammaf(a_0);
    // sm_helper[s].denominator.y = a_n_s * __logf (b_n_s_y) + \
    //   0.5 * count_s * __logf (M_PI) + count_s * __logf (2) + lgammaf(a_0);
    // sm_helper[s].denominator.z = a_n_s * __logf (b_n_s_z) + \
    //   0.5 * count_s * __logf (M_PI) + count_s * __logf (2) + lgammaf(a_0);      

    // a_0 =a_n_f;
    // sm_helper[k].numerator_f_app =a_0*__logf(b_0)+lgammaf(a_n_f)+0.5*__logf(v_n_f);
    // sm_helper[k].denominator_f.x = a_n_f * __logf (b_n_f_x) + \
    //   0.5 * count_f * __logf (M_PI) + count_f * __logf (2) + lgammaf(a_0);
    // sm_helper[k].denominator_f.y = a_n_f * __logf (b_n_f_y) + \
    //   0.5 * count_f * __logf (M_PI) + count_f * __logf (2) + lgammaf(a_0);
    // sm_helper[k].denominator_f.z = a_n_f * __logf (b_n_f_z) + \
    //   0.5 * count_f * __logf (M_PI) + count_f * __logf (2) + lgammaf(a_0);        

    a_0 =a_n_f;
    // sm_helper[k].numerator_f_app =a_0*__logf(b_0)+lgammaf(a_n_f)+0.5*__logf(v_n_f);

    // sm_helper[k].denominator_f.x = a_n_f * __logf (b_n_f_x) + \
    // //   0.5 * count_f * __logf (M_PI) + count_f * __logf (2) + lgammaf(a_0);
    // sm_helper[k].denominator_f.y = a_n_f * __logf (b_n_f_y) + \
    // //   0.5 * count_f * __logf (M_PI) + count_f * __logf (2) + lgammaf(a_0);
    // sm_helper[k].denominator_f.z = a_n_f * __logf (b_n_f_z) + \
    // //   0.5 * count_f * __logf (M_PI) + count_f * __logf (2) + lgammaf(a_0);        


    /********************
  
            Shape
   
    **********************/

    int prior_mu_shape_count = sp_params[k].prior_mu_shape_count;
    int prior_sigma_shape_count = sp_params[k].prior_sigma_shape_count;
    double3 prior_sigma_shape = sp_params[k].prior_sigma_shape;
    double2 prior_mu_shape = sp_params[k].prior_mu_shape;
    int nu_prior = prior_sigma_shape_count;
    int nu_post_k = nu_prior + count_k;
    int nu_post_s = nu_prior + count_s;
    int nu_post_f = nu_prior + count_f;

    float det_prior = 1;
    float det_k = 1;
    float det_s = 1;
    float det_f = 1;
    
    float lprob_const =-lgamma(nu_prior/2)-lgamma((nu_prior-1)/2.)  \
      +(nu_prior/2.)*__logf(det_prior); 
    float lprob_k = lgamma(nu_post_k/2)+ lgamma((nu_post_k-1)/2.) \
      - count_k *__logf(M_PI) - (nu_post_k/2.) * __logf(det_k) \
      - __logf(nu_prior/nu_post_k);
    float lprob_s = lgamma(nu_post_s/2)+ lgamma((nu_post_s-1)/2.) \
      - count_s *__logf(M_PI) - (nu_post_s/2.) * __logf(det_s) \
      - __logf(nu_prior/nu_post_s);
    float lprob_f = lgamma(nu_post_f/2)+ lgamma((nu_post_f-1)/2.) \
      - count_f *__logf(M_PI) - (nu_post_f/2.) * __logf(det_f) \
      - __logf(nu_prior/nu_post_f);
    
    // -- write marginal likelihood of data for shape [p(D)] --
    sm_helper[k].lprob_shape = lprob_k + lprob_const;
    sm_helper[s].lprob_shape = lprob_s + lprob_const;
    sm_helper[k].lprob_f_shape = lprob_f + lprob_const;
}   

__global__
void merge_likelihood_p(const float* img, int* sm_pairs,
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


__global__ void calc_hasting_ratio_p(const float* img, int* sm_pairs,
                                   spix_params* sp_params,
                                   spix_helper* sp_helper,
                                   spix_helper_sm* sm_helper,
                                   const int npix, const int nbatch, const int width,
                                   const int nftrs, const int nspix_buffer,
                                   float alpha_hasting_ratio) {

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


__global__ void calc_hasting_ratio2_p(const float* img, int* sm_pairs,
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
void split_hastings_ratio_p(const float* img, int* sm_pairs,
                          spix_params* sp_params,
                          spix_helper* sp_helper,
                          spix_helper_sm* sm_helper,
                            bool* split_dir,
                          const int npix, const int nbatch,
                          const int width, const int nftrs,
                          const int nspix_buffer,
                          float log_alpha_hasting_ratio,
                          int max_nspix, int* max_sp ) {
  //  we treat the "alpha" as "log_alpha"

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
    // float count_f = count_k + count_s;

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


    // float lprob_k_app = __ldg(&sm_helper[k].numerator_app);
    // float lprob_s_app = __ldg(&sm_helper[s].numerator_app);
    // float lprob_f_app = __ldg(&sm_helper[k].numerator_f_app);

    float lprob_k_app_cond = __ldg(&sm_helper[k].numerator_app);
    float lprob_k_app_ucond = __ldg(&sm_helper[k].b_n_shape_det);
    float lprob_s_app_cond = __ldg(&sm_helper[s].numerator_app);
    float lprob_s_app_ucond = __ldg(&sm_helper[s].b_n_shape_det);
    float lprob_f_app = __ldg(&sm_helper[k].numerator_f_app);

    // sm_helper[k].numerator_app = lprob_k_cond;
    // sm_helper[k].b_n_shape_det = lprob_k_ucond;
    // sm_helper[s].numerator_app = lprob_s_cond;
    // sm_helper[s].b_n_shape_det = lprob_s_ucond;
    // sm_helper[k].numerator_f_app = lprob_f;


    float lprob_k_shape =  __ldg(&sm_helper[k].lprob_shape);
    float lprob_s_shape =  __ldg(&sm_helper[s].lprob_shape);
    float lprob_f_shape =  __ldg(&sm_helper[k].lprob_f_shape);


     //printf("hasating:x k: %d, count: %f, den: %f, %f, %f, b_n: %f, %f, %f, num: %f \n",k, count_k,  sm_helper[k].denominator.x, sm_helper[k].denominator.y,  sm_helper[k].denominator.z,   __logf (sm_helper[k].b_n_app.x) ,  __logf (sm_helper[k].b_n_app.y),   __logf (sm_helper[k].b_n_app.z), sm_helper[k].numerator_app);

    // float log_nominator = __logf(alpha_hasting_ratio)+ lgammaf(count_k)\
    //   + total_marginal_k + lgammaf(count_s) + total_marginal_s;

    // float log_nominator = __logf(alpha_hasting_ratio) \
    //   + lgammaf(count_k) + lgammaf(count_s) + lprob_s_app + lprob_k_app;

    // use Gamma(z) ~= sqrt(2*pi) * pow(z,z-1/2) * exp(-z)
    // float gamma_term = (count_s-0.5)*__logf(count_s) + (count_k-0.5)*__logf(count_k)\
    //   -(count_s+count_k-0.5)*__logf(count_s+count_k)+0.5*__logf(2*M_PI);

    // float _gamma_term_s = 0.5*log(2*M_PI) + -count_s + (count_s-0.5)*log(count_s);
    // float _gamma_term_k = 0.5*log(2*M_PI) + -count_k + (count_k-0.5)*log(count_k);
    // float _gamma_term_f = 0.5*log(2*M_PI) + -count_f + (count_f-0.5)*log(count_f);
    // if count_s > 100

    // float log_nominator = __logf(alpha_hasting_ratio) + lprob_s_app + lprob_k_app;
    // float log_nominator = __logf(alpha_hasting_ratio) + lprob_s_app + lprob_k_app \
    //   + lgammaf(count_k) + lgammaf(count_s);
    // double log_nominator = lgammaf(count_k) + lgammaf(count_s);
    // double log_nominator = 0;

    // log_nominator=total_marginal_k+ total_marginal_s + lprob_k_shape + lprob_s_shape;
    // float log_denominator = lgammaf(count_f) + total_marginal_f + lprob_f_shape;

    // log_nominator= __logf(alpha_hasting_ratio)+ total_marginal_k+ total_marginal_s;

    // log_nominator= __logf(alpha_hasting_ratio);

    // float log_denominator = lgammaf(count_f) + total_marginal_f;
    // float log_denominator =total_marginal_f;
    // float log_denominator =0;
    // float log_denominator = lprob_f_app;
    double gamma_terms = lgamma(count_k) + lgamma(count_s) - lgamma(count_f);
    // printf("lprob_k_app,lprob_s_app,lprob_f_app: %lf,%lf,%lf\n",
    //        lprob_k_app,lprob_s_app,lprob_f_app);

    double pair = lprob_k_app_cond + lprob_s_app_ucond;
    double pair_alt = lprob_k_app_ucond + lprob_s_app_cond;
    bool select = (pair > pair_alt);
    pair = select ? pair : pair_alt;

    sm_helper[k].hasting = log_alpha_hasting_ratio + gamma_terms + pair - lprob_f_app;

    // ".merge" is merely a bool variable; nothing about merging here. only splitting
    sm_helper[k].merge = (sm_helper[k].hasting > 0); // why "-2"?
    sm_helper[s].merge = (sm_helper[k].hasting > 0);

    if((sm_helper[k].merge)) // split step
      {

        s = atomicAdd(max_sp,1) +1; // ? can't multiple splits happen at one time? yes :D
        sm_pairs[2*k] = s;

        split_dir[k] = select; // k = cond and s = ucond? OR visa-versa

        /*************************************************

              Init New Shape Info and Update Priors

        *************************************************/

        // -- update shape prior --
        int prior_count = max((int)(sp_params[k].prior_count/2),1);
        sp_params[k].prior_count = prior_count;

        // -- [appearance] prior --
        float3 prior_mu_app;
        prior_mu_app.x = 0;
        prior_mu_app.y = 0;
        prior_mu_app.z = 0;
        sp_params[s].prior_mu_app = prior_mu_app;
        sp_params[s].prior_mu_app_count = 1;

        // -- [shape] prior --
        double2 prior_mu_shape;
        prior_mu_shape.x = 0;
        prior_mu_shape.y = 0;
        sp_params[s].prior_mu_shape = prior_mu_shape;
        sp_params[s].prior_mu_shape_count = 1;
        double3 prior_sigma_shape;
        prior_sigma_shape.x = prior_count;
        prior_sigma_shape.y = prior_count;
        prior_sigma_shape.z = 0;
        sp_params[s].prior_sigma_shape = prior_sigma_shape;
        sp_params[s].prior_sigma_shape_count = prior_count;
        

      }

}

__global__ void merge_sp_p(int* seg, bool* border,
                           int* sm_pairs, spix_params* sp_params,
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

__global__ void split_sp_p(int* seg, int* sm_seg1, int* sm_pairs,
                         spix_params* sp_params,
                           spix_helper_sm* sm_helper, bool* split_dir,
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
    int split_label = sm_pairs[2*k];
    bool is_k_cond_and_s_ucond = split_dir[k];
    int update_label = is_k_cond_and_s_ucond ? k2 : k;
    if(sm_seg1[idx]==update_label) seg[idx] = split_label;
    //seg[idx] = sm_seg1[idx];
    //printf("Add the following: %d - %d'\n", k,sm_pairs[2*k]);
    sp_params[split_label].valid = 1;
    // sp_params[sm_pairs[2*k]].valid = 1;
    // sp_params[sm_pairs[2*k]].prior_count = sp_params[sm_pairs[2*k]].prior_count;
    // sp_params[k].prior_sigma_shape.x = count*count;
    // sp_params[k].prior_sigma_shape.z = count*count;

    // ?

    return;  
}



__global__ void remove_sp_p(int* sm_pairs, spix_params* sp_params,
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


























/******************************************************************

                         Appearance

******************************************************************/

__device__ float3 calc_app_mean_mode_sm(double3 sample_sum, float3 prior_mu,
                                        int count, int prior_count) {
  float3 post_mu;
  post_mu.x = (sample_sum.x + prior_count * prior_mu.x)/(count + prior_count);
  post_mu.y = (sample_sum.y + prior_count * prior_mu.y)/(count + prior_count);
  post_mu.z = (sample_sum.z + prior_count * prior_mu.z)/(count + prior_count);
  // post_mu.x = sample_sum.x/count;
  // post_mu.y = sample_sum.y/count;
  // post_mu.z = sample_sum.z/count;
  return post_mu;
}



/******************************************************************

                         Shape

******************************************************************/

__device__ double2 calc_shape_sample_mean_sm(int2 sum_shape, int count) {
    double2 mu_shape;
	if (count>0){
      mu_shape.x = sum_shape.x/count;
      mu_shape.y = sum_shape.y/count;
	}else{
      mu_shape.x = sum_shape.x;
      mu_shape.y = sum_shape.y;
    }
    return mu_shape;
}

__device__ double2 calc_shape_mean_mode_sm(double2& mu, double2 prior_mu,
                                           int count, int prior_count) {
  mu.x = (prior_count * prior_mu.x + count*mu.x)/(prior_count + count);
  mu.y = (prior_count * prior_mu.y + count*mu.y)/(prior_count + count);
  return mu;
}

__device__ double3 calc_shape_sigma_mode_sm(longlong3 sq_sum, double2 mu,
                                            double3 prior_sigma, double2 prior_mu,
                                            int count, int prior_count) {

    // -- prior sigma_s --
    double3 sigma_opt = outer_product_term_sm(prior_mu, mu, count, prior_count);

	// -- sample covairance --
    double3 sigma_mode;
    if (count>3){
      sigma_mode.x = sq_sum.x - mu.x * mu.x * count;
      sigma_mode.y = sq_sum.y - mu.x * mu.y * count;
      sigma_mode.z = sq_sum.z - mu.y * mu.y * count;
    }else{
      sigma_mode.x = sq_sum.x;
      sigma_mode.y = sq_sum.y;
      sigma_mode.z = sq_sum.z;
    }

	double tcount = (double) count + prior_count;
    // -- compute cov matrix [.x = dx*dx   .y = dx*dy    .z = dy*dy] --
    sigma_mode.x = (prior_sigma.x + sigma_mode.x + sigma_opt.x) / (tcount + 3.0);
    sigma_mode.y = (prior_sigma.y + sigma_mode.y + sigma_opt.y) / (tcount + 3.0);
    sigma_mode.z = (prior_sigma.z + sigma_mode.z + sigma_opt.z) / (tcount + 3.0);

    return sigma_mode;
}


/************************************************************


                   Helper Functions


************************************************************/

__device__ double marginal_likelihood_app_sm(double3 sum_obs,double3 sq_sum_obs,
                                             float3 prior_mu,int _num_obs,
                                             int _num_prior, float sigma2) {
  // ref: from https://www.cs.ubc.ca/~murphyk/Papers/bayesGauss.pdf
  // Equation 55 with modifications from Eq 57
  double tau2 = 2; // ~= mean has 95% prob to be within (-1,1)
  float num_obs = (float)_num_obs;
  float num_prior = (float)max(_num_prior,1);
  // sigma2 = maxf(sigma2 / num_prior,0.000001); // Eq 57

  double lprob_num = 1/2. * log(sigma2) - num_obs/2.0 * log(2*M_PI*sigma2) \
    - log(num_obs * tau2 + sigma2)/2.;
  double denom = 2*(num_obs*tau2+sigma2);
  double3 lprob;
  // lprob.x = lprob_num - sq_sum_obs.x/(2*sigma2) \
  //   + tau2*sum_obs.x*sum_obs.x/(sigma2*denom);

  lprob.x = lprob_num - sq_sum_obs.x/(2*sigma2) - prior_mu.x*prior_mu.x/(2*tau2) \
    + tau2*sum_obs.x*sum_obs.x/(sigma2*denom)                   \
    + sigma2*prior_mu.x*prior_mu.x/(tau2*denom) + 2*sum_obs.x*prior_mu.x/denom;
  lprob.y = lprob_num - sq_sum_obs.y/(2*sigma2) - prior_mu.y*prior_mu.y/(2*tau2) \
    + tau2*sum_obs.y*sum_obs.y/(sigma2*denom)                   \
    + sigma2*prior_mu.y*prior_mu.y/(tau2*denom) + 2*sum_obs.y*prior_mu.y/denom;
  lprob.z = lprob_num - sq_sum_obs.z/(2*sigma2) - prior_mu.z*prior_mu.z/(2*tau2) \
    + tau2*sum_obs.z*sum_obs.z/(sigma2*denom)                   \
    + sigma2*prior_mu.z*prior_mu.z/(tau2*denom) + 2*sum_obs.z*prior_mu.z/denom;

  // lprob.x = -sq_sum_obs.x/(2*sigma2) - prior_mu.x*prior_mu.x/(2*tau2) \
  //   + tau2*sum_obs.x*sum_obs.x/(sigma2*denom)                   \
  //   + sigma2*prior_mu.x*prior_mu.x/(tau2*denom) + 2*sum_obs.x*prior_mu.x/denom;
  // lprob.y = -sq_sum_obs.y/(2*sigma2) - prior_mu.y*prior_mu.y/(2*tau2) \
  //   + tau2*sum_obs.y*sum_obs.y/(sigma2*denom)                   \
  //   + sigma2*prior_mu.y*prior_mu.y/(tau2*denom) + 2*sum_obs.y*prior_mu.y/denom;
  // lprob.z = -sq_sum_obs.z/(2*sigma2) - prior_mu.z*prior_mu.z/(2*tau2) \
  //   + tau2*sum_obs.z*sum_obs.z/(sigma2*denom)                   \
  //   + sigma2*prior_mu.z*prior_mu.z/(tau2*denom) + 2*sum_obs.z*prior_mu.z/denom;

  // lprob.x = -(sum_obs.x/num_obs-prior_mu.x)*(sum_obs.x/num_obs-prior_mu.x);
  // lprob.y = -(sum_obs.y/num_obs-prior_mu.y)*(sum_obs.y/num_obs-prior_mu.y);
  // lprob.z = -(sum_obs.z/num_obs-prior_mu.z)*(sum_obs.z/num_obs-prior_mu.z);

  double _lprob;
  _lprob = lprob.x+lprob.y+lprob.z;
  return _lprob;
}


__device__ double3 outer_product_term_sm(double2 prior_mu, double2 mu,
                                      int obs_count, int prior_count) {
    double pscale = (1.0*obs_count*prior_count)/(obs_count+prior_count);
    double3 deltas;
    deltas.x = mu.x - prior_mu.x;
    deltas.y = mu.y - prior_mu.y;
    double3 opt_sigma;
    opt_sigma.x = pscale * deltas.x * deltas.x;
    opt_sigma.y = pscale * deltas.x * deltas.y;
    opt_sigma.z = pscale * deltas.y * deltas.y;
    return opt_sigma;
}


// Function to compute the determinant of a symmetric 2x2 covariance matrix
__device__ double determinant2x2_sm(double3 sigma) {
    // det(Sigma) = sigma11 * sigma22 - sigma12^2
    double det = sigma.x * sigma.z - sigma.y * sigma.y;
    if (det <= 0){
      sigma.x = sigma.x + 0.00001;
      sigma.z = sigma.z + 0.00001;
      det = sigma.x * sigma.z - sigma.y * sigma.y;
      if(det <=0) det = 0.0001;//hack
    }
    return det;
}

