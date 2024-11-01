
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

#include "split_prop.h"
// #include "update_params.h"

#include <stdio.h>
#ifndef WIN32
#include <unistd.h>
#endif

__host__
int run_split_prop(const float* img, int* seg, bool* border,
                   spix_params* sp_params, spix_helper* sp_helper,
                   spix_helper_sm_v2* sm_helper,
                   int* sm_seg1 ,int* sm_seg2, int* sm_pairs,
                   float alpha_hastings, float sigma2_app,
                   int& count, int idx, int max_spix,
                   const int npix, const int nbatch,
                   const int width, const int height,
                   const int nftrs, const int nspix_buffer){

  // -- (left/right,up/down,new/old) --
  count += 1;
  int split_direction = count%2+1;
  int oldnew_choice = count%4;

  // -- run split --
  max_spix = CudaCalcSplitCandidate_p(img, seg, border,
                                       sp_params, sp_helper, sm_helper,
                                       sm_seg1, sm_seg2, sm_pairs,
                                       npix,nbatch,width,height,nftrs,
                                       nspix_buffer, max_spix,
                                       split_direction, oldnew_choice,
                                       alpha_hastings, sigma2_app);

  return max_spix;
}


void saveIntArray(int* data, int H, int W, const std::string& filename) {
    // Create a PyTorch tensor from the raw pointer
    // Note: 'torch::kInt' specifies that the tensor will have an integer data type
    auto options_i32 = torch::TensorOptions().dtype(torch::kInt32)
      .layout(torch::kStrided).device("cuda");
    torch::Tensor tensor = torch::from_blob(data, {H, W}, options_i32);

    // Save the tensor to a file
    torch::save(tensor, filename);
}


__host__ int CudaCalcSplitCandidate_p(const float* img, int* seg, bool* border,
                                      spix_params* sp_params, spix_helper* sp_helper,
                                      spix_helper_sm_v2* sm_helper,
                                      int* sm_seg1, int* sm_seg2, int* sm_pairs,
                                      const int npix, const int nbatch, const int width,
                                      const int height, const int nftrs,
                                      const int nspix_buffer, int max_spix,
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
    // printf("width: %d\n",width);
    // -- splits --
    init_sm_p<<<BlockPerGrid2,ThreadPerBlock>>>(img,seg,sp_params,sm_helper,
                                                split_dir,nspix_buffer,
                                                nbatch, width, nftrs, sm_pairs);
    init_split_p<<<BlockPerGrid2,ThreadPerBlock>>>(border,sm_seg1,sp_params,
                                                   sm_helper, split_dir, nspix_buffer,
                                                   nbatch, width, height, direction,
                                                   seg, max_sp, max_spix);
    init_split_p<<<BlockPerGrid2,ThreadPerBlock>>>(border,sm_seg2,sp_params,
                                                   sm_helper, split_dir, nspix_buffer,
                                                   nbatch, width,height, -direction,
                                                   seg, max_sp, max_spix);

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

    // -- save the segmentations for inspection --
    // saveIntArray(sm_seg1,height,width,"sm_seg1.pth");
    // saveIntArray(sm_seg2,height,width,"sm_seg2.pth");
    

    // updates the segmentation to the two regions; split either left/right or up/down.
    // previous it's a sum of distances from the center; now its a new label
    calc_seg_split_p<<<BlockPerGrid,ThreadPerBlock>>>(sm_seg1,sm_seg2, seg,
                                                      oldnew_choice, npix,
                                                      nbatch, max_spix);
    // saveIntArray(sm_seg1,height,width,"prop_seg.pth");

    // std::string fname_split1_post = "split1_post";
    // write_tensor_to_file_v2(sm_seg1,height,width,fname_split1_post);

    // computes summaries stats for each split
    sum_by_label_split_p<<<BlockPerGrid,ThreadPerBlock>>>(img, sm_seg1, sp_params,
                                                          sm_helper, npix, nbatch,
                                                          width,nftrs,max_spix);
    split_marginal_likelihood_p<<<BlockPerGrid2,ThreadPerBlock>>>(\
       sp_params,sm_helper,npix,nbatch,height,width,nspix_buffer,
        sigma2_app, max_spix);

    // fprintf(stdout,"[s_m.cu] max_spix: %d\n",max_spix);
    split_hastings_ratio_p<<<BlockPerGrid2,ThreadPerBlock>>>(img, sm_pairs, sp_params,
                                                           sp_helper, sm_helper,
                                                             split_dir,
                                                           npix, nbatch, width, nftrs,
                                                           nspix_buffer,
                                                           alpha_hasting_ratio,
                                                           max_spix, max_sp);

    // -- do the split --
    split_sp_p<<<BlockPerGrid,ThreadPerBlock>>>(seg,sm_seg1,sm_pairs,
                                                sp_params, sm_helper,
                                                split_dir, npix,
                                                nbatch, width, height, max_spix);


    // gpuErrchk( cudaPeekAtLastError() );
    // gpuErrchk( cudaDeviceSynchronize() );

    cudaMemcpy(&max_spix, max_sp, sizeof(int), cudaMemcpyDeviceToHost);
    // printf("max_sp: %d\n",max_spix);
    cudaFree(split_dir);
    cudaFree(max_sp);
    cudaFree(done_gpu);

    return max_spix;
}



__global__ void init_sm_p(const float* img, const int* seg_gpu,
                          spix_params* sp_params,
                          spix_helper_sm_v2* sm_helper, bool* split_dir,
                          const int nspix_buffer, const int nbatch,
                          const int width,const int nftrs,int* sm_pairs) {
	int k = threadIdx.x + blockIdx.x * blockDim.x;  // the label
	if (k>=nspix_buffer) return;
	//if (sp_params[k].valid == 0) return;

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

    sm_helper[k].merge = false;
    sm_helper[k].remove = false;
    sm_pairs[k*2+1] = 0;
    sm_pairs[k*2] = 0;
   
    
    // "UpDown" if sY > sX -> inv_sX > inv_sY; 
    double inv_sX = sp_params[k].sigma_shape.x;
    double inv_sY = sp_params[k].sigma_shape.z;
    split_dir[k] = inv_sY < inv_sX; // bool for "UpDown"; o.w. "LeftRight"

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
                             spix_helper_sm_v2* sm_helper,
                             bool* split_dir,
                             const int nspix_buffer,
                             const int nbatch, const int width,
                             const int height, const int direction,
                             const int* seg, int* max_sp, int max_spix) {

    // todo: add batch -- no nftrs
	int k = threadIdx.x + blockIdx.x * blockDim.x;  // the label
	if (k>=nspix_buffer) return;
	if (sp_params[k].valid == 0) return;
    // *max_sp = max_spix+1;
    *max_sp = max_spix;
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
                                 int nbatch, int max_spix) {

  // todo -- nbatch
    int t = threadIdx.x + blockIdx.x * blockDim.x;
	if (t>=npix) return;
    int seg_val = __ldg(&seg[t]);

    if(sm_seg1[t]>__ldg(&sm_seg2[t])){ seg_val += max_spix; }
    sm_seg1[t] = seg_val;

    return;
}

__global__ void sum_by_label_split_p(const float* img, const int* seg,
                                   spix_params* sp_params,
                                   spix_helper_sm_v2* sm_helper,
                                   const int npix, const int nbatch,
                                   const int width, const int nftrs, int max_spix) {
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
                                 spix_helper_sm_v2* sm_helper,
                                 const int npix, const int nbatch,
                                 const int height, const int width,
                                 const int nspix_buffer,
                                   float sigma2_app, int max_spix){

  /********************
           Init
    **********************/

    // -- init --
	int k = threadIdx.x + blockIdx.x * blockDim.x;  // the label
	if (k>=nspix_buffer) return;
	if (sp_params[k].valid == 0) return;

    // -- split --
    int s = k + max_spix;
	if (s>=nspix_buffer) return;
    int _count_f = __ldg(&sp_params[k].count);
    int count_k = __ldg(&sm_helper[k].count);
    int count_s = __ldg(&sm_helper[s].count);
    int count_f = count_k + count_s;

    if((count_f<1)||( count_k<1)||(count_s<1)) return;


    /********************
  
          Appearance
   
    **********************/

    // -- prior counts --
    int prior_count = sp_params[k].prior_count;
    int prior_count_div2 = max(prior_count/2,8);

    // -- priors --
    sp_params[s].prior_mu_app.x = 0;
    sp_params[s].prior_mu_app.y = 0;
    sp_params[s].prior_mu_app.z = 0;
    float3 mu_pr_s = sp_params[s].prior_mu_app;
    mu_pr_s.x = 0;
    mu_pr_s.y = 0;
    mu_pr_s.z = 0;
    float3 mu_pr_k = sp_params[k].prior_mu_app; // USING the prior
    // float3 mu_pr_k = mu_pr_s; // USING NO prior
    float3 mu_pr_f = mu_pr_k;

    //   its just classification :D

    // -- summary stats --
    double3 sum_s = sm_helper[s].sum_app;
    double3 sum_k = sm_helper[k].sum_app;
    double3 sum_f;
    sum_f.x = (sum_s.x + sum_k.x)/2.0;
    sum_f.y = (sum_s.y + sum_k.y)/2.0;
    sum_f.z = (sum_s.z + sum_k.z)/2.0;
    // sum_f.x = (sum_s.x + sum_k.x);
    // sum_f.y = (sum_s.y + sum_k.y);
    // sum_f.z = (sum_s.z + sum_k.z);
    double3 mu_k_app = sample_mean_app(sum_k, count_k);
    double3 mu_s_app = sample_mean_app(sum_s, count_s);
    double3 mu_f_app = sample_mean_app(sum_f, count_f);

    // double3 sq_sum_s = sm_helper[s].sum_app;
    // double3 sq_sum_k = sm_helper[k].sum_app;
    // double3 sq_sum_f;
    // sq_sum_f.x = (sq_sum_s.x + sq_sum_k.x)/2.0;
    // sq_sum_f.y = (sq_sum_s.y + sq_sum_k.y)/2.0;
    // sq_sum_f.z = (sq_sum_s.z + sq_sum_k.z)/2.0;

    // -- marginal likelihoods --
    // ONLY USE NON_INFO PRIOR; mu_pr_k -> mu_pr_s ALWAYS but  "cond" should use mu_pr_k
    // double lprob_k_cond_app = marginal_likelihood_app_sm(sum_k,sq_sum_k,mu_pr_k,count_k,
    //                                                  prior_count_div2,sigma2_app);
    // // double lprob_k_ucond_app = marginal_likelihood_app_sm(sum_k,sq_sum_k,mu_pr_s,
    // //                                                       count_k,prior_count_div2,
    // //                                                       sigma2_app);
    // double lprob_k_ucond_app = 0;//only compare the two conditioned ones?
    // double lprob_s_cond_app = marginal_likelihood_app_sm(sum_s,sq_sum_s,mu_pr_k,count_s,
    //                                                  prior_count_div2,sigma2_app);
    // double lprob_s_ucond_app = 0; // only compare the two conditioned ones?
    // // double lprob_s_ucond_app = marginal_likelihood_app_sm(sum_s,sq_sum_s,mu_pr_s,
    // //                                                       count_s,prior_count_div2,
    // //                                                       sigma2_app);
    // double lprob_f_app = marginal_likelihood_app_sm(sum_f,sq_sum_f,mu_pr_f,count_f,
    //                                             prior_count_div2,sigma2_app);

    // // -- write --
    // sm_helper[k].lprob_k_cond_app = lprob_k_cond_app;
    // sm_helper[k].lprob_k_ucond_app = lprob_k_ucond_app;
    // sm_helper[s].lprob_s_cond_app = lprob_s_cond_app;
    // sm_helper[s].lprob_s_ucond_app = lprob_s_ucond_app;
    // sm_helper[k].lprob_f_app = lprob_f_app;


    /********************
  
            Shape
   
    **********************/

    // -- priors --
    double3 new_prior_sigma;
    new_prior_sigma.x = prior_count_div2;
    new_prior_sigma.y = 0;
    new_prior_sigma.z = prior_count_div2;

    double2 prior_mu_shape = sp_params[k].prior_mu_shape;
    double3 prior_sigma_shape = sp_params[k].prior_sigma_shape;
    // prior_sigma_shape.x = prior_count*prior_count;
    // prior_sigma_shape.y = 0;
    // prior_sigma_shape.z = prior_count*prior_count;
    // float3 sigma_shape_pr_f = mu_shape_pr_k;
    // float3 mu_pr_s = sp_params[s].prior_mu_app;
    // mu_pr_s.x = 0;
    // mu_pr_s.y = 0;
    // mu_pr_s.z = 0;

    // // -- summary stats --
    int2 sum_shape_s = sm_helper[s].sum_shape;
    int2 sum_shape_k = sm_helper[k].sum_shape;
    int2 sum_shape_f;
    sum_shape_f.x = sum_shape_s.x + sum_shape_k.x;
    sum_shape_f.y = sum_shape_s.y + sum_shape_k.y;
    double2 mu_s_shape = calc_shape_sample_mean_sm(sum_shape_s,count_s);
    double2 mu_k_shape = calc_shape_sample_mean_sm(sum_shape_k,count_k);
    double2 mu_f_shape = calc_shape_sample_mean_sm(sum_shape_f,count_f);

    ulonglong3 sq_sum_shape_s = sm_helper[s].sq_sum_shape;
    ulonglong3 sq_sum_shape_k = sm_helper[k].sq_sum_shape;
    ulonglong3 sq_sum_shape_f;
    sq_sum_shape_f.x = sq_sum_shape_s.x + sq_sum_shape_k.x;
    sq_sum_shape_f.y = sq_sum_shape_s.y + sq_sum_shape_k.y;
    sq_sum_shape_f.z = sq_sum_shape_s.z + sq_sum_shape_k.z;

    // -- posterior shape [s] --
    double3 sigma_shape_s = calc_shape_sample_sigma_sm(sq_sum_shape_s,mu_s_shape,
                                                       count_s);
    // double3 sigma_shape_s = calc_shape_sigma_mode_sm(sq_sum_shape_s,mu_s_shape,
    //                                                  prior_sigma_shape,
    //                                                  count_s,prior_count_div2);
    // double det_sigma_shape_s = determinant2x2_sm(sigma_shape_s);

    // -- posterior shape [k] --
    double3 sigma_shape_k = calc_shape_sample_sigma_sm(sq_sum_shape_k,mu_k_shape,
                                                       count_k);
    // double3 sigma_shape_k = calc_shape_sigma_mode_sm(sq_sum_shape_k,mu_k_shape,
    //                                                  prior_sigma_shape,
    //                                                  count_k,prior_count_div2);
    // double det_sigma_shape_k = determinant2x2_sm(sigma_shape_k);

    // -- posterior shape [f] --
    double3 sigma_shape_f = calc_shape_sample_sigma_sm(sq_sum_shape_f,mu_f_shape,
                                                       count_f);
    // double3 sigma_shape_f = calc_shape_sigma_mode_sm(sq_sum_shape_f,mu_f_shape,
    //                                                  prior_sigma_shape,
    //                                                  count_f,prior_count_div2);
    // double det_sigma_shape_f = determinant2x2_sm(sigma_shape_f);


    // printf("info: %d,%lf,%lf,%lf,%lf,%lf,%lf\n",count_s,
    //        sigma_shape_s.x,sigma_shape_s.y,sigma_shape_s.z,
    //        prior_sigma_shape.x,prior_sigma_shape.y,prior_sigma_shape.z);

    // printf("info: %d|%lf|%lld,%lld,%lld|%d,%d|%lf,%lf|%lf,%lf,%lf\n",
    //        count_s,det_sigma_shape_s,
    //        sq_sum_shape_s.x,sq_sum_shape_s.y,sq_sum_shape_s.z,
    //        sum_shape_s.x,sum_shape_s.y,
    //        mu_s_shape.x,mu_s_shape.y,
    //        sigma_shape_s.x,sigma_shape_s.y,sigma_shape_s.z);

    // printf("info[%d]: %d,%d,%d,%d|%lf|%lld,%lld,%lld|%d,%d|%lf,%lf|%lf,%lf,%lf\n",k,
    //        count_f,_count_f,count_s,count_k,det_sigma_shape_f,
    //        sq_sum_shape_f.x,sq_sum_shape_f.y,sq_sum_shape_f.z,
    //        sum_shape_f.x,sum_shape_f.y,
    //        mu_f_shape.x,mu_f_shape.y,
    //        sigma_shape_f.x,sigma_shape_f.y,sigma_shape_f.z);

    // printf("info: %d,%lf,%lf,%lf\n",count_s,
    //        sigma_shape_s.x,sigma_shape_s.y,sigma_shape_s.z);
    // x ~= 300, y ~= -5000, z ~= -51000

    // printf("info: %lf,%lf,%lf\n",prior_sigma_shape.x,prior_sigma_shape.y,prior_sigma_shape.z);

    // printf("info: %d,%d,%lf,%lf,%lf|%lf,%lf,%lf|%lf,%lf|%lf,%lf,%lf|%lf,%lf|%d,%d\n",
    //        count_s,count_k,det_sigma_shape_s,det_sigma_shape_k,det_sigma_shape_f,
    //        sigma_shape_s.x,sigma_shape_s.y,sigma_shape_s.z,
    //        mu_s_shape.x,mu_s_shape.y,
    //        sigma_shape_k.x,sigma_shape_k.y,sigma_shape_k.z,
    //        mu_k_shape.x,mu_k_shape.y,
    //        prior_count*prior_count,prior_count*prior_count/4);

    // printf("info: %d,%d,%lf,%lf,%lf|%lld,%lld,%lld|%lld,%lld,%lld\n",
    //        count_s,count_k,det_sigma_shape_s,det_sigma_shape_k,det_sigma_shape_f,
    //        sq_sum_shape_s.x,sq_sum_shape_s.y,sq_sum_shape_s.z,
    //        sq_sum_shape_k.x,sq_sum_shape_k.y,sq_sum_shape_k.z);

    // -- marginal likelihoods --
    // int pr_count = prior_count;
    // int pr_count2 = max(prior_count/2,8);
    // double det_pr = pr_count*pr_count;
    // double det_pr2 = pr_count2*pr_count2;
    // double lprob_k_cond_shape = marginal_likelihood_shape_sm(det_sigma_shape_k,
    //                                                          det_pr2,
    //                                                          pr_count2,count_k);
    // double lprob_s_cond_shape = marginal_likelihood_shape_sm(det_sigma_shape_s,
    //                                                          det_pr2,
    //                                                          pr_count2,count_s);
    // double lprob_f_shape = marginal_likelihood_shape_sm(det_sigma_shape_f,
    //                                                     det_pr,
    //                                                     pr_count2,count_f);

    double _countf2 = count_f/2.;
    double count_cmp_s = (_countf2 - count_s)/(_countf2/2.);
    count_cmp_s = 2*count_cmp_s*count_cmp_s;
    double count_cmp_k = (_countf2 - count_k)/(_countf2/2.);
    count_cmp_k = 2*count_cmp_k*count_cmp_k;

    // -- ... --
    // double tmp0,tmp1,tmp2;
    double lprob_k_cond_shape = 0;
    lprob_k_cond_shape = wasserstein_sm(sigma_shape_k,prior_sigma_shape);
    // tmp0 = wasserstein_sm(sigma_shape_k,prior_sigma_shape);
    lprob_k_cond_shape += wasserstein_sm(sigma_shape_s,new_prior_sigma);
    lprob_k_cond_shape += l2_delta_app(mu_k_app,mu_pr_k);
    lprob_k_cond_shape += l2_delta_shape(mu_k_shape,prior_mu_shape,height,width);
    // lprob_k_cond_shape += count_cmp_k;

    double lprob_s_cond_shape = 0;
    lprob_s_cond_shape = wasserstein_sm(sigma_shape_s,prior_sigma_shape);
    // tmp1 = wasserstein_sm(sigma_shape_s,prior_sigma_shape);
    lprob_s_cond_shape += wasserstein_sm(sigma_shape_k,new_prior_sigma);
    lprob_s_cond_shape += l2_delta_app(mu_s_app,mu_pr_k);
    lprob_s_cond_shape += l2_delta_shape(mu_s_shape,prior_mu_shape,height,width);
    // lprob_k_cond_shape += count_cmp_s;

    double lprob_f_shape = 0;
    lprob_f_shape = wasserstein_sm(sigma_shape_f,prior_sigma_shape);
    // tmp2 = wasserstein_sm(sigma_shape_f,prior_sigma_shape);
    lprob_f_shape += l2_delta_app(mu_f_app,mu_pr_k);
    lprob_f_shape += l2_delta_shape(mu_f_shape,prior_mu_shape,height,width);

    if ((lprob_k_cond_shape < lprob_f_shape) or (lprob_s_cond_shape < lprob_f_shape)){
    // printf("info[%d->%d]: %d,%d,%d,%d|%lf,%lf,%lf|%lf,%lf,%lf||%lld,%lld,%lld|%lld,%lld,%lld|%lld,%lld,%lld||%lf,%lf,%lf|%lf,%lf,%lf|%lf,%lf,%lf||%lf,%lf,%lf|%lf,%lf,%lf|\n",
    printf("info[%d->%d]: %d,%d,%d,%d|%lf,%lf,%lf||%lf,%lf,%lf|%lf,%lf,%lf|%lf,%lf,%lf||%lf,%lf,%lf|%lf,%lf,%lf|\n",
           k,s,prior_count,count_f,count_s,count_k,
           // tmp0,tmp1,tmp2,
           lprob_s_cond_shape,lprob_k_cond_shape,lprob_f_shape,
           // sq_sum_shape_s.x,sq_sum_shape_s.y,sq_sum_shape_s.z,
           // sq_sum_shape_k.x,sq_sum_shape_k.y,sq_sum_shape_k.z,
           // sq_sum_shape_f.x,sq_sum_shape_f.y,sq_sum_shape_f.z,
           sigma_shape_s.x,sigma_shape_s.y,sigma_shape_s.z,
           sigma_shape_k.x,sigma_shape_k.y,sigma_shape_k.z,
           sigma_shape_f.x,sigma_shape_f.y,sigma_shape_f.z,
           prior_sigma_shape.x,prior_sigma_shape.y,prior_sigma_shape.z,
           new_prior_sigma.x,new_prior_sigma.y,new_prior_sigma.z);
    }

    // printf("info: %d,%lf,%lf,%lf,%lf,%lf,%lf\n",count_s,
    //        sigma_shape_s.x,sigma_shape_s.y,sigma_shape_s.z,
    //        prior_sigma_shape.x,prior_sigma_shape.y,prior_sigma_shape.z);

    // -- write --
    sm_helper[k].lprob_k_cond_shape = lprob_k_cond_shape;
    sm_helper[k].lprob_k_ucond_shape = 0;
    sm_helper[s].lprob_s_cond_shape = lprob_s_cond_shape;
    sm_helper[s].lprob_s_ucond_shape = 0;
    sm_helper[k].lprob_f_shape = lprob_f_shape;



}


__global__
void split_hastings_ratio_p(const float* img, int* sm_pairs,
                          spix_params* sp_params,
                          spix_helper* sp_helper,
                          spix_helper_sm_v2* sm_helper,
                            bool* split_dir,
                          const int npix, const int nbatch,
                          const int width, const int nftrs,
                          const int nspix_buffer,
                          float log_alpha_hasting_ratio,
                          int max_spix, int* max_sp ) {
  //  we treat the "alpha" as "log_alpha"

  // todo -- add nbatch and nftrs
	// getting the index of the pixel
	int k = threadIdx.x + blockIdx.x * blockDim.x;  // the label

	if (k>=nspix_buffer) return;
	if (sp_params[k].valid == 0) return;
    
    int s = k + max_spix;
    if(s>=nspix_buffer) return;
    // float _count_f = __ldg(&sp_params[k].count);
    float count_k = __ldg(&sm_helper[k].count);
    float count_s = __ldg(&sm_helper[s].count);
    float count_f = count_k + count_s;

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

    double lprob_k_cond_shape = __ldg(&sm_helper[k].lprob_k_cond_shape);
    double lprob_k_ucond_shape = 0;
    double lprob_s_cond_shape = __ldg(&sm_helper[s].lprob_s_cond_shape);
    double lprob_s_ucond_shape = 0;
    double lprob_f_shape = __ldg(&sm_helper[k].lprob_f_shape);
    double orig_shape = lprob_f_shape;

    float lprob_k_cond_app = __ldg(&sm_helper[k].lprob_k_cond_app);
    float lprob_k_ucond_app = __ldg(&sm_helper[k].lprob_k_ucond_app);
    float lprob_s_cond_app = __ldg(&sm_helper[s].lprob_s_cond_app);
    float lprob_s_ucond_app = __ldg(&sm_helper[s].lprob_s_ucond_app);
    float lprob_f_app = __ldg(&sm_helper[k].lprob_f_app);
    // float lprob_f_app = __ldg(&sm_helper[k].numerator_f_app);

    // sm_helper[k].numerator_app = lprob_k_cond;
    // sm_helper[k].b_n_shape_det = lprob_k_ucond;
    // sm_helper[s].numerator_app = lprob_s_cond;
    // sm_helper[s].b_n_shape_det = lprob_s_ucond;
    // sm_helper[k].numerator_f_app = lprob_f;


    // float lprob_k_shape =  __ldg(&sm_helper[k].lprob_shape);
    // float lprob_s_shape =  __ldg(&sm_helper[s].lprob_shape);
    // float lprob_f_shape =  __ldg(&sm_helper[k].lprob_f_shape);


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


    // double pair = lprob_k_cond_app + lprob_s_ucond_app;
    // double pair_alt = lprob_k_ucond_app + lprob_s_cond_app;
    // bool select = (pair > pair_alt);
    // pair = select ? pair : pair_alt;

    double pair =  lprob_k_cond_shape;
    double pair_alt = lprob_s_cond_shape;
    bool select = (pair < pair_alt);
    pair = select ? pair : pair_alt;

    // -- get hasting for merge --
    double alpha = exp(log_alpha_hasting_ratio);
    double gamma2_num = lgammaf(count_f) + lgammaf(alpha) + \
      lgammaf(alpha / 2 + count_k) + lgammaf(alpha / 2 + count_f -  count_k);
    double gamma2_denom = log_alpha_hasting_ratio + lgammaf(count_k) +
     lgammaf(count_f -  count_k) + lgammaf(alpha + count_f) +
     lgammaf(alpha / 2) + lgammaf(alpha / 2);
    double hastings_merge = gamma2_num - gamma2_denom;

    // float hasting = log_alpha_hasting_ratio + gamma_terms + pair - lprob_f_app;
    float hastings_split = log_alpha_hasting_ratio + gamma_terms + pair - lprob_f_app;
    sm_helper[k].hasting = hastings_split;

    // bool is_merge = (sm_helper[k].hasting > 0) and (hastings_merge < 0);
    // bool is_merge = (sm_helper[k].hasting > 0); 
    bool is_merge = pair < lprob_f_shape;

    // ".merge" is merely a bool variable; nothing about merging here. only splitting
    // sm_helper[k].merge = (sm_helper[k].hasting > 0); // why "-2"?
    // sm_helper[s].merge = (sm_helper[k].hasting > 0);
    sm_helper[k].merge = is_merge;
    sm_helper[s].merge = is_merge;

    int _prior_count = sp_params[k].prior_count;
    // printf("pr_count,count_f,count_k,count_s,pair,lprob_f_app: %d,%f,%f,%f,%lf,%lf\n",
    //        _prior_count,count_f,count_k,count_s,pair,lprob_f_app);

    // printf("info: %d|%f,%f,%f|%lf,%lf,%lf\n",
    //        _prior_count,count_f,count_k,count_s,
    //        lprob_k_cond_shape,lprob_s_cond_shape,lprob_f_shape);

    if(is_merge) // split step
      {

        s = atomicAdd(max_sp,1) +1; // ? can't multiple splits happen at one time? yes :D
        sm_pairs[2*k] = s;
        printf("info [%d->%d]\n",k,s);

        split_dir[k] = select; // [true] k = cond and s = ucond? OR [false] visa-versa

        /*************************************************

              Init New Shape Info and Update Priors

        *************************************************/

        // -- update shape prior --
        int prior_count = max((int)(sp_params[k].prior_count/2),8);

        // int prior_count = max((int)(sp_params[k].prior_count/1.41421356),8);
        sp_params[k].prior_count = prior_count; //must be done
        sp_params[s].prior_count = prior_count;

        // printf("pr_count,hasting: %d,%lf,%lf,%lf,%lf\n",
        //        prior_count,hasting,lprob_k_cond_app,lprob_s_cond_app,lprob_f_app);
        // printf("pr_count,hasting: %d,%lf,%lf\n",prior_count,hasting,hastings_merge);
        // printf("pr_count,count_f,count_k,count_s,hS,hM: %d,%f,%f,%f,%lf,%lf\n",
        //        prior_count,count_f,count_k,count_s,hastings_split,hastings_merge);

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
        prior_sigma_shape.y = 0;
        prior_sigma_shape.z = prior_count;
        sp_params[s].prior_sigma_shape = prior_sigma_shape;
        sp_params[s].prior_sigma_shape_count = prior_count;
        sp_params[s].is_cond = false;

      }

}


__global__ void split_sp_p(int* seg, int* sm_seg1, int* sm_pairs,
                         spix_params* sp_params,
                           spix_helper_sm_v2* sm_helper, bool* split_dir,
                         const int npix, const int nbatch,
                         const int width, const int height, int max_spix){   

  // todo: add nbatch, no sftrs
    int idx = threadIdx.x + blockIdx.x * blockDim.x;  
    if (idx>=npix) return; 
    int k = seg[idx]; // center 
    int k2 = k + max_spix;
    if ((sm_helper[k].merge == false)||sm_helper[k2].merge == false){
      return;
    }
    int split_label = sm_pairs[2*k];
    bool is_k_cond_and_s_ucond = split_dir[k];
    int update_label = is_k_cond_and_s_ucond ? k2 : k;
    if(sm_seg1[idx]==update_label){
      // printf("info[%d->%d]\n",seg[idx],split_label);
      seg[idx] = split_label;
    }
    //seg[idx] = sm_seg1[idx];
    //printf("Add the following: %d - %d'\n", k,sm_pairs[2*k]);
    sp_params[split_label].valid = 1;
    // sp_params[sm_pairs[2*k]].valid = 1;
    // sp_params[sm_pairs[2*k]].prior_count = sp_params[sm_pairs[2*k]].prior_count;
    // sp_params[k].prior_sigma_shape.x = count*count;
    // sp_params[k].prior_sigma_shape.z = count*count;

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

__device__ double3 sample_mean_app(double3 sum_app, int _count) {
  double count = 1.*_count;
    double3 mu_app;
	if (count>0){
      mu_app.x = sum_app.x/count;
      mu_app.y = sum_app.y/count;
      mu_app.z = sum_app.z/count;
	}else{
      mu_app.x = sum_app.x;
      mu_app.y = sum_app.y;
      mu_app.z = sum_app.z;
    }
    return mu_app;
}

__device__ double2 calc_shape_sample_mean_sm(int2 sum_shape, int _count) {
  double count = 1.*_count;
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

__device__ double3 calc_shape_sample_sigma_sm(ulonglong3 sq_sum,
                                              double2 mu,int _count) {
	// -- sample covairance --
    double count = 1.0*_count;
    double3 sigma_mode;
    if (count>0){
      sigma_mode.x = (sq_sum.x/count - mu.x * mu.x);
      sigma_mode.y = (sq_sum.y/count - mu.x * mu.y);
      sigma_mode.z = (sq_sum.z/count - mu.y * mu.y);
    }else{
      sigma_mode.x = sq_sum.x;
      sigma_mode.y = sq_sum.y;
      sigma_mode.z = sq_sum.z;
    }
    return sigma_mode;
}

__device__ double3 calc_shape_sigma_mode_sm(ulonglong3 sq_sum, double2 mu,
                                            double3 prior_sigma,
                                            int count, int prior_count) {

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
    sigma_mode.x = (prior_count*prior_sigma.x + sigma_mode.x) / (tcount + 3.0);
    sigma_mode.y = (prior_count*prior_sigma.y + sigma_mode.y) / (tcount + 3.0);
    sigma_mode.z = (prior_count*prior_sigma.z + sigma_mode.z) / (tcount + 3.0);

    return sigma_mode;
}


/************************************************************


                   Helper Functions


************************************************************/

__device__ double delta_mean_app_sm(double3 mu_est,double3 mu_prior) {
  double delta = (mu_est.x - mu_prior.x)*(mu_est.x - mu_prior.x);
  delta += (mu_est.y - mu_prior.y)*(mu_est.y - mu_prior.y);
  delta += (mu_est.z - mu_prior.z)*(mu_est.z - mu_prior.z);
  return delta/12.; // div 3 and each by 2 since [-1,1]
}

__device__ double l2_delta_shape(double2 mu_est,double2 mu_prior,
                                 int _height, int _width){
  double width = 1.*_width;
  double height = 1.*_height;
  double delta_x = (mu_est.x - mu_prior.x)/width;
  double delta_y = (mu_est.y - mu_prior.y)/height;
  double delta = delta_x*delta_x + delta_y*delta_y;
  return sqrt(delta/2.);
}

__device__ double l2_delta_app(double3 mu_est,float3 mu_prior){
    double delta = (mu_est.x - mu_prior.x)*(mu_est.x - mu_prior.x);
    delta += (mu_est.y - mu_prior.y)*(mu_est.y - mu_prior.y);
    delta += (mu_est.z - mu_prior.z)*(mu_est.z - mu_prior.z);
    return sqrt(delta/3.);
}


__device__ double wasserstein_sm(double3 sigma_est,double3 sigma_prior) {

  // Step 1: Compute eigenvalues for sigma_est
  double2 eigen_est = eigenvals_cov(sigma_est);
  double lambda1_est = eigen_est.x;
  double lambda2_est = eigen_est.y;

  // Step 2: Compute eigenvalues for sigma_prior
  double2 eigen_prior = eigenvals_cov(sigma_prior);
  double lambda1_prior = eigen_prior.x;
  double lambda2_prior = eigen_prior.y;

  // Step 3: Compute the trace term
  double trace_term = sigma_est.x + sigma_est.z + sigma_prior.x + sigma_prior.z;

  // Step 4: Compute the cross term using the square root of the products of eigenvalues
  double cross_term = 2.0 * (sqrt(lambda1_est * lambda1_prior) + \
                             sqrt(lambda2_est * lambda2_prior));

  // Step 5: Wasserstein squared distance
  double wasserstein_distance_squared = trace_term - cross_term;

  // Return the square root to get the actual Wasserstein distance
  return sqrt(wasserstein_distance_squared);

}

__device__ double2 eigenvals_cov(double3 cov) {

  // -- unpack --
  double s11 = cov.x;
  double s12 = cov.y;
  double s22 = cov.z;

  // Calculate the trace and determinant
  double trace = s11 + s22;
  double determinant = s11 * s22 - s12 * s12;

  // Calculate the square root term
  double term = sqrt((trace / 2) * (trace / 2) - determinant);

  // Compute the two eigenvalues
  double lambda1 = (trace / 2) + term;
  double lambda2 = (trace / 2) - term;

  return make_double2(lambda1, lambda2);
}

__device__ double marginal_likelihood_shape_sm(double det_post, double det_pr,
                                               int _pr_count,int _num_obs) {
  if (det_post < 0.001){
    return -10000000;
  }
  double num_obs = 1.*_num_obs;
  double pr_count = 1.*_pr_count;
  double post_count = 1.*_pr_count + num_obs;
  double gamma2_post = lgamma(post_count/2) + lgamma((post_count-1)/2);
  double gamma2_pr = lgamma(pr_count/2) + lgamma((pr_count-1)/2);
  double h_const = num_obs * log(M_PI) + log(pr_count) - log(post_count);
  // double lprob = pr_count*log(det_pr) - post_count*log(det_post) + gamma2_post - gamma2_pr + h_const;
  double lprob = pr_count*log(det_pr) - post_count*log(det_post);
  return lprob;
}

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











// /******************************************************************


//                          Shape


// ******************************************************************/




// -- likelihood of proposed mean --
__device__ double calc_shape_mean_ll_sm(double2 mu, double2 prior_mu,
                                     double3 inv_sigma, double det_prior){
  double dx = mu.x - prior_mu.x;
  double dy = mu.y - prior_mu.y;
  double lprob = -1/2.*(dx*dx*inv_sigma.x+2.*dx*dy*inv_sigma.y+dy*dy*inv_sigma.z);
  lprob += -(3/2.)*log(2*M_PI) - log(det_prior)/2.;
  return lprob;
}

// Compute the posterior mode of the covariance matrix
__device__ double3 calc_shape_sigma_mode_simp_sm(ulonglong3 sq_sum, double2 mu,
                                              double3 prior_sigma_s, double2 prior_mu,
                                              int count, int prior_count) {
  
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

    int sigma2_prior = prior_count*prior_count;
    int total_count = count + prior_count;
    sigma_mode.x = (sigma2_prior + sigma_mode.x)/(total_count + 3.0);
    sigma_mode.y = (sigma_mode.y) / (total_count + 3.0);
    sigma_mode.z = (sigma2_prior + sigma_mode.z)/(total_count + 3.0);

    return sigma_mode;
}



__device__ double calc_shape_sigma_ll_sm(double3 sigma_s, double3 prior_sigma,
                                      double det_sigma, int df){

    // Compute the determinants
    double det_prior = determinant2x2_sm(prior_sigma); // Determinant of prior covariance

    // Inverse of the prior covariance matrix
    double3 inv_prior_sigma = inverse2x2_sm(prior_sigma,det_prior);

    // Compute trace of (inv(prior_sigma) * sigma_s)
    double trace_term = trace2x2_sm(inv_prior_sigma, sigma_s);

    // Compute log-likelihood for inverse Wishart distribution
    double lprob = (df / 2.0) * log(det_prior) - ((df + 3 + 1) / 2.0) * log(det_sigma) - 0.5 * trace_term - lgamma(df/2.0) - lgamma((df-1)/2.0) - log(M_PI)/2.0 - df*log(2);

    // Save the computed log likelihood into the helper structure
    return lprob;
}









/************************************************************


                   Helper Functions


************************************************************/





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

// Function to compute the inverse of a symmetric 2x2 covariance matrix
__device__ double3 inverse2x2_sm(double3 sigma, double det) {
  // double det = determinant2x2(sigma);
    double3 inv_sigma;

    // Inverse of 2x2 symmetric matrix:
    // [sigma11 sigma12]^-1 = 1/det * [ sigma22  -sigma12 ]
    // [sigma12 sigma22]        [ -sigma12  sigma11 ]

    inv_sigma.x = sigma.z / det;   // sigma22 / det
    inv_sigma.y = -sigma.y / det;  // -sigma12 / det
    inv_sigma.z = sigma.x / det;   // sigma11 / det

    return inv_sigma;
}



// Function to compute the trace of (inv(Sigma_prior) * Sigma_sample)
__device__ double trace2x2_sm(double3 inv_sigma_prior, double3 sigma) {
    // Trace(inv(Sigma_prior) * Sigma_sample) for symmetric 2x2 matrix:
    // Tr([inv11 inv12] * [sigma11 sigma12],
    //    [inv12 inv22]   [sigma12 sigma22])
    //  = inv11 * sigma11 + inv22 * sigma22 + 2 * inv12 * sigma12
    return inv_sigma_prior.x * sigma.x + inv_sigma_prior.z * sigma.z + 2.0 * inv_sigma_prior.y * sigma.y;
}



