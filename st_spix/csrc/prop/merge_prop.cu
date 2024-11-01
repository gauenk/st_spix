
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

#include "merge_prop.h"
// #include "update_params.h"

#include <stdio.h>
#ifndef WIN32
#include <unistd.h>
#endif


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

