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


#include <assert.h>
#include "update_prop_seg.h"
#ifndef MY_SP_SHARE_H
#define MY_SP_SHARE_H
#include "../bass/share/sp.h"
#endif
#include "update_seg_helper.h"
// #include <tuple>

#include <stdio.h>
#ifndef WIN32
#include <unistd.h>
#endif

__host__ void update_prop_seg(float* img, int* seg,
                              int* seg_potts_label ,bool* border,
                              superpixel_params* sp_params, 
                              superpixel_params* sp_params_prev, 
                              superpixel_GPU_helper* sp_gpu_helper,
                              const float3 J_i, const float logdet_Sigma_i, 
                              bool cal_cov, float i_std, int s_std,
                              int nInnerIters, const int nPixels,
                              const int nSPs, int nSPs_buffer,
                              int nbatch, int xdim, int ydim, int nftrs,
                              float beta_potts_term, bool use_transition,
                              float* debug_seg){
    
    int num_block = ceil( double(nPixels) / double(THREADS_PER_BLOCK) ); 
    // int num_block2 = ceil( double(nPixels*4) / double(THREADS_PER_BLOCK) ); 

    dim3 ThreadPerBlock(THREADS_PER_BLOCK,1);
    dim3 BlockPerGrid(num_block,nbatch);
    // dim3 BlockPerGrid2(num_block2,nbatch);
    float* debug_seg_i = debug_seg;

    int single_border = 0;
    // cudaMemset(post_changes, 0, nPixels*sizeof(post_changes_helper));
    for (int iter = 0 ; iter < nInnerIters; iter++){

        debug_seg_i = debug_seg + iter * nPixels * 45;

    	// strides of 2*2
        cudaMemset(border, 0, nPixels*sizeof(bool));
        find_border_pixels<<<BlockPerGrid,ThreadPerBlock>>>(seg, border, nPixels,
                                                            nbatch, xdim, ydim,
                                                            single_border);
        // gpuErrchk( cudaPeekAtLastError() );
        // gpuErrchk( cudaDeviceSynchronize() );

        int xmod3 = (rand() > 0.5) ? 1 : 0; // reverse order half of the time
        // bool yorder = (rand() > 0.5);
        for (int _xmod3 = 0 ; _xmod3 <2; _xmod3++){
            int ymod3 = (rand() > 0.5) ? 1 : 0; // reverse order half of the time
            for (int _ymod3 = 0; _ymod3 <2; _ymod3++){
                //find the border pixels
              // if (rand() > 0.75){ continue; }
                update_prop_seg_subset<<<BlockPerGrid,ThreadPerBlock>>>(img, seg, \
                     seg_potts_label,border, sp_params, sp_params_prev, \
                     sp_gpu_helper, J_i, logdet_Sigma_i, cal_cov, \
                     i_std, s_std, nPixels, nSPs, \
                     nbatch, xdim, ydim, nftrs, xmod3, \
                     ymod3, beta_potts_term, use_transition, debug_seg_i);
                // gpuErrchk( cudaPeekAtLastError() );
                // gpuErrchk( cudaDeviceSynchronize() );
                ymod3 = 1 - ymod3; // update x mod 3: 0 -> 1 and 1 -> 0
            }
            xmod3 = 1 - xmod3; // update x mod 3: 0 -> 1 and 1 -> 0
        }


    }
    cudaMemset(border, 0, nPixels*sizeof(bool));
    find_border_pixels<<<BlockPerGrid,ThreadPerBlock>>>(\
           seg, border, nPixels, nbatch, xdim, ydim, single_border);
}


/*
* Update the superpixel labels for pixels 
* that are on the boundary of the superpixels
* and on the (xmod3, ymod3) position of 3*3 block
*/
__global__  void update_prop_seg_subset(
    float* img, int* seg, int* seg_potts_label, bool* border,
    superpixel_params* sp_params, 
    superpixel_params* sp_params_prev, 
    superpixel_GPU_helper* sp_gpu_helper,
    const float3 J_i, const float logdet_Sigma_i,  
    bool cal_cov, float i_std, int s_std, 
    const int nPts,const int nSuperpixels,
    const int nbatch, const int xdim, const int ydim, const int nftrs,
    const int xmod3, const int ymod3, const float beta_potts_term,
    const bool use_transition, float* debug_seg){

    int label_check;
    int idx = threadIdx.x + blockIdx.x*blockDim.x;
   // idx = idx_img;

    int seg_idx = idx; 
    if (seg_idx>=nPts)  return;
    // todo; add batch info here.

    int x = seg_idx % xdim;  
    if (x % 2 != xmod3) return;
    int y = seg_idx / xdim;   
    if (y % 2 != ymod3) return;
    
    if (border[seg_idx]==0) return;
    // strides of 2*2

    //float beta = 4;
    //printf("(%d, %d) - %d, %d, %d \n", x,y , idx_cache,threadIdx.x );
    const bool x_greater_than_1 = (x>1);
    const bool y_greater_than_1 = (y>1);
    const bool x_smaller_than_xdim_minus_1 = x<(xdim-1);
    const bool y_smaller_than_ydim_minus_1 = y<(ydim-1);
    if ((!x_greater_than_1)||(!y_greater_than_1)||(!x_smaller_than_xdim_minus_1)||(!y_smaller_than_ydim_minus_1)) return;
   
   /*if(sp_params[ seg[seg_idx]].count==1) 
    {
        seg[seg_idx]=seg[seg_idx-1];
        return;
    }*/

    
    int C = seg[seg_idx]; // center 

    // N = S = W = E = OUT_OF_BOUNDS_LABEL; // init to out-of-bounds 
    
    bool nbrs[9];
    //float potts_term[4];
    //potts_term[0] = potts_term[1] = potts_term[2] = potts_term[3] = 0;
    bool isNvalid = 0;
    bool isSvalid = 0;
    bool isEvalid = 0;
    bool isWvalid = 0; 
    float beta = beta_potts_term;
    //printf("Beta: %f", beta);

    int count_diff_nbrs_N=0;
    int count_diff_nbrs_S=0;
    int count_diff_nbrs_E=0;
    int count_diff_nbrs_W=0;

//NW =N = NE =W = E = SW = S = SE=5 ;
// init       

    float2 res_max;
    res_max.x = -9999;
    //post_changes[seg_idx].post[4] = -9999;
    int NW =__ldg(&seg[seg_idx-xdim-1]);
    int N = __ldg(&seg[seg_idx-xdim]);
    int NE = __ldg(&seg[seg_idx-xdim+1]);
    int W = __ldg(&seg[seg_idx-1]);
    int E = __ldg(&seg[seg_idx+1]);
    int SW = __ldg(&seg[seg_idx+xdim-1]);
    int S = __ldg(&seg[seg_idx+xdim]);
    int SE =__ldg(&seg[seg_idx+xdim+1]);  
    
    //N :
    set_nbrs(NW, N, NE,  W, E, SW, S, SE, N, nbrs);
    count_diff_nbrs_N = ischangbale_by_nbrs(nbrs);
    isNvalid = nbrs[8];
    if(!isNvalid) return;
    
    //W :
    set_nbrs(NW, N, NE,  W, E, SW, S, SE, W, nbrs);
    count_diff_nbrs_W = ischangbale_by_nbrs(nbrs);
    isWvalid = nbrs[8];
    if(!isWvalid) return;

    //S :
    set_nbrs(NW, N, NE,  W, E, SW, S, SE, S, nbrs);
    count_diff_nbrs_S = ischangbale_by_nbrs(nbrs);
    isSvalid = nbrs[8];
    if(!isSvalid) return;

    //E:
    set_nbrs(NW, N, NE,  W, E, SW, S, SE, E, nbrs);
    // check 8 nbrs and save result if valid to change to the last place of array
    // return how many nbrs different for potts term calculation
    count_diff_nbrs_E = ischangbale_by_nbrs(nbrs);
    isEvalid = nbrs[8];
    if(!isEvalid) return;
   
    // -- index image --
    float* imgC = img + idx * 3;
    float* debugC = debug_seg + idx * 45;
    // debugC[44] = 1;

    // -- compute posterior --
    label_check = N;
    assert(label_check >= 0);
    float xfer0,xfer1;
    xfer0=xfer1=0;
    res_max = cal_posterior_prop(imgC,seg,x,y,sp_params,sp_params_prev,
                                 label_check,J_i,logdet_Sigma_i,i_std,s_std,
                                 count_diff_nbrs_N,beta,res_max,
                                 xfer0,xfer1,debugC);
    // int P = -1;
    // float xfer_P = 0; // p for "Pick"
    label_check = S;
    assert(label_check >= 0);
    if(label_check!=N){
      // if (use_transition){
      //   calc_transition(xfer0,xfer1,N,S,C,imgC,x,y,
      //                   sp_params,sp_params_prev,sp_gpu_helper);
      // }
      res_max = cal_posterior_prop(imgC,seg,x,y,sp_params,sp_params_prev,
                                   label_check,J_i,logdet_Sigma_i,i_std,s_std,
                                   count_diff_nbrs_S,beta,res_max,
                                   xfer0,xfer1,debugC+11);
    }

    label_check = W;
    assert(label_check >= 0);
    if((label_check!=S)&&(label_check!=N)){
      // if (use_transition){
      //   calc_transition(xfer0,xfer1,res_max.y,label_check,C,imgC,x,y,
      //                   sp_params,sp_params_prev,sp_gpu_helper);
      // }
      res_max = cal_posterior_prop(imgC,seg,x,y,sp_params,sp_params_prev,
                                   label_check,J_i,logdet_Sigma_i,i_std,s_std,
                                   count_diff_nbrs_W,beta,res_max,
                                   xfer0,xfer1,debugC+22);
    }
    
    label_check = E;
    assert(label_check >= 0);
    if((label_check!=W)&&(label_check!=S)&&(label_check!=N)){
      // if (use_transition){      
      //   calc_transition(xfer0,xfer1,res_max.y,label_check,
      //                   C,imgC,x,y,sp_params,sp_params_prev,sp_gpu_helper);
      // }
      res_max = cal_posterior_prop(imgC,seg,x,y, sp_params, sp_params_prev,
                                   label_check,J_i, logdet_Sigma_i,i_std,s_std,
                                   count_diff_nbrs_E,beta,res_max,
                                   xfer0,xfer1,debugC+33);
    }

    seg[seg_idx] = res_max.y;
    return;

}



// __device__ inline
// float calc_mle_update(float* pix, int x, int y,
//                       superpixel_params* sp_params, 
//                       superpixel_params* sp_params_prev, 
//                       superpixel_GPU_helper* sp_gpu_helper,
//                       int spix_p, int spix_c){
  
//   // -=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-
//   // --        color means          --
//   // -=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-

//   float mu_i_x_curr = sp_gpu_helper[spix_c].mu_i_sum.x - pix[0];
//   float mu_i_y_curr = sp_gpu_helper[spix_c].mu_i_sum.y - pix[1];
//   float mu_i_z_curr = sp_gpu_helper[spix_c].mu_i_sum.z - pix[2];

//   float mu_i_x_prop = sp_gpu_helper[spix_p].mu_i_sum.x + pix[0];
//   float mu_i_y_prop = sp_gpu_helper[spix_p].mu_i_sum.y + pix[1];
//   float mu_i_z_prop = sp_gpu_helper[spix_p].mu_i_sum.z + pix[2];

//   // -=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-
//   // --       spatial cov           --
//   // -=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-

//   int xx = x * x;
//   int xy = x * y;
//   int yy = y * y;

//   int count_int = sp_gpu_helper[spix_c].count;
//   double C00 = sp_gpu_helper[spix_c].sigma_s_sum.x - xx;
//   double C01 =  sp_gpu_helper[spix_c].sigma_s_sum.y - xy;
//   double C11 = sp_gpu_helper[spix_c].sigma_s_sum.z - yy; 
//   double total_count = (double) sp_params[spix_c].count + a_prior;
//   double4 cov_curr = get_spatial_cov(C00,C01,C11,count_int,total_count);

//   count_int = sp_gpu_helper[spix_p].count;
//   C00 = sp_gpu_helper[spix_p].sigma_s_sum.x + xx;
//   C01 =  sp_gpu_helper[spix_p].sigma_s_sum.y + xy;
//   C11 = sp_gpu_helper[spix_p].sigma_s_sum.z + yy; 
//   total_count = (double) sp_params[spix_p].count + a_prior;
//   double4 cov_prop = get_spatial_cov(C00,C01,C11,count_int,total_count);


//   // double C00 = sp_gpu_helper[k].sigma_s_sum.x;
//   // double C01 =  sp_gpu_helper[k].sigma_s_sum.y ;
//   // double C11 = sp_gpu_helper[k].sigma_s_sum.z; 


//   // float y2_curr = sp_gpu_helper[spix_c].sum_y2 + y*y;
//   // float x2_curr = sp_gpu_helper[spix_c].sum_x2 + x*x;


// 	// 	sp_params[k].log_count = log(count);
// 	//     mu_x = sp_gpu_helper[k].mu_s_sum.x / count;   
// 	//     mu_y = sp_gpu_helper[k].mu_s_sum.y / count;  
// 	// 	sp_params[k].mu_s.x = mu_x; 
// 	//     sp_params[k].mu_s.y = mu_y;

// 	//     sp_params[k].mu_i.x = sp_gpu_helper[k].mu_i_sum.x / count;
// 	// 	sp_params[k].mu_i.y = sp_gpu_helper[k].mu_i_sum.y / count;
//   	// 	sp_params[k].mu_i.z = sp_gpu_helper[k].mu_i_sum.z / count;

// 	//    /* sp_params[k].sigma_s.x = sp_gpu_helper[k].mu_i_sum.x / count *0;
// 	// 	sp_params[k].sigma_s.y = sp_gpu_helper[k].mu_i_sum.y / count *0;
//   	// 	sp_params[k].sigma_s.z = sp_gpu_helper[k].mu_i_sum.z / count * 0;	
//     //    */
// 	// 	//printf(" k is %d , %f,  %f, %f\n",k,sp_gpu_helper[k].mu_i_sum.x, sp_gpu_helper[k].mu_i_sum.y,sp_gpu_helper[k].mu_i_sum.z);
// 	// }

// 	// //calculate the covariance
	
// 	// double C00 = sp_gpu_helper[k].sigma_s_sum.x ;
// 	// double C01 =  sp_gpu_helper[k].sigma_s_sum.y ;
// 	// double C11 = sp_gpu_helper[k].sigma_s_sum.z; 


// }



// /***********************************************************************


//                     Update and Read Cov


// ***********************************************************************/

// __device__ inline
// double4 get_updated_cov(int x, int y, superpixel_GPU_helper* sp_gpu_helper,
//                         superpixel_params* sp_params, int spix, int sign){
//   // mu_x,mu_y already updated
//   int count = sp_params[spix].count + sign;
//   float mu_x = (sp_gpu_helper[k].mu_s_sum.x + sign*x)/count;
//   float mu_y = (sp_gpu_helper[k].mu_s_sum.y + sign*x)/count;
//   double C00 = sp_gpu_helper[spix].sigma_s_sum.x + sign*xx;
//   double C01 =  sp_gpu_helper[spix].sigma_s_sum.y + sign*xy;
//   double C11 = sp_gpu_helper[spix].sigma_s_sum.z + sign*yy; 
//   double total_count = (double) count + a_prior;
//   double4 cov = compute_inv_cov(C00,C01,C11,count,total_count,mu_x,mu_y);
//   return cov;
// }  

// __device__ inline
// double4 compute_inv_cov(double C00, double C01, double C11,
//                         int count_int, int total_count){

//   if (count_int > 3){	    
//     //update cumulative count and covariance
//     C00 = C00 - mu_x * mu_x * count;
//     C01 = C01 - mu_x * mu_y * count;
//     C11 = C11 - mu_y * mu_y * count;
//   }

//   C00 = (prior_sigma_s_2 + C00) / (total_count - 3.0);
//   C01 = C01 / (total_count - 3);
//   C11 = (prior_sigma_s_2 + C11) / (total_count - 3.0);

//   double detC = C00 * C11 - C01 * C01;
//   if (detC <= 0){
//       C00 = C00 + 0.00001;
//       C11 = C11 + 0.00001;
//       detC = C00*C11-C01*C01;
//       if(detC <=0) detC = 0.0001;//hack
//   }

//   //Take the inverse of sigma_space to get J_space
//   double4 inv_cov;
//   inv_cov.x = C11 / detC;
//   inv_cov.y = -C01 / detC;
//   inv_cov.z = C00 / detC;
//   inv_cov.w = log(detC);
//   return inv_cov;
// }

