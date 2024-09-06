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
#include "update_seg.h"
#include "../share/sp.h"

#include <stdio.h>
#ifndef WIN32
#include <unistd.h>
#endif


__device__ __forceinline__ float atomicMaxFloat (float * addr, float value) {
    float old;
    old = (value >= 0) ? __int_as_float(atomicMax((int *)addr, __float_as_int(value))) :
         __uint_as_float(atomicMin((unsigned int *)addr, __float_as_uint(value)));

    return old;
}
__host__ void CudaFindBorderPixels(const int* seg, bool* border, const int nPixels, const int nbatch,const int xdim,const int ydim, const int single_border){
    int num_block = ceil( double(nPixels) / double(THREADS_PER_BLOCK) ); 
    dim3 ThreadPerBlock(THREADS_PER_BLOCK,1);
    dim3 BlockPerGrid(num_block,nbatch);
    find_border_pixels<<<BlockPerGrid,ThreadPerBlock>>>(seg,border,nPixels,
                                                        nbatch, xdim, ydim,
                                                        single_border);
}

__host__ void CudaFindBorderPixels_end(const int* seg, bool* border, const int nPixels, const int nbatch,const int xdim,const int ydim, const int single_border){   
    int num_block = ceil( double(nPixels) / double(THREADS_PER_BLOCK) ); 
    dim3 ThreadPerBlock(THREADS_PER_BLOCK,1);
    dim3 BlockPerGrid(num_block,nbatch);
    find_border_pixels_end<<<BlockPerGrid,ThreadPerBlock>>>(seg,border,nPixels,
                                                            nbatch, xdim, ydim,
                                                            single_border);
}




__global__  void find_border_pixels(const int* seg, bool* border, const int nPixels,
                                    const int nbatch, const int xdim, const int ydim,
                                    const int single_border){   
    int idx = threadIdx.x + blockIdx.x * blockDim.x;  
    if (idx>=nPixels) return; 

    border[idx]=0;  // init        
    // todo; add batch info here
    int x = idx % xdim;
    int y = idx / xdim;

    int C =  __ldg(&seg[idx]); // center 
    int N,S,E,W; // north, south, east,west            

    // -- check out of bounds --
    if ((y<1)||(x<1)||(y>=(ydim-1))||(x>=(xdim-1)))
    {
        border[idx] = 1;
        return;
    }
    N = __ldg(&seg[idx-xdim]); // above
    W = __ldg(&seg[idx-1]);  // left
    S = __ldg(&seg[idx+xdim]); // below
    E = __ldg(&seg[idx+1]);  // right
           
    // bool check0 = (C == N) and (N == W) and (W == S);
    if ( (C!=N) || (C!=S) || (C!=E) || (C!=W) ){
            border[idx]=1;  
    }


    return;        
}


__global__  void find_border_pixels_end(const int* seg, bool* border, const int nPixels,
                                        const int nbatch, const int xdim, const int ydim,
                                        const int single_border){   
    int idx = threadIdx.x + blockIdx.x * blockDim.x;  
    if (idx>=nPixels) return; 

    border[idx]=0;  // init        
    
    // todo; add batch info here
    int x = idx % xdim;
    int y = idx / xdim;

    int C = seg[idx]; // center 
    int N,S,E,W; // north, south, east,west            
    N=S=W=E=OUT_OF_BOUNDS_LABEL; // init 
    
    if (y>1){
        N = seg[idx-xdim]; // above
    }          
    if (x>1){
        W = seg[idx-1];  // left
    }
    if (y<ydim-1){
        S = seg[idx+xdim]; // below
    }   
    if (x<xdim-1){
        E = seg[idx+1];  // right
    }       
   
    // If the nbr is different from the central pixel and is not out-of-bounds,
    // then it is a border pixel.
    if ( (N>=0 && C!=N) || (S>=0 && C!=S) || (E>=0 && C!=E) || (W>=0 && C!=W) ){
            if (N>=0 && C>N) border[idx]=1; 
            if (S>=0 && C>S) border[idx]=1;
            if (E>=0 && C>E) border[idx]=1;
            if (W>=0 && C>W) border[idx]=1;
    }

    return;        
}



// __global__ void cal_posterior( float* img, int* seg, bool* border,
//                                superpixel_params* sp_params,
//                                float3 J_i, float logdet_Sigma_i,
//                                float i_std, int s_std, int* changes,
//                                int nPts , int xdim,
//                                post_changes_helper* post_changes)
// {
    
//     int idx_res = threadIdx.x + blockIdx.x*blockDim.x;
//     int idx_inside = idx_res%4;
//     int idx = idx_res/4;
// 	if (idx_res>=4*nPts)  return;
//     if (border[idx]==0) return;

//     int post = post_changes[idx].skip_post[idx_inside];
//     int seg_idx = post_changes[idx].changes[idx_inside];
//     if(post || ! seg_idx) return;
//     float res = -1000; // some large negative number

//     float* imgC = img + idx * 3;
//     int x = idx % xdim;  
//     int y = idx / xdim;  
//     //if (idx>154064)
//     //    printf("%d ,%d , %d\n",idx_inside, idx, seg_idx);
//     const float x0 = imgC[0]-sp_params[seg_idx].mu_i.x;
//     const float x1 = imgC[1]-sp_params[seg_idx].mu_i.y;
//     const float x2 = imgC[2]-sp_params[seg_idx].mu_i.z;

//     const int d0 = x - sp_params[seg_idx].mu_s.x;
//     const int d1 = y - sp_params[seg_idx].mu_s.y;
//     //color component
//     const float J_i_x = J_i.x;
//     const float J_i_y = J_i.y;
//     const float J_i_z = J_i.z;
//     const float sigma_s_x = sp_params[seg_idx].sigma_s.x;
//     const float sigma_s_y = sp_params[seg_idx].sigma_s.y;
//     const float sigma_s_z = sp_params[seg_idx].sigma_s.z;
//     const float logdet_sigma_s = sp_params[seg_idx].logdet_Sigma_s;


//     res = res - (x0*x0*J_i_x + x1*x1*J_i_y + x2*x2*J_i_z);   //res = -calc_squared_mahal_3d(imgC,mu_i,J_i);
//     res = res -logdet_Sigma_i;
//     //space component
//     res = res - d0*d0*sigma_s_x;
//     res = res - d1*d1*sigma_s_z;
//     res = res -  2*d0*d1*sigma_s_y;            // res -= calc_squared_mahal_2d(pt,mu_s,J_s);
//     res = res -  logdet_sigma_s;
//     if (res > atomicMaxFloat(&post_changes[idx].post[4],res))
//     seg[idx] = seg_idx;

    
    
    
//     //res += potts_res;

//     return;        
// }

__host__ void update_seg(float* img, int* seg, int* seg_potts_label ,bool* border,
                         superpixel_params* sp_params, 
                         const float3 J_i, const float logdet_Sigma_i, 
                         bool cal_cov, float i_std, int s_std,
                         int nInnerIters, const int nPixels,
                         const int nSPs, int nSPs_buffer,
                         int nbatch, int xdim, int ydim, int nftrs,
                         float beta_potts_term){
    
    int num_block = ceil( double(nPixels) / double(THREADS_PER_BLOCK) ); 
    // int num_block2 = ceil( double(nPixels*4) / double(THREADS_PER_BLOCK) ); 

    dim3 ThreadPerBlock(THREADS_PER_BLOCK,1);
    dim3 BlockPerGrid(num_block,nbatch);
    // dim3 BlockPerGrid2(num_block2,nbatch);

    int single_border = 0;
    // cudaMemset(post_changes, 0, nPixels*sizeof(post_changes_helper));
    for (int iter = 0 ; iter < nInnerIters; iter++){
    	// strides of 2*2
        // cudaMemset(border, 0, nPixels*sizeof(bool));
        cudaMemset(border, 0, nPixels*sizeof(bool));
        find_border_pixels<<<BlockPerGrid,ThreadPerBlock>>>(seg, border, nPixels,
                                                            nbatch, xdim, ydim,
                                                            single_border);
        // gpuErrchk( cudaPeekAtLastError() );
        // gpuErrchk( cudaDeviceSynchronize() );

        for (int xmod3 = 0 ; xmod3 <2; xmod3++){
            for (int ymod3 = 0; ymod3 <2; ymod3++){
                //find the border pixels
                update_seg_subset<<<BlockPerGrid,ThreadPerBlock>>>(img, seg, \
                     seg_potts_label,border, sp_params, J_i, logdet_Sigma_i,\
                     cal_cov, i_std, s_std, nPixels, nSPs, \
                     nbatch, xdim, ydim, nftrs, xmod3, \
                     ymod3, beta_potts_term);
                // gpuErrchk( cudaPeekAtLastError() );
                // gpuErrchk( cudaDeviceSynchronize() );

            }
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
__global__  void update_seg_subset(
    float* img, int* seg, int* seg_potts_label, bool* border,
    superpixel_params* sp_params, 
    const float3 J_i, const float logdet_Sigma_i,  
    bool cal_cov, float i_std, int s_std, 
    const int nPts,const int nSuperpixels,
    const int nbatch, const int xdim, const int ydim, const int nftrs,
    const int xmod3, const int ymod3,
    const float beta_potts_term){

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

    
    //int C = seg[seg_idx]; // center 

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

    // -- compute posterior --
    label_check = N;
    assert(label_check >= 0);
    res_max = cal_posterior_new(imgC,seg,x,y,sp_params,label_check,
                                J_i,logdet_Sigma_i,i_std,s_std,
                                count_diff_nbrs_N,beta,res_max);
    label_check = S;
    assert(label_check >= 0);
    if(label_check!=N)
    res_max = cal_posterior_new(imgC,seg,x,y,sp_params,label_check,
                                J_i,logdet_Sigma_i,i_std,s_std,
                                count_diff_nbrs_S,beta,res_max);

    label_check = W;
    assert(label_check >= 0);
    if((label_check!=S)&&(label_check!=N))   
    res_max = cal_posterior_new(imgC,seg,x,y,sp_params,label_check,J_i,
                                logdet_Sigma_i,i_std,s_std,
                                count_diff_nbrs_W,beta,res_max);
    
    label_check = E;
    assert(label_check >= 0);
    if((label_check!=W)&&(label_check!=S)&&(label_check!=N))      
    res_max = cal_posterior_new(imgC,seg,x,y,sp_params,label_check,J_i,
                                logdet_Sigma_i,i_std,s_std,
                                count_diff_nbrs_E,beta,res_max);
    seg[seg_idx] = res_max.y;
    return;
}
// __device__ inline float2 cal_posterior_new(
//     float* imgC, int* seg, int x, int y,
//     superpixel_params* sp_params,
//     int seg_idx, float3 J_i,
//     float logdet_Sigma_i, float i_std, int s_std,
//     post_changes_helper* post_changes, float potts,
//     float beta, float2 res_max){

