
#include "cuda.h"
#include "cuda_runtime.h"
#define THREADS_PER_BLOCK 1024
#include "update_prop_params.h"
#include <math.h>

/**********************************************************

             -=-=-=-=- Main Function -=-=-=-=-=-

***********************************************************/

__host__ void update_prop_params(const float* img, const int* spix,
                                 superpixel_params* sp_params,
                                 superpixel_GPU_helper* sp_gpu_helper,
                                 int* prev_means, int* prev_spix,
                                 const int npixels, const int nspix,
                                 const int nspix_buffer, const int nbatch,
                                 const int xdim, const int ydim, const int nftrs){

  	dim3 ThreadPerBlock(THREADS_PER_BLOCK,1);
    int num_block1 = ceil( double(npixels) / double(THREADS_PER_BLOCK) ); 
	int num_block2 = ceil( double(nspix_buffer) / double(THREADS_PER_BLOCK) );
    dim3 BlockPerGrid1(num_block1,nbatch);
    dim3 BlockPerGrid2(num_block2,nbatch);
    clear_fields<<<BlockPerGrid2,ThreadPerBlock>>>(sp_params,sp_gpu_helper,
                                                   nspix,nspix_buffer,nftrs);
	cudaMemset(sp_gpu_helper, 0, nspix_buffer*sizeof(superpixel_GPU_helper));
    sum_by_label<<<BlockPerGrid1,ThreadPerBlock>>>(img,spix,sp_params,sp_gpu_helper,
                                                   npixels,nbatch,xdim,nftrs);
	calculate_mu_and_sigma<<<BlockPerGrid2,ThreadPerBlock>>>(\
     sp_params, sp_gpu_helper, prev_means, prev_spix, nspix, nspix_buffer); 

}

__global__
void clear_fields(superpixel_params* sp_params,
                  superpixel_GPU_helper* sp_gpu_helper,
                  const int nsuperpixel,
                  const int nsuperpixel_buffer,
                  const int nftrs){

	int k = threadIdx.x + blockIdx.x * blockDim.x;  // the label
	if (k>=nsuperpixel_buffer) return;
	if (sp_params[k].valid == 0) return;

	sp_params[k].count = 0;
	sp_params[k].log_count = 0.1;
	
	float3 mu_i;
	mu_i.x = 0;
	mu_i.y = 0;
	mu_i.z = 0;
	sp_params[k].mu_i = mu_i;

	double2 mu_s;
	mu_s.x = 0;
	mu_s.y = 0;
	sp_params[k].mu_s = mu_s;
}


__global__
void sum_by_label(const float* img,
                  const int* spix, superpixel_params* sp_params,
                  superpixel_GPU_helper* sp_gpu_helper,
                  const int npixels, const int nbatch,
                  const int xdim, const int nftrs) {
    // todo -- add nbatch and nftrs
    // getting the index of the pixel
    int t = threadIdx.x + blockIdx.x * blockDim.x;
	if (t>=npixels) return;

	//get the label
	int k = spix[t];
    if (k == -1){ return; } // invalid label

	atomicAdd(&sp_params[k].count, 1);
	atomicAdd(&sp_gpu_helper[k].mu_i_sum.x, img[3*t]);
	atomicAdd(&sp_gpu_helper[k].mu_i_sum.y, img[3*t+1]);
	atomicAdd(&sp_gpu_helper[k].mu_i_sum.z, img[3*t+2]);


	int x = t % xdim;
	int y = t / xdim; 
	int xx = x * x;
	int xy = x * y;
	int yy = y * y;

	atomicAdd(&sp_gpu_helper[k].mu_s_sum.x, x);
	atomicAdd(&sp_gpu_helper[k].mu_s_sum.y, y);
    atomicAdd((unsigned long long *)&sp_gpu_helper[k].sigma_s_sum.x, xx);
	atomicAdd((unsigned long long *)&sp_gpu_helper[k].sigma_s_sum.y, xy);
	atomicAdd((unsigned long long *)&sp_gpu_helper[k].sigma_s_sum.z, yy);
	
}



__global__
void calculate_mu_and_sigma(superpixel_params*  sp_params,
                            superpixel_GPU_helper* sp_gpu_helper,
                            int* prev_means, int* prev_spix,
                            const int nsuperpixel, const int nsuperpixel_buffer) {

    // -- update thread --
	int k = threadIdx.x + blockIdx.x * blockDim.x; // the label
	if (k>=nsuperpixel_buffer) return;
	if (sp_params[k].valid == 0) return;
    
    // -- read previou spix info --
    int prev_k = prev_spix[k];
    // int* means_prev_s = means_prev + prev_k*5;
    // int prev_mu_i_x = means_prev_s[0];
    // int prev_mu_i_y = means_prev_s[1];
    // int prev_mu_i_z = means_prev_s[1];
    // int prev_mu_s_x = means_prev_s[2];
    // int prev_mu_s_y = means_prev_s[3];

    // -- read curr --
	int count_int = sp_params[k].count;
	float a_prior = sp_params[k].prior_count;
	float prior_sigma_s_2 = a_prior * a_prior;
	double count = count_int * 1.0;
	double mu_x = 0.0;
	double mu_y = 0.0;

	// -- calculate the mean --
	if (count_int>0){

		sp_params[k].log_count = log(count);
	    mu_x = sp_gpu_helper[k].mu_s_sum.x / count;
	    mu_y = sp_gpu_helper[k].mu_s_sum.y / count;
		sp_params[k].mu_s.x = mu_x;
	    sp_params[k].mu_s.y = mu_y;
        
	    sp_params[k].mu_i.x = sp_gpu_helper[k].mu_i_sum.x / count;
		sp_params[k].mu_i.y = sp_gpu_helper[k].mu_i_sum.y / count;
  		sp_params[k].mu_i.z = sp_gpu_helper[k].mu_i_sum.z / count;

	}

	// -- calculate the covariance --
	double C00 = sp_gpu_helper[k].sigma_s_sum.x;
	double C01 = sp_gpu_helper[k].sigma_s_sum.y;
	double C11 = sp_gpu_helper[k].sigma_s_sum.z; 
	double total_count = (double) sp_params[k].count + a_prior;
	if (count_int > 3){
	    C00 = C00 - mu_x * mu_x * count;
	    C01 = C01 - mu_x * mu_y * count;
	    C11 = C11 - mu_y * mu_y * count;
	}

    // -- invert cov matrix --
    C00 = (prior_sigma_s_2 + C00) / (total_count - 3.0);
    C01 = C01 / (total_count - 3);
    C11 = (prior_sigma_s_2 + C11) / (total_count - 3.0);

    double detC = C00 * C11 - C01 * C01;
    if (detC <= 0){
      C00 = C00 + 0.00001;
      C11 = C11 + 0.00001;
      detC = C00*C11-C01*C01;
      if(detC <=0) detC = 0.0001;//hack
    }

    // -- finish-up inverse cov --
    sp_params[k].sigma_s.x = C11 / detC;     
    sp_params[k].sigma_s.y = -C01 / detC; 
    sp_params[k].sigma_s.z = C00 / detC; 
    sp_params[k].logdet_Sigma_s = log(detC);

}
