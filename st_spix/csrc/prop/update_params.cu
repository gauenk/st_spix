
// #include "cuda.h"
// #include "cuda_runtime.h"
// #define THREADS_PER_BLOCK 1024
#include "update_params.h"
#include <math.h>

/**********************************************************

             -=-=-=-=- Main Function -=-=-=-=-=-

***********************************************************/

__host__ void update_params(const float* img, const int* spix,
                            spix_params* sp_params,spix_helper* sp_helper,
                            const int npixels, const int nspix_buffer,
                            const int nbatch, const int width, const int nftrs){

  	dim3 ThreadPerBlock(THREADS_PER_BLOCK,1);
    int num_block1 = ceil( double(npixels) / double(THREADS_PER_BLOCK) ); 
	int num_block2 = ceil( double(nspix_buffer) / double(THREADS_PER_BLOCK) );
    dim3 BlockPerGrid1(num_block1,nbatch);
    dim3 BlockPerGrid2(num_block2,nbatch);
    clear_fields<<<BlockPerGrid2,ThreadPerBlock>>>(sp_params,sp_helper,
                                                   nspix_buffer,nftrs);
	cudaMemset(sp_helper, 0, nspix_buffer*sizeof(spix_helper));
    sum_by_label<<<BlockPerGrid1,ThreadPerBlock>>>(img,spix,sp_params,sp_helper,
                                                   npixels,nbatch,width,nftrs);
	calculate_mu_and_sigma<<<BlockPerGrid2,ThreadPerBlock>>>(sp_params,
                                                             sp_helper, nspix_buffer); 
}

__global__
void clear_fields(spix_params* sp_params,
                  spix_helper* sp_helper,
                  const int nsuperpixel_buffer,
                  const int nftrs){

	int k = threadIdx.x + blockIdx.x * blockDim.x;  // the label
	if (k>=nsuperpixel_buffer) return;
	if (sp_params[k].valid == 0) return;
	sp_params[k].count = 0;
	
	float3 mu_i;
	mu_i.x = 0;
	mu_i.y = 0;
	mu_i.z = 0;
	sp_params[k].mu_i = mu_i;

    float3 sigma_i;
    sigma_i.x = 0;
    sigma_i.y = 0;
    sigma_i.z = 0;
	sp_params[k].sigma_i = sigma_i;

	double2 mu_s;
	mu_s.x = 0;
	mu_s.y = 0;
	sp_params[k].mu_s = mu_s;
}


__global__
void sum_by_label(const float* img,
                  const int* spix, spix_params* sp_params,
                  spix_helper* sp_helper,
                  const int npixels, const int nbatch,
                  const int width, const int nftrs) {

    // todo -- add nbatch and nftrs
    // getting the index of the pixel
    int t = threadIdx.x + blockIdx.x * blockDim.x;
	if (t>=npixels) return;

	//get the label
	int k = spix[t];
    if (k == -1){ return; } // invalid label

    float3 pix = *(float3*)(img+3*t);
	atomicAdd(&sp_params[k].count, 1);
	atomicAdd(&sp_helper[k].mu_i_sum.x, pix.x);
	atomicAdd(&sp_helper[k].mu_i_sum.y, pix.y);
	atomicAdd(&sp_helper[k].mu_i_sum.z, pix.z);
	atomicAdd(&sp_helper[k].sigma_i_sum.x, pix.x*pix.x);
	atomicAdd(&sp_helper[k].sigma_i_sum.y, pix.x*pix.y);
	atomicAdd(&sp_helper[k].sigma_i_sum.z, pix.y*pix.y);

    // -- pix variance --
	// atomicAdd(&sp_helper[k].sigma_i_sum.x, img[3*t]);
	// atomicAdd(&sp_helper[k].sigma_i_sum.y, img[3*t+1]);
	// atomicAdd(&sp_helper[k].sigma_i_sum.z, img[3*t+2]);

	int x = t % width;
	int y = t / width; 
	int xx = x * x;
	int xy = x * y;
	int yy = y * y;

	atomicAdd(&sp_helper[k].mu_s_sum.x, x);
	atomicAdd(&sp_helper[k].mu_s_sum.y, y);
    atomicAdd((unsigned long long *)&sp_helper[k].sigma_s_sum.x, xx);
	atomicAdd((unsigned long long *)&sp_helper[k].sigma_s_sum.y, xy);
	atomicAdd((unsigned long long *)&sp_helper[k].sigma_s_sum.z, yy);
	
}


__global__
void calculate_mu_and_sigma(spix_params*  sp_params,
                            spix_helper* sp_helper,
                            const int nsuperpixel_buffer) {

    // -- update thread --
	int k = threadIdx.x + blockIdx.x * blockDim.x; // the label
	if (k>=nsuperpixel_buffer) return;
	if (sp_params[k].valid == 0) return;
    
    // -- read previou spix info --
    // int prev_spix = sp_params[
	// int parent_spix = sp_params[k].parent_spix;
    // bool has_parent = parent_spix >= 0;

    // if (has_parent){
    //   prev_params[parent_spix].count;
    // }
    // int prev_k = prev_spix[k];
    // if (prev_k != -1){
    //   pass
    // }
  
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
    double2 prior_mu_s = sp_params[k].prior_mu_s;
	double count = count_int * 1.0;
    double2 mu_s;
    float3 mu_i;
	// double mu_x = 0.0;
	// double mu_y = 0.0;

	// -- calculate means --
	if (count_int>0){
	    mu_s.x = sp_helper[k].mu_s_sum.x/count;
	    mu_s.y = sp_helper[k].mu_s_sum.y/count;
        mu_i.x = sp_helper[k].mu_i_sum.x/count;
        mu_i.y = sp_helper[k].mu_i_sum.y/count;
        mu_i.z = sp_helper[k].mu_i_sum.z/count;
	}

    /**************************************************
   
                     Appearance Information

    ***************************************************/


    /**************************************************
   
                     Spatial Information

    ***************************************************/

    // -- sigma mode --
    int df = count;
    int lam = count;
    double3 prior_sigma_s = sp_params[k].prior_sigma_s;
    double3 sigma_mode = compute_cov_mode(sp_helper[k].sigma_s_sum,mu_s,
                                          prior_sigma_s,prior_mu_s,count,lam,df);
    double detC = determinant2x2(sigma_mode);

    // -- mu mode [AFTER sigma mode] --
    // mu_s = compute_mean_mode(mu_s);

    // -- compute covariance matrix prior likelihood --
    sp_params[k].prior_sigma_s_lprob = (float)calc_cov_likelihood(sigma_mode,
                                                                  prior_sigma_s,
                                                                  detC,df+count);
    // -- write --
    sp_params[k].mu_i = mu_i;
    sp_params[k].mu_s = mu_s;
    sp_params[k].sigma_s = inverse2x2(sigma_mode, detC);
    sp_params[k].logdet_sigma_s = log(detC);

}



// Compute the posterior mode of the covariance matrix
__device__ double3 compute_cov_mode(longlong3 sigma_s_sum, double2 mu_s,
                                    double3 prior_sigma_s, double2 prior_mu_s,
                                    int count, int lam, int df) {

    // -- prior sigma_s --
    double3 sigma_opt = outer_product_term(prior_mu_s, mu_s, lam, count);

	// -- sample covairance --
    double3 sigma_mode;
	double df_post = (double) count + df;
	if (count > 3){
      sigma_mode.x = sigma_s_sum.x - mu_s.x * mu_s.x * count;
      sigma_mode.y = sigma_s_sum.y - mu_s.x * mu_s.y * count;
      sigma_mode.z = sigma_s_sum.z - mu_s.y * mu_s.y * count;
	}else{
      sigma_mode.x = count;
      sigma_mode.y = 0;
      sigma_mode.z = count;
    }

    // -- compute cov matrix [.x = dx*dx   .y = dx*dy    .z = dy*dy] --
    sigma_mode.x = (prior_sigma_s.x + sigma_mode.x + sigma_opt.x) / (df_post + 3.0);
    sigma_mode.y = (prior_sigma_s.y + sigma_mode.y + sigma_opt.y) / (df_post + 3.0);
    sigma_mode.z = (prior_sigma_s.z + sigma_mode.z + sigma_opt.z) / (df_post + 3.0);
    return sigma_mode;
}

/************************************************************


            Calculate the Prior Cov Likelihood


************************************************************/

__device__ double3 outer_product_term(double2 prior_mu_s, double2 mu_s,
                                      int lam, int count) {
    double pscale = (1.0*lam*count)/(lam+count);
    double3 deltas;
    deltas.x = mu_s.x - prior_mu_s.x;
    deltas.y = mu_s.y - prior_mu_s.y;
    double3 prior_sigma_s;
    prior_sigma_s.x = pscale * deltas.x * deltas.x;
    prior_sigma_s.y = pscale * deltas.x * deltas.y;
    prior_sigma_s.z = pscale * deltas.y * deltas.y;
    return prior_sigma_s;
}


// Function to compute the determinant of a symmetric 2x2 covariance matrix
__device__ double determinant2x2(double3 sigma) {
    // det(Sigma) = sigma11 * sigma22 - sigma12^2
    double det = sigma.x * sigma.z - sigma.y * sigma.y;
    if (det <= 0){
      sigma.x = sigma.x + 0.00001;
      sigma.z = sigma.z + 0.00001;
      det = determinant2x2(sigma);
      if(det <=0) det = 0.0001;//hack
    }
    return det;
}

// Function to compute the inverse of a symmetric 2x2 covariance matrix
__device__ double3 inverse2x2(double3 sigma, double det) {
  // double det = determinant2x2(sigma);
    double3 inv_sigma;
    
    // Inverse of 2x2 symmetric matrix:
    // [sigma11 sigma12]^-1 = 1/det * [ sigma22  -sigma12 ]
    // [sigma12 sigma22]        [ -sigma12  sigma11 ]
    
    inv_sigma.x = sigma.z / det;   // sigma22 / det
    inv_sigma.z = sigma.x / det;   // sigma11 / det
    inv_sigma.y = -sigma.y / det;  // -sigma12 / det

    return inv_sigma;
}

// Function to compute the trace of (inv(Sigma_prior) * Sigma_sample)
__device__ double trace2x2(double3 inv_sigma_prior, double3 sigma) {
    // Trace(inv(Sigma_prior) * Sigma_sample) for symmetric 2x2 matrix:
    // Tr([inv11 inv12] * [sigma11 sigma12],
    //    [inv12 inv22]   [sigma12 sigma22])
    //  = inv11 * sigma11 + inv22 * sigma22 + 2 * inv12 * sigma12
    
    return inv_sigma_prior.x * sigma.x + inv_sigma_prior.z * sigma.z + 2.0 * inv_sigma_prior.y * sigma.y;
}

__device__
float calc_cov_likelihood(double3 sigma_s, double3 prior_sigma_s,
                           double det_sigma, int df){

    // Compute the determinants
    double det_prior = determinant2x2(prior_sigma_s);  // Determinant of prior covariance matrix

    // Inverse of the prior covariance matrix
    double3 inv_prior_sigma_s = inverse2x2(prior_sigma_s,det_prior);

    // Compute trace of (inv(prior_sigma_s) * sigma_s)
    double trace_term = trace2x2(inv_prior_sigma_s, sigma_s);

    // Compute log-likelihood for inverse Wishart distribution
    double log_likelihood = (df / 2.0) * log(det_prior) - ((df + 3 + 1) / 2.0) * log(det_sigma) - 0.5 * trace_term - log(tgamma(df/2.0)) - log(tgamma((df-1)/2.0)) - log(M_PI)/2.0 - df*log(2);

    // Save the computed log likelihood into the helper structure
    return log_likelihood;
}
