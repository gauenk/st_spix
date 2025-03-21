
// #include "cuda.h"
// #include "cuda_runtime.h"
// #define THREADS_PER_BLOCK 1024
#include "update_params.h"
#include <math.h>


/**********************************************************

             -=-=-=-=- Main Function -=-=-=-=-=-

***********************************************************/

/***********************************************

           Compute Posterior Mode

************************************************/
__host__ void update_params(const float* img, const int* spix,
                            spix_params* sp_params,spix_helper* sp_helper,
                            float sigma2_app, const int npixels,
                            const int sp_size, const int nspix_buffer,
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
    // calc_posterior_mode<<<BlockPerGrid2,ThreadPerBlock>>>(sp_params,sp_helper,
    //                                                       sigma2_app,sp_size,
    //                                                       nspix_buffer); 
    calc_simple_update<<<BlockPerGrid2,ThreadPerBlock>>>(sp_params,sp_helper,
                                                         sigma2_app, sp_size,
                                                         nspix_buffer);
}

/*****************************************************************

         Compute Only the Summary Statistics
     [for init; maybe a misplaced function in the code base]

******************************************************************/


__host__ void update_params_summ(const float* img, const int* spix,
                                 spix_params* sp_params,spix_helper* sp_helper,
                                 float sigma_app, const int npixels,
                                 const int nspix_buffer, const int nbatch,
                                 const int width, const int nftrs){

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
    calc_summ_stats<<<BlockPerGrid2,ThreadPerBlock>>>(sp_params,sp_helper,
                                                      sigma_app,nspix_buffer); 
}

__global__
void clear_fields(spix_params* sp_params,
                  spix_helper* sp_helper,
                  const int nsuperpixel_buffer,
                  const int nftrs){

	int k = threadIdx.x + blockIdx.x * blockDim.x;  // the label
	if (k>=nsuperpixel_buffer) return;
	// if (sp_params[k].valid == 0) return;

	sp_params[k].count = 0;
	float3 mu_app;
	mu_app.x = 0;
	mu_app.y = 0;
	mu_app.z = 0;
	sp_params[k].mu_app = mu_app;

    // float3 sigma_app;
    // sigma_app.x = 0;
    // sigma_app.y = 0;
    // sigma_app.z = 0;
	// sp_params[k].sigma_app = sigma_app;

	double2 mu_shape;
	mu_shape.x = 0;
	mu_shape.y = 0;
	sp_params[k].mu_shape = mu_shape;

	double3 sigma_shape;
	sigma_shape.x = 0;
	sigma_shape.y = 0;
	sigma_shape.z = 0;
	sp_params[k].sigma_shape = sigma_shape;

}


__global__
void sum_by_label(const float* img, const int* spix,
                  spix_params* sp_params, spix_helper* sp_helper,
                  const int npixels, const int nbatch,
                  const int width, const int nftrs) {

    // todo -- add nbatch and nftrs
    // getting the index of the pixel
    int t = threadIdx.x + blockIdx.x * blockDim.x;
	if (t>=npixels) return;

	//get the label
	int k = spix[t];
    if (k < 0){ return; } // invalid label

    float3 pix = *(float3*)(img+3*t);
	atomicAdd(&sp_params[k].count, 1);
	atomicAdd(&sp_helper[k].sum_app.x, pix.x);
	atomicAdd(&sp_helper[k].sum_app.y, pix.y);
	atomicAdd(&sp_helper[k].sum_app.z, pix.z);
	atomicAdd(&sp_helper[k].sq_sum_app.x, pix.x*pix.x);
	atomicAdd(&sp_helper[k].sq_sum_app.y, pix.y*pix.y);
	atomicAdd(&sp_helper[k].sq_sum_app.z, pix.z*pix.z);

	int x = t % width;
	int y = t / width; 
	atomicAdd(&sp_helper[k].sum_shape.x, x);
	atomicAdd(&sp_helper[k].sum_shape.y, y);
    atomicAdd((unsigned long long *)&sp_helper[k].sq_sum_shape.x, x*x);
	atomicAdd((unsigned long long *)&sp_helper[k].sq_sum_shape.y, x*y);
	atomicAdd((unsigned long long *)&sp_helper[k].sq_sum_shape.z, y*y);
	
}


/***********************************************************

          Summary Statistics for
          -> Normal-Inverse-Gamma for Appearance (_app)
          -> Normal-Inverse-Wishart for Shape (_shape)

************************************************************/

__global__
void calc_summ_stats(spix_params*  sp_params,spix_helper* sp_helper,
                     float sigma_app, const int nsuperpixel_buffer) {

    // -- update thread --
	int k = threadIdx.x + blockIdx.x * blockDim.x; // the label
	if (k>=nsuperpixel_buffer) return;
	if (sp_params[k].valid == 0) return;
    
    // -- read curr --
	int count_int = sp_params[k].count;
	float a_prior = sp_params[k].prior_count;
	float prior_sigma_s_2 = a_prior * a_prior;
	double count = count_int * 1.0;
    double2 mu_shape;
    float3 mu_app;
    double3 sigma_shape;

    // --  sample means --
	// if (count_int<=0){ return; }
	if (count_int<=0){
      sp_params[k].valid = 0; // invalidate empty spix?
      return;
    }

    // -- appearance --
    mu_app.x = sp_helper[k].sum_app.x / count;
    mu_app.y = sp_helper[k].sum_app.y / count;
    mu_app.z = sp_helper[k].sum_app.z / count;

    // -- view --
    sp_params[k].mu_app.x = mu_app.x;
    sp_params[k].mu_app.y = mu_app.y;
    sp_params[k].mu_app.z = mu_app.z;
    // sp_params[k].sigma_app.x = sp_helper[k].sq_sum_app.x/count - mu_app.x*mu_app.x;
    // sp_params[k].sigma_app.y = sp_helper[k].sq_sum_app.y/count - mu_app.y*mu_app.y;
    // sp_params[k].sigma_app.z = sp_helper[k].sq_sum_app.z/count - mu_app.z*mu_app.z;

    // -- shape --
    mu_shape.x = sp_helper[k].sum_shape.x / count;
    mu_shape.y = sp_helper[k].sum_shape.y / count;
    sp_params[k].mu_shape = mu_shape;
    // sp_params[k].mu_shape.x = mu_shape.x;
    // sp_params[k].mu_shape.y = mu_shape.y;

    // -- sample covariance [NOT inverse] for shape --
    sigma_shape.x = sp_helper[k].sq_sum_shape.x/count - mu_shape.x*mu_shape.x;
    sigma_shape.y = sp_helper[k].sq_sum_shape.y/count - mu_shape.x*mu_shape.y;
    sigma_shape.z = sp_helper[k].sq_sum_shape.z/count - mu_shape.y*mu_shape.y;

    // -- correct sample cov if not invertable --
    double det = sigma_shape.x*sigma_shape.z - sigma_shape.y*sigma_shape.y;
    if (det <= 0){
      sigma_shape.x = sigma_shape.x + 0.00001;
      sigma_shape.z = sigma_shape.z + 0.00001;
      det = sigma_shape.x * sigma_shape.z - sigma_shape.y * sigma_shape.y;
      if (det<=0){ det = 0.00001; } // safety hack
    }

    sp_params[k].sigma_shape.x = sigma_shape.x;
    sp_params[k].sigma_shape.y = sigma_shape.y;
    sp_params[k].sigma_shape.z = sigma_shape.z;
    sp_params[k].logdet_sigma_shape = log(det);

}


__global__
void calc_simple_update(spix_params*  sp_params,spix_helper* sp_helper,
                        float sigma_app, const int sp_size,
                        const int nsuperpixel_buffer) {

    // -- update thread --
	int k = threadIdx.x + blockIdx.x * blockDim.x; // the label
	if (k>=nsuperpixel_buffer) return;
	if (sp_params[k].valid == 0) return;
    
    // -- read curr --
	int count_int = sp_params[k].count;
	float pc = sp_params[k].prior_count; 
	float prior_sigma_s_2 = pc * pc;
	// float prior_sigma_s_2 = pc * sp_size;
	double count = count_int * 1.0;
    double2 mu_shape;
    float3 mu_app;
    double3 sigma_shape;
	// double total_count = (double) count_int + pc;
	double total_count = (double) count_int + pc*50;

    // --  sample means --
	// if (count_int<=0){ return; }
	if (count_int<=0){
      sp_params[k].valid = 0; // invalidate empty spix?
      return;
    }

    // -- get prior --
    double3 prior_sigma_shape = sp_params[k].prior_sigma_shape;
    double pr_det = prior_sigma_shape.x*prior_sigma_shape.z - \
      prior_sigma_shape.y*prior_sigma_shape.y;
    // printf("[a:%d] sxx,syx,syy: %lf %lf %lf %lf\n",
    //        k,prior_sigma_shape.x,prior_sigma_shape.y,prior_sigma_shape.z,pr_det);
    double tmp = prior_sigma_shape.x;
    prior_sigma_shape.x = prior_sigma_shape.z/pr_det;
    prior_sigma_shape.y = prior_sigma_shape.y/pr_det;
    prior_sigma_shape.z = tmp/pr_det;
    // printf("[b:%d] sxx,syx,syy: %lf %lf %lf %f\n",
    //        k,prior_sigma_shape.x,prior_sigma_shape.y,prior_sigma_shape.z,pc);
    // prior_sigma_shape.x = pc * sp_size;
    // prior_sigma_shape.y = 0;
    // prior_sigma_shape.z = pc * sp_size;
    prior_sigma_shape.x = pc;// * pc;
    prior_sigma_shape.y = 0;
    prior_sigma_shape.z = pc;// * pc;


    // -- appearance --
    mu_app.x = sp_helper[k].sum_app.x / count;
    mu_app.y = sp_helper[k].sum_app.y / count;
    mu_app.z = sp_helper[k].sum_app.z / count;
    // printf("%2.3f, %2.3f, %2.3f\n",mu_app.x,mu_app.y,mu_app.z);
    sp_params[k].mu_app.x = mu_app.x;
    sp_params[k].mu_app.y = mu_app.y;
    sp_params[k].mu_app.z = mu_app.z;
    // sp_params[k].sigma_app.x = sp_helper[k].sq_sum_app.x/count - mu_app.x*mu_app.x;
    // sp_params[k].sigma_app.y = sp_helper[k].sq_sum_app.y/count - mu_app.y*mu_app.y;
    // sp_params[k].sigma_app.z = sp_helper[k].sq_sum_app.z/count - mu_app.z*mu_app.z;

    // -- shape --
    mu_shape.x = sp_helper[k].sum_shape.x / count;
    mu_shape.y = sp_helper[k].sum_shape.y / count;
    sp_params[k].mu_shape.x = mu_shape.x;
    sp_params[k].mu_shape.y = mu_shape.y;

    // -- sample covariance --
    sigma_shape.x = sp_helper[k].sq_sum_shape.x - count*mu_shape.x*mu_shape.x;
    sigma_shape.y = sp_helper[k].sq_sum_shape.y - count*mu_shape.x*mu_shape.y;
    sigma_shape.z = sp_helper[k].sq_sum_shape.z - count*mu_shape.y*mu_shape.y;

    // -- inverse --
    sigma_shape.x = (prior_sigma_s_2 + sigma_shape.x) / (total_count - 3.0);
    sigma_shape.y = sigma_shape.y / (total_count - 3);
    sigma_shape.z = (prior_sigma_s_2 + sigma_shape.z) / (total_count - 3.0);
    // sigma_shape.x = (pc*prior_sigma_shape.x + sigma_shape.x) / (total_count + 3.0);
    // sigma_shape.y = (pc*prior_sigma_shape.y + sigma_shape.y) / (total_count + 3.0);
    // sigma_shape.z = (pc*prior_sigma_shape.z + sigma_shape.z) / (total_count + 3.0);

    // -- correct sample cov if not invertable --
    double det = sigma_shape.x*sigma_shape.z - sigma_shape.y*sigma_shape.y;
    if (det <= 0){
      sigma_shape.x = sigma_shape.x + 0.00001;
      sigma_shape.z = sigma_shape.z + 0.00001;
      det = sigma_shape.x * sigma_shape.z - sigma_shape.y * sigma_shape.y;
      if (det<=0){ det = 0.00001; } // safety hack
    }
    sp_params[k].sigma_shape.x = sigma_shape.z/det;
    sp_params[k].sigma_shape.y = -sigma_shape.y/det;
    sp_params[k].sigma_shape.z = sigma_shape.x/det;
    sp_params[k].logdet_sigma_shape = log(det);

}

/***********************************************************

          Posterior Modes of
          -> Normal-Inverse-Gamma for Appearance (_app)
          -> Normal-Inverse-Wishart for Shape (_shape)

************************************************************/

__global__
void calc_posterior_mode(spix_params*  sp_params, spix_helper* sp_helper,
                         float sigma2_app, int sp_size, const int nsuperpixel_buffer) {

    // -- update thread --
	int k = threadIdx.x + blockIdx.x * blockDim.x; // the label
	if (k>=nsuperpixel_buffer) return;
	if (sp_params[k].valid == 0) return;
	if (sp_params[k].count == 0) return;
    
    // -- read curr --
	int count_int = sp_params[k].count;
	// float pc = sp_params[k].prior_count;
	float prior_count = sp_params[k].prior_count;
	double count = count_int * 1.0;
    double lprob = 0.0;
    bool prop = sp_params[k].prop;

    // -- unpack --
    // int prior_mu_app_count = sp_params[k].prior_mu_app_count;
    // int prior_sigma_app_count = sp_params[k].prior_sigma_app_count;
    // int prior_mu_shape_count = sp_params[k].prior_mu_shape_count;
    // int prior_sigma_shape_count = sp_params[k].prior_sigma_shape_count;
    // prior_mu_app_count = prior_count; // only if conditoned on something... ow 1

    /**************************************************
   
                     Appearance Information

    ***************************************************/

	// -- calculate means --
    float3 prior_mu_app = sp_params[k].prior_mu_app;
    // prior_mu_app.x = 0;
    // prior_mu_app.y = 0;
    // prior_mu_app.z = 0;
    // prior_mu_app_count = 1;
    // float3 prior_sigma_app = sp_params[k].prior_sigma_app;

    float3 post_mu_app = calc_app_sample_mean(sp_helper[k].sum_app, count);
    // float3 post_mu_app = calc_app_mean_mode(sp_helper[k].sum_app,prior_mu_app,
    //                                         count,prior_mu_app_count);


    // float3 post_mu_app = calc_app_mean_mode(sp_helper[k].sum_app,prior_mu_app,
    //                                         count,prior_mu_app_count);
    // this shouldn't be "sigma2_app" but rather the "tau"? or sigma2_app/count?
    // lprob += calc_app_mean_ll(post_mu_app,prior_mu_app,sigma2_app);
    // float3 post_sigma_app = calc_app_sigma_mode(sp_helper[k].sq_sum_app,
    //                                             sp_helper[k].sum_app,count,
    //                                             prior_sigma_app,prior_mu_app,
    //                                             prior_sigma_app_count,
    //                                             prior_mu_app_count);
    // float det_sigma_app = post_sigma_app.x*post_sigma_app.y*post_sigma_app.z;
    // lprob += calc_app_sigma_ll(post_sigma_app, prior_sigma_app, prior_sigma_app_count);

    /**************************************************
   
                     Spatial Information

    ***************************************************/

    // -- manipulate prior cov --
    double2 prior_mu_shape = sp_params[k].prior_mu_shape;
    // prior_mu_shape.x = 0;
    // prior_mu_shape.y = 0;
    double3 prior_sigma_shape = sp_params[k].prior_sigma_shape;
    // prior_sigma_shape.x = prior_count*prior_count;
    // prior_sigma_shape.y = 0;
    // prior_sigma_shape.z = prior_count*prior_count;
    double det_prior = determinant2x2(prior_sigma_shape);
    double3 prior_isigma_shape = inverse2x2(prior_sigma_shape,det_prior);

    // -- sample mean --
    double2 mu_shape = calc_shape_sample_mean(sp_helper[k].sum_shape,count);
    // -- sigma mode --
    longlong3 _sq_sum = sp_helper[k].sq_sum_shape;
    // printf("[ud] %lld,%lld,%lld|%lf,%lf\n",
    //        _sq_sum.x,_sq_sum.y,_sq_sum.z,mu_shape.x,mu_shape.y);
    double3 sigma_shape = calc_shape_sigma_mode_simp(sp_helper[k].sq_sum_shape,
                                                     mu_shape,prior_sigma_shape,
                                                     prior_mu_shape,count,
                                                     prior_count,sp_size);
    // double3 sigma_shape = calc_shape_sigma_mode(sp_helper[k].sq_sum_shape,mu_shape,
    //                                             prior_sigma_shape,prior_mu_shape,
    //                                             count,prior_count);
    double det_sigma_shape = determinant2x2(sigma_shape);
    double3 isigma_shape = inverse2x2(sigma_shape,det_sigma_shape);

    // -- mu mode [AFTER sigma mode] --
    // mu_shape = calc_shape_mean_mode(mu_shape, prior_mu_shape,
    //                                 count, prior_mu_shape_count);

    // -- compute covariance matrix prior likelihood --
    lprob += calc_shape_mean_ll(mu_shape,prior_mu_shape,prior_isigma_shape,det_prior);
    lprob += calc_shape_sigma_ll(sigma_shape,prior_sigma_shape,det_sigma_shape,count);

    /*****************************************************

                      Write

    *****************************************************/

    // -- appearance --
    // sp_params[k].sigma_app.x = sp_helper[k].sq_sum_app.x/count - mu_app.x*mu_app.x;
    // sp_params[k].sigma_app.y = sp_helper[k].sq_sum_app.y/count - mu_app.y*mu_app.y;
    // sp_params[k].sigma_app.z = sp_helper[k].sq_sum_app.z/count - mu_app.z*mu_app.z;

    // -- appearance --
    sp_params[k].mu_app = post_mu_app;
    // sp_params[k].sigma_app = post_sigma_app;
    // sp_params[k].logdet_sigma_app = log(det_sigma_app);

    // -- shape --
    sp_params[k].mu_shape = mu_shape;
    sp_params[k].sigma_shape = isigma_shape;
    sp_params[k].logdet_sigma_shape = log(det_sigma_shape);

    // -- prior prob --
    sp_params[k].prior_lprob = lprob;
}



/******************************************************************

                         Appearance

******************************************************************/

__device__ float3 calc_app_sample_mean(float3 sample_sum, double count) {
  float3 post_mu;
  post_mu.x = (sample_sum.x)/(count);
  post_mu.y = (sample_sum.y)/(count);
  post_mu.z = (sample_sum.z)/(count);
  return post_mu;
}

__device__ float3 calc_app_mean_mode(float3 sample_sum, float3 prior_mu,
                                     int count, float prior_count) {
  float3 post_mu;
  post_mu.x = (sample_sum.x + prior_count * prior_mu.x)/(count + prior_count);
  post_mu.y = (sample_sum.y + prior_count * prior_mu.y)/(count + prior_count);
  post_mu.z = (sample_sum.z + prior_count * prior_mu.z)/(count + prior_count);
  // post_mu.x = sample_sum.x/count;
  // post_mu.y = sample_sum.y/count;
  // post_mu.z = sample_sum.z/count;
  return post_mu;
}

__device__ double calc_app_mean_ll(float3 mu_app, float3 prior_mu, float sigma2_app){
  float dx = mu_app.x - prior_mu.x;
  float dy = mu_app.y - prior_mu.y;
  float dz = mu_app.z - prior_mu.z;
  float det_prior_sigma = sigma2_app*sigma2_app*sigma2_app;
  double lprob = 0.;
  lprob += -1/2.*(dx*dx/sigma2_app + dy*dy/sigma2_app + dz*dz/sigma2_app);
  lprob += -(3/2.)*log(2*M_PI) - log(det_prior_sigma)/2.;
  return lprob;
}

__device__ float3 calc_app_sigma_mode(double3 sample_sq_sum, double3 sample_sum,
                                      int count, float3 prior_sigma, float3 prior_mu,
                                      int prior_count_sigma, int prior_count_mu) {
	// -- sample covairance --
    double3 sample_var;
	// double df_post = (double) count + prior_count_sigma;
    sample_var.x = sample_sq_sum.x - sample_sum.x*sample_sum.x/count;
    sample_var.y = sample_sq_sum.y - sample_sum.y*sample_sum.y/count;
    sample_var.z = sample_sq_sum.z - sample_sum.z*sample_sum.z/count;

    // -- outer product term --
    float rescale = prior_count_mu * count / (prior_count_mu + count);
    float3 op_term;
    op_term.x = (sample_sum.x/count - prior_mu.x);
    op_term.y = (sample_sum.y/count - prior_mu.y);
    op_term.z = (sample_sum.z/count - prior_mu.z);
    op_term.x = rescale*op_term.x*op_term.x;
    op_term.y = rescale*op_term.y*op_term.y;
    op_term.z = rescale*op_term.z*op_term.z;

    // -- sigma mode --
    float div = 1/(1.0*(prior_count_sigma + count));
    float3 sigma_mode;
    sigma_mode.x = div*(prior_count_sigma * prior_sigma.x + sample_var.x + op_term.x);
    sigma_mode.y = div*(prior_count_sigma * prior_sigma.y + sample_var.y + op_term.y);
    sigma_mode.z = div*(prior_count_sigma * prior_sigma.z + sample_var.z + op_term.z);

    // -- [remove me; debug] --
    sigma_mode.x = prior_sigma.x;
    sigma_mode.y = prior_sigma.y;
    sigma_mode.z = prior_sigma.z;

    return sigma_mode;
}

__device__ double calc_app_sigma_ll(float3 sigma, float3 prior_sigma, float prior_count) {

    // -- inverse-gamma params --
    double alpha = prior_count/2.;
    // double aconst = alpha * log(beta) - lgamma(alpha);
    double aconst = alpha - lgamma(alpha);

    // Log-likelihood for each dimension
    double lprob = 0.0;
    lprob += aconst - (alpha + 1.0) * log(sigma.x) - alpha * prior_sigma.x / sigma.x;
    lprob += aconst - (alpha + 1.0) * log(sigma.y) - alpha * prior_sigma.y / sigma.y;
    lprob += aconst - (alpha + 1.0) * log(sigma.z) - alpha * prior_sigma.z / sigma.z;
    return lprob;
}



/******************************************************************

                         Shape

******************************************************************/

__device__ double2 calc_shape_sample_mean(int2 sum_shape, double count) {
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

__device__ double2 calc_shape_mean_mode(double2& mu, double2 prior_mu,
                                        int count, float prior_count) {
  mu.x = (prior_count * prior_mu.x + count*mu.x)/(prior_count + count);
  mu.y = (prior_count * prior_mu.y + count*mu.y)/(prior_count + count);
  return mu;
}


// -- likelihood of proposed mean --
__device__ double calc_shape_mean_ll(double2 mu, double2 prior_mu,
                                     double3 inv_sigma, double det_prior){
  double dx = mu.x - prior_mu.x;
  double dy = mu.y - prior_mu.y;
  double lprob = -1/2.*(dx*dx*inv_sigma.x+2.*dx*dy*inv_sigma.y+dy*dy*inv_sigma.z);
  lprob += -(3/2.)*log(2*M_PI) - log(det_prior)/2.;
  return lprob;
}

// Compute the posterior mode of the covariance matrix
__device__ double3 calc_shape_sigma_mode_simp(longlong3 sq_sum, double2 mu,
                                              double3 prior_sigma_s, double2 prior_mu,
                                              int count, float prior_count, int sp_size) {
  
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

    float sigma2_prior = prior_count*sp_size;
    double total_count = 1.*(count + prior_count);
    sigma_mode.x = (sigma2_prior + sigma_mode.x)/(total_count + 3.0);
    sigma_mode.y = (sigma_mode.y) / (total_count + 3.0);
    sigma_mode.z = (sigma2_prior + sigma_mode.z)/(total_count + 3.0);

    return sigma_mode;
}


// Compute the posterior mode of the covariance matrix_
__device__ double3 calc_shape_sigma_mode(longlong3 sq_sum, double2 mu,
                                         double3 prior_sigma, double2 prior_mu,
                                         int count, float prior_count) {

    // -- prior sigma_s --
    // double3 sigma_opt = outer_product_term(prior_mu, mu, prior_count, count);

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
    // sigma_mode.x = (prior_count*prior_sigma.x + sigma_mode.x + sigma_opt.x) / (tcount + 3.0);
    // sigma_mode.y = (prior_count*prior_sigma.y + sigma_mode.y + sigma_opt.y) / (tcount + 3.0);
    // sigma_mode.z = (prior_count*prior_sigma.z + sigma_mode.z + sigma_opt.z) / (tcount + 3.0);
    sigma_mode.x = (prior_count*prior_sigma.x + sigma_mode.x) / (tcount + 3.0);
    sigma_mode.y = (prior_count*prior_sigma.y + sigma_mode.y) / (tcount + 3.0);
    sigma_mode.z = (prior_count*prior_sigma.z + sigma_mode.z) / (tcount + 3.0);


    return sigma_mode;
}

__device__ double calc_shape_sigma_ll(double3 sigma_s, double3 prior_sigma,
                                      double det_sigma, int df){

    // Compute the determinants
    double det_prior = determinant2x2(prior_sigma);  // Determinant of prior covariance

    // Inverse of the prior covariance matrix
    double3 inv_prior_sigma = inverse2x2(prior_sigma,det_prior);

    // Compute trace of (inv(prior_sigma) * sigma_s)
    double trace_term = trace2x2(inv_prior_sigma, sigma_s);

    // Compute log-likelihood for inverse Wishart distribution
    double lprob = (df / 2.0) * log(det_prior) - ((df + 3 + 1) / 2.0) * log(det_sigma) - 0.5 * trace_term - lgamma(df/2.0) - lgamma((df-1)/2.0) - log(M_PI)/2.0 - df*log(2);

    // Save the computed log likelihood into the helper structure
    return lprob;
}


/************************************************************


                   Helper Functions


************************************************************/

__device__ double3 outer_product_term(double2 prior_mu, double2 mu,
                                      int obs_count, float prior_count) {
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
__device__ double determinant2x2(double3 sigma) {
    // det(Sigma) = sigma11 * sigma22 - sigma12^2
    double det = sigma.x * sigma.z - sigma.y * sigma.y;
    if (det <= 0){
      sigma.x = sigma.x + 0.00001;
      sigma.z = sigma.z + 0.00001;
      det = sigma.x * sigma.z - sigma.y * sigma.y;
      // det = determinant2x2(sigma);
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
    inv_sigma.y = -sigma.y / det;  // -sigma12 / det
    inv_sigma.z = sigma.x / det;   // sigma11 / det

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

