
/* #include "update_params.h" */
#include <math.h>


// // ---- Appearance [mean,cov] ---
// __device__ float3 calc_app_mean_mode(double3 sample_sum, float3 prior_mu,
//                                      int count, int prior_count);
// __device__ float3 calc_app_sigma_mode(double3 sq_sample_sum, double3 sample_sum,
//                                       int count, float3 prior_sigma, float3 prior_mu,
//                                       int prior_count_sigma, int prior_count_mu);
// __device__ double calc_app_mean_ll(float3 mu_app, float3 prior_mu, float sigma_app);
// // __device__ double calc_app_mean_ll(float3 mu_app, float3 prior_mu, float3 prior_sigma);
// __device__ double calc_app_sigma_ll(float3 sigma, float3 prior_sigma, int prior_count);

// // ---- Shape [mean,cov] ---
// __device__ double2 calc_shape_sample_mean(int2 sum_shape, int count);
// __device__ double2 calc_shape_mean_mode(double2& mu, double2 prior_mu,
//                                         int count, int lam);
// __device__ double3 calc_shape_sigma_mode_simp(longlong3 sq_sum, double2 mu,
//                                               double3 prior_sigma_s, double2 prior_mu,
//                                               int count, int prior_count);
// __device__ double3 calc_shape_sigma_mode(longlong3 sq_sum, double2 mu,
//                                          double3 prior_sigma, double2 prior_mu,
//                                          int count, int prior_count);
// /* __device__ double3 calc_shape_sigma_mode(longlong3 sigma_s_sum, double2 mu_s, */
// /*                                        double3 prior_sigma_s, double2 prior_mu_s, */
// /*                                        int count, int lam, int df); */
// __device__ double calc_shape_mean_ll(double2 mu, double2 prior_mu,
//                                      double3 inv_prior_sigma, double det_prior);
// __device__ double calc_shape_sigma_ll(double3 sigma_s, double3 prior_sigma_s,
//                                       double det_sigma, int df);

// /* __device__ float calc_shape_mean_ll(double2 mu_s,double2 prior_mu_s, */
// /*                                     int lam, int count); */
// /* __device__ float calc_shape_cov_ll(double3 sigma_s, double3 prior_sigma_s, */
// /*                                    double det_sigma, int df); */

// /* // ---- Shape [mean,cov] --- */
// /* __device__ float3 calc_app_mean_mode(double3 mu_sum, float3 prior_mu, */
// /*                                      int count, int lam); */

// /************************************************************

//             Calculate the Prior Cov Likelihood

// ************************************************************/

// __device__ double3 outer_product_term(double2 prior_mu_s, double2 mu_s,
//                                       int lam, int count);
// __device__ double determinant2x2(double3 sigma);
// __device__ double3 inverse2x2(double3 sigma, double det);
// __device__ double trace2x2(double3 inv_sigma_prior, double3 sigma);
// __device__ float calc_cov_likelihood(double3 sigma_s, double3 prior_sigma_s,
//                                      double det_sigma, int df);


/************************************************************


                   Helper Functions


************************************************************/

__device__ double3 outer_product_term(double2 prior_mu, double2 mu,
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



/******************************************************************

                         Appearance

******************************************************************/

__device__ float3 calc_app_mean_mode(double3 sample_sum, float3 prior_mu,
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

__device__ double calc_app_mean_ll(float3 mu_app, float3 prior_mu, float sigma_app){
  float dx = mu_app.x - prior_mu.x;
  float dy = mu_app.y - prior_mu.y;
  float dz = mu_app.z - prior_mu.z;
  float det_prior_sigma = sigma_app*sigma_app*sigma_app;
  float sigma_app2 = sigma_app * sigma_app;
  double lprob = 0.;
  lprob += -1/2.*(dx*dx/sigma_app2 + dy*dy/sigma_app2 + dz*dz/sigma_app2);
  lprob += -(3/2.)*__logf(2*M_PI) - __logf(det_prior_sigma)/2.;
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

__device__ double calc_app_sigma_ll(float3 sigma, float3 prior_sigma, int prior_count) {

    // -- inverse-gamma params --
    double alpha = prior_count/2.;
    // double aconst = alpha * log(beta) - lgamma(alpha);
    double aconst = alpha - lgamma(alpha);

    // Log-likelihood for each dimension
    double lprob = 0.0;
    lprob += aconst - (alpha + 1.0) * __logf(sigma.x) - alpha * prior_sigma.x / sigma.x;
    lprob += aconst - (alpha + 1.0) * __logf(sigma.y) - alpha * prior_sigma.y / sigma.y;
    lprob += aconst - (alpha + 1.0) * __logf(sigma.z) - alpha * prior_sigma.z / sigma.z;
    return lprob;
}



/******************************************************************

                         Shape

******************************************************************/

__device__ double2 calc_shape_sample_mean(int2 sum_shape, int count) {
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
                                        int count, int prior_count) {
  mu.x = (prior_count * prior_mu.x + count*mu.x)/(prior_count + count);
  mu.y = (prior_count * prior_mu.y + count*mu.y)/(prior_count + count);
  return mu;
}


// -- likelihood of proposed mean --
__device__ double calc_shape_mean_ll(double2 mu, double2 prior_mu,
                                     double3 inv_sigma, double det_prior){
  double dx = mu.x - prior_mu.x;
  double dy = mu.y - prior_mu.y;
  float lprob = -1/2.*(dx*dx*inv_sigma.x+dx*dy*inv_sigma.y+2.*dy*dy*inv_sigma.z);
  lprob += -(3/2.)*__logf(2*M_PI) - __logf(det_prior)/2.;
  return lprob;
}

// Compute the posterior mode of the covariance matrix
__device__ double3 calc_shape_sigma_mode_simp(longlong3 sq_sum, double2 mu,
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


// Compute the posterior mode of the covariance matrix_
    // double3 sigma_k = calc_shape_sigma_mode(sq_sum_shape_k,mu_shape_k,
    //                                         prior_sigma_shape,
    //                                         prior_mu_shape,count_k,kappa_shape);

// double3 sigma_shape = calc_shape_sigma_mode(sp_helper[k].sq_sum_shape,mu_shape,
//                                             prior_sigma_shape,prior_mu_shape,
//                                             count,prior_count);
__device__ double3 calc_shape_sigma_mode(longlong3 sq_sum, double2 mu,
                                         double3 prior_sigma, double2 prior_mu,
                                         int count, int prior_count) {

    // -- prior sigma_s --
    double3 sigma_opt = outer_product_term(prior_mu, mu, count, prior_count);

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

__device__ double calc_shape_sigma_ll(double3 sigma_s, double3 prior_sigma,
                                     double det_sigma, int df){

    // Compute the determinants
    double det_prior = determinant2x2(prior_sigma);  // Determinant of prior covariance

    // Inverse of the prior covariance matrix
    double3 inv_prior_sigma = inverse2x2(prior_sigma,det_prior);

    // Compute trace of (inv(prior_sigma) * sigma_s)
    double trace_term = trace2x2(inv_prior_sigma, sigma_s);

    // Compute log-likelihood for inverse Wishart distribution
    double lprob = (df / 2.0) * __logf(det_prior) - ((df + 3 + 1) / 2.0) * __logf(det_sigma) - 0.5 * trace_term - log(tgamma(df/2.0)) - log(tgamma((df-1)/2.0)) - __logf(M_PI)/2.0 - df*__logf(2);

    // Save the computed log likelihood into the helper structure
    return lprob;
}

