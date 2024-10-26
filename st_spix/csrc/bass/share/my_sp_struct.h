#include <stdio.h>
#include "cuda.h"
#include "cuda_runtime.h"
#include "device_launch_parameters.h"
/* #include <torch/types.h> */
#include "../../prop/pch.h"


struct PySuperpixelParams{
  // -- appearance --
  torch::Tensor mu_app;
  torch::Tensor sigma_app;
  torch::Tensor logdet_sigma_app;
  torch::Tensor prior_mu_app;
  torch::Tensor prior_sigma_app;
  torch::Tensor prior_mu_app_count;
  torch::Tensor prior_sigma_app_count;
  // -- shape --
  torch::Tensor mu_shape;
  torch::Tensor sigma_shape;
  torch::Tensor logdet_sigma_shape;
  torch::Tensor prior_mu_shape;
  torch::Tensor prior_sigma_shape;
  torch::Tensor prior_mu_shape_count;
  torch::Tensor prior_sigma_shape_count;
  // -- helpers --
  torch::Tensor counts;
  torch::Tensor prior_counts;
  torch::Tensor ids;
};

struct alignas(16) superpixel_params{
    float3 mu_i;
    double3 sigma_s;
    double2 mu_s;
    double logdet_Sigma_s;
    int count;
    double log_count;
    int valid;
    float prior_count;
    int prior_spix;
};


struct alignas(16) superpixel_GPU_helper{
    /* float3 mu_i_sum;  // with respect to nSps */
    double3 mu_i_sum;  // with respect to nSps
    int2 mu_s_sum;
    longlong3 sigma_s_sum;
};

struct alignas(16) superpixel_GPU_helper_sm {
    float3 squares_i;
    int count_f;
    float3 b_n;
    float3 b_n_f;
    float3 numerator;
    float3 denominator;
    float3 numerator_f;
    float3 denominator_f;
    float hasting;
    bool merge; // a bool
    bool remove;
    bool stop_bfs;
    float3 mu_i_sum;
    int count;
    int max_sp;
};

/*********************************

  -=-=-=-    Float3   -=-=-=-=-

**********************************/


struct alignas(16) spix_params{
    // -- appearance --
    float3 mu_app;
    /* float3 sigma_app; */
    float3 prior_mu_app;
    /* float3 prior_sigma_app; */
    int prior_mu_app_count; // kappa (kappa_app)
    /* int prior_sigma_app_count; // nu */
    // -- shape --
    double2 mu_shape;
    double3 sigma_shape;
    double2 prior_mu_shape;
    double3 prior_sigma_shape;
    int prior_mu_shape_count; // lambda term; (prior mu count)
    int prior_sigma_shape_count; // nu term; (prior shape count)
    // -- helpers --
    /* double logdet_sigma_app; */
    double logdet_sigma_shape;
    double logdet_prior_sigma_shape;
    // -- priors --
    double prior_lprob;
    /* double prior_mu_i_lprob; */
    /* double prior_sigma_i_lprob; */
    /* double prior_mu_s_lprob; */
    /* double prior_sigma_s_lprob; */
    // -- helpers --
    int count;
    float prior_count; // df and lam for shape and appearance
    int valid;
};

struct alignas(16) spix_helper{
    float3 sum_app;
    double3 sq_sum_app;
    int2 sum_shape;
    longlong3 sq_sum_shape;
};

struct alignas(16) spix_helper_sm {
    double3 sum_app;
    double3 sq_sum_app;
    int2 sum_shape;
    longlong3 sq_sum_shape;
    /* float3 squares_i; */
    int count_f;
    /* float3 b_n; */
    /* float3 b_n_f; */
    float3 b_n_app;
    float3 b_n_f_app;
    /* float3 b_n_shape; */
    /* float3 b_n_shape_f; */
    float b_n_shape_det;
    float b_n_f_shape_det;
    /* float3 numerator; */
    float numerator_app;
    float numerator_f_app;
    float3 denominator;
    float3 denominator_f;
    float lprob_shape;
    float lprob_f_shape;
    float hasting;
    bool merge; // a bool
    bool remove;
    bool stop_bfs;
    int count;
    int max_sp;
};

struct alignas(16) spix_helper_sm_v2 {
    double3 sum_app;
    double3 sq_sum_app;
    int2 sum_shape;
    longlong3 sq_sum_shape;
    float3 b_n_app;
    float3 b_n_f_app;
    /* float3 b_n_shape; */
    /* float3 b_n_shape_f; */
    float b_n_shape_det;
    float b_n_f_shape_det;
    /* float3 numerator; */
    float numerator_app;
    float numerator_f_app;
    float3 denominator;
    float3 denominator_f;
    float lprob_shape;
    float lprob_f_shape;
    float hasting;
    bool merge; // a bool
    bool remove;
    bool stop_bfs;
    int count;
    int max_sp;
}


/*********************************

  -=-=-=-    "Float6"   -=-=-=-=-

**********************************/


/* struct alignas(16) spix_params_f6{ */
/*     // -- appearance -- */
/*     float mu_app[6]; */
/*     float sigma_app[6]; */
/*     float prior_mu_app[6]; */
/*     float prior_sigma_app[6]; */
/*     int prior_mu_app_count; */
/*     int prior_sigma_app_count; */
/*     // -- shape -- */
/*     double2 mu_shape; */
/*     double3 sigma_shape; */
/*     double2 prior_mu_shape; */
/*     double3 prior_sigma_shape; */
/*     int prior_mu_shape_count; */
/*     int prior_sigma_shape_count; */
/*     // -- helpers -- */
/*     double logdet_sigma_app; */
/*     double logdet_sigma_shape; */
/*     double logdet_prior_sigma_shape; */
/*     // -- priors -- */
/*     double prior_lprob; */
/*     /\* double prior_mu_i_lprob; *\/ */
/*     /\* double prior_sigma_i_lprob; *\/ */
/*     /\* double prior_mu_s_lprob; *\/ */
/*     /\* double prior_sigma_s_lprob; *\/ */
/*     // -- helpers -- */
/*     int count; */
/*     float prior_count; // df and lam for shape and appearance */
/*     int valid; */
/* }; */


/* struct alignas(16) spix_helper_f6{ */
/*     double[6] sum_app; */
/*     double[6] sq_sum_app; */
/*     int2 sum_shape; */
/*     longlong3 sq_sum_shape; */
/* }; */


/* struct alignas(16) spix_helper_sm_f6{ */
/*     float[6] sum_app; */
/*     double[6] sq_sum_app; */
/*     double2 sum_shape; */
/*     /\* float3 squares_i; *\/ */
/*     int count_f; */
/*     float[6] b_n; */
/*     float[6] b_n_f; */
/*     float[6] numerator; */
/*     float[6] denominator; */
/*     float[6] numerator_f; */
/*     float[6] denominator_f; */
/*     float hasting; */
/*     bool merge; // a bool */
/*     bool remove; */
/*     bool stop_bfs; */
/*     int count; */
/*     int max_sp; */
/* }; */


/***
-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-
-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-
-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-
***/

/* struct alignas(16) superpixel_GPU_helper_ptrs{ */
/*     float* mu_i_sum; */
/*     int* mu_s_sum; */
/*     longlong* sigma_s_sum; */
/* }; */



/* struct alignas(16) superpixel_GPU_helper_sm_ptrs { */
/*   float* squares_i; */
/*   float* b_n; */
/*   float* b_n_f; */
/*   float* numerator; */
/*   float* denominator; */
/*   float* numerator_f; */
/*   float* denominator_f; */
/*   float* mu_i_sum; */
/* } */

/* struct alignas(16) superpixel_GPU_helper_sm { */
/*     float3 squares_i; */
/*     int count_f; */
/*     float3 b_n; */
/*     float3 b_n_f; */
/*     float3 numerator; */
/*     float3 denominator; */
/*     float3 numerator_f; */
/*     float3 denominator_f; */
/*     float hasting; */
/*     bool merge; // a bool */
/*     bool remove; */
/*     bool stop_bfs; */
/*     float3 mu_i_sum; */
/*     int count; */
/*     int max_sp; */
/* }; */

struct alignas(16) sm_GPU_helper {
    float3 squares_i;
    int count_f;
    float3 b_n;
    float3 b_n_f;
    float3 numerator;
    float3 denominator;
    float3 numerator_f;
    float3 denominator_f;
    float hasting;
    bool merge; // a bool
    bool remove;
    bool stop_bfs;
    float3 mu_i_sum;
    int count;
    int max_sp;
};

struct alignas(16) post_changes_helper{
    int changes[4];
    float post[5];
    bool skip_post[5];
    bool skip_post_calc[4];
};

struct alignas(16)  superpixel_options{
    int nPixels_in_square_side,area;
    float i_std, s_std, prior_count;
    bool permute_seg, calc_cov, use_hex;
    int prior_sigma_s_sum;
    int nEMIters, nInnerIters;
    float beta_potts_term;
    float alpha_hasting;
    int split_merge_start;
};

#define gpuErrchk(ans) { gpuAssert((ans), __FILE__, __LINE__); }
inline void gpuAssert(cudaError_t code, const char *file, int line, bool abort=true)
{
   if (code != cudaSuccess)
   {
      fprintf(stderr,"GPUassert: %s %s %d\n", cudaGetErrorString(code), file, line);
      if (abort) exit(code);
   }
}

/* #define CHECK_CUDA_ERROR(val) check((val), #val, __FILE__, __LINE__) */
/* void check(cudaError_t err, const char* const func, const char* const file, */
/*            const int line) */
/* { */
/*     if (err != cudaSuccess) */
/*     { */
/*         std::cerr << "CUDA Runtime Error at: " << file << ":" << line */
/*                   << std::endl; */
/*         std::cerr << cudaGetErrorString(err) << " " << func << std::endl; */
/*         // We don't exit when we encounter CUDA errors in this example. */
/*         // std::exit(EXIT_FAILURE); */
/*     } */
/* } */

/* #define CHECK_LAST_CUDA_ERROR() checkLast(__FILE__, __LINE__) */
/* void checkLast(const char* const file, const int line) */
/* { */
/*     cudaError_t const err{cudaGetLastError()}; */
/*     if (err != cudaSuccess) */
/*     { */
/*         std::cerr << "CUDA Runtime Error at: " << file << ":" << line */
/*                   << std::endl; */
/*         std::cerr << cudaGetErrorString(err) << std::endl; */
/*         // We don't exit when we encounter CUDA errors in this example. */
/*         // std::exit(EXIT_FAILURE); */
/*     } */
/* } */
