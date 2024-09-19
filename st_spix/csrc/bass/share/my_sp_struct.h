#include <stdio.h>
#include "cuda.h"
#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#include <torch/types.h>


struct PySuperpixelParams{
  torch::Tensor mu_i;
  torch::Tensor mu_s;
  torch::Tensor sigma_s;
  torch::Tensor logdet_Sigma_s;
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

/* struct alignas(16) superpixel_GPU_helper_ptrs{ */
/*     float* mu_i_sum; */
/*     int* mu_s_sum; */
/*     longlong* sigma_s_sum; */
/* }; */

struct alignas(16) superpixel_GPU_helper{
    /* float3 mu_i_sum;  // with respect to nSps */
    double3 mu_i_sum;  // with respect to nSps
    int2 mu_s_sum;
    longlong3 sigma_s_sum;
};


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
