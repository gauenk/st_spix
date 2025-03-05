#ifndef SHARED_STRUCTS
#define SHARED_STRUCTS

// -- cuda --
#include "cuda.h"
#include "cuda_runtime.h"
#include "device_launch_parameters.h"

// -- thrust --
#include <thrust/device_vector.h>


struct SuperpixelParams{
  // -- appearance --
  thrust::device_vector<float> mu_app;
  thrust::device_vector<float>  prior_mu_app;
  // -- shape --
  thrust::device_vector<double> mu_shape;
  thrust::device_vector<double> sigma_shape;
  thrust::device_vector<float> logdet_sigma_shape;
  thrust::device_vector<double> prior_mu_shape;
  thrust::device_vector<double> prior_sigma_shape;
  thrust::device_vector<double> sample_sigma_shape;
  // -- helpers --
  thrust::device_vector<int> counts;
  thrust::device_vector<int> sm_counts;
  thrust::device_vector<float> prior_counts;
  thrust::device_vector<int> ids;
  // -- counts for split priors --
  thrust::device_vector<float> invalid_counts;

  // Constructor to initialize all vectors with the given size
  SuperpixelParams(size_t size) {
    mu_app.resize(3*size);
    prior_mu_app.resize(3*size);
    mu_shape.resize(2*size);
    sigma_shape.resize(3*size);
    logdet_sigma_shape.resize(size);
    prior_mu_shape.resize(2*size);
    prior_sigma_shape.resize(3*size);
    sample_sigma_shape.resize(3*size);
    counts.resize(size);
    invalid_counts.resize(size);
    sm_counts.resize(size);
    prior_counts.resize(size);
    ids.resize(size);
  }
};




struct alignas(16) superpixel_options {
    int nPixels_in_square_side,area;
    float i_std, s_std, prior_count;
    bool permute_seg, calc_cov, use_hex;
    int prior_sigma_s_sum;
    int nEMIters, nInnerIters;
    float beta_potts_term;
    float alpha_hasting;
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
    double3 sample_sigma_shape;
    int prior_mu_shape_count; // lambda term; (prior mu count)
    int prior_sigma_shape_count; // nu term; (prior shape count)
    double3 prior_icov;
    // double3 prior_icov_eig;
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
    float icount; // invalid count; to set the prior for split
    int sm_count;
    float prior_count; // df and lam for shape and appearance
    int valid;
    bool prop;
};

struct alignas(16) spix_helper{
    float3 sum_app;
    double3 sq_sum_app;
    longlong2 sum_shape;
    longlong3 sq_sum_shape;
};

struct alignas(16) spix_helper_sm {
    double3 sum_app;
    double3 sq_sum_app;
    longlong2 sum_shape;
    // int2 sum_shape;
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
    // -- ... --
    float lprob_shape;
    // -- ... --
    float lprob_f;
    float lprob_s_cond;
    float lprob_s_ucond;
    float lprob_k_cond;
    float lprob_k_ucond;
    // -- [app; for dev] --
    float lprob_f_app;
    float lprob_s_cond_app;
    float lprob_k_cond_app;
    // -- ... --
    float hasting;
    bool select;
    bool merge; // a bool
    bool remove;
    bool stop_bfs;
    int count;
    int max_sp;
};


struct alignas(16) spix_helper_sm_v2 {
    // -- summary stats --
    double3 sum_app;
    double3 sq_sum_app;
    longlong2 sum_shape;
    longlong3 sq_sum_shape;
    // -- computed stats --
    double3 sigma_s;
    double3 sigma_k;
    double3 sigma_f;
    // -- whatever --
    float3 b_n_app;
    float3 b_n_f_app;
    float b_n_shape_det;
    float b_n_f_shape_det;
    float numerator_app;
    float numerator_f_app;
    float3 denominator;
    float3 denominator_f;
    // -- app --
    float lprob_k_cond_app;
    float lprob_k_ucond_app;
    float lprob_s_cond_app;
    float lprob_s_ucond_app;
    float lprob_f_app;
    // -- shape --
    float lprob_k_cond_shape;
    float lprob_k_ucond_shape;
    float lprob_s_cond_shape;
    float lprob_s_ucond_shape;
    float lprob_f_shape;
    // -- helpers --
    float hasting;
    bool merge; // a bool
    bool select;
    bool remove;
    int count;
    int ninvalid;
    int max_sp;
};


// -- basic --
template <typename T>
bool parse_argument(int &i, int argc, char **argv, const std::string &arg,
                    const std::string &option, T &value) {
    if (arg == option) {
        if (i + 1 < argc) {
            ++i;
            if constexpr (std::is_same<T, int>::value) {
                value = std::stoi(argv[i]);
            } else if constexpr (std::is_same<T, float>::value) {
                value = std::stof(argv[i]);
            } else if constexpr (std::is_same<T, bool>::value) {
                value = std::stoi(argv[i]) == 1;
            } else if constexpr (std::is_same<T, const char *>::value || std::is_same<T, std::string>::value) {
                value = argv[i];
            }
        } else {
            std::cerr << option << " option requires an argument." << std::endl;
            return false;
        }
    }
    return true;
}


#define gpuErrchk(ans) { gpuAssert((ans), __FILE__, __LINE__); }
inline void gpuAssert(cudaError_t code, const char *file, int line, bool abort=true)
{
   if (code != cudaSuccess)
   {
      fprintf(stderr,"GPUassert: %s %s %d\n", cudaGetErrorString(code), file, line);
      if (abort) exit(code);
   }
}



#endif

