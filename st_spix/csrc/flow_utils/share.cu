#include <torch/types.h>
#include <cuda.h>
#include <cuda_runtime.h>
#include <vector>
#include <cstddef>
#include <cuda/std/type_traits>
#include <cstdio>
#include <math.h>

template<typename dtype=int>
__device__ __forceinline__ dtype bounds(dtype val, int lim ){
  dtype vval = val;
  if (val < 0){
    vval = -val; // want ("-1" -> "1") _not_ ("-1" -> "0")
    // vval = 10; // want ("-1" -> "1") _not_ ("-1" -> "0")
  }else if (val > (lim-1)){
    vval = 2*(lim-1)-val; // want ("H" -> "H-2") _not_ ("H" -> "H-1")
    // vval = 10;
  }
  return vval;
}

template<typename itype=int>
__device__ __forceinline__
bool check_bound_v2(itype val, int upper){
  return (val >= 0) && (val < (upper-1));
}

template<typename itype=int>
__device__ __forceinline__
bool check_bound(itype val, int upper){
  return (val >= 0) && (val <= (upper-1));
}

__device__ __forceinline__ 
void bilin2d_interpolate_v1(float* src_pix, float* dest, float* cnts,
                            float hi, float wi, int H, int W, int F){


  // -- interpolation coordinates --
  int h0 = __float2int_rz(hi);
  int w0 = __float2int_rz(wi);
  // int h1 = __float2int_rz(hi+1);
  // int w1 = __float2int_rz(wi+1);
  int h1 = h0+1;
  int w1 = w0+1;

  // -- init interpolation weights --  
  float weight_h0 = 1 - (hi - h0);
  float weight_h1 = hi - h0;//1 - (h1 - hi);
  float weight_w0 = 1 - (wi - w0);
  float weight_w1 = wi - w0;//1 - (w1 - wi);

  // -- interpolation weights --
  float w00 = weight_h0 * weight_w0;
  float w01 = weight_h0 * weight_w1;
  float w10 = weight_h1 * weight_w0;
  float w11 = weight_h1 * weight_w1;
  
  // -- cases --
  bool valid_hi = ((hi>=0) and (hi<(H-1)));
  bool valid_wi = ((wi>=0) and (wi<(W-1)));
  if ( valid_hi and valid_wi ){

      // -- 00 --
      float* dest_pix = dest + F*(h0*W + w0);
      float* cnts_pix = cnts + (h0*W + w0);
      atomicAdd(cnts_pix,w00);
      for (int fidx = 0; fidx < F; fidx++){
        atomicAdd(&(dest_pix[fidx]),w00*src_pix[fidx]);
      }

      // -- 01 --
      dest_pix = dest + F*(h0*W + w1);
      cnts_pix = cnts + (h0*W + w1);
      atomicAdd(cnts_pix,w01);
      for (int fidx = 0; fidx < F; fidx++){
        atomicAdd(&(dest_pix[fidx]),w01*src_pix[fidx]);
      }

      // -- 10 --
      dest_pix = dest + F*(h1*W + w0);
      cnts_pix = cnts + (h1*W + w0);
      atomicAdd(cnts_pix,w10);
      for (int fidx = 0; fidx < F; fidx++){
        atomicAdd(&(dest_pix[fidx]),w10*src_pix[fidx]);
      }

      // -- 11 --
      dest_pix = dest + F*(h1*W + w1);
      cnts_pix = cnts + (h1*W + w1);
      atomicAdd(cnts_pix,w11);
      for (int fidx = 0; fidx < F; fidx++){
        atomicAdd(&(dest_pix[fidx]),w11*src_pix[fidx]);
      }

    
  }


}

__device__ __forceinline__ 
void bilin2d_interpolate(float* src_pix, float* dest, float* cnts,
                         float hi, float wi, int H, int W, int F, float eps){

  // -- interpolated locations --
  int h_interp,w_interp;
  float interp_h;
  float interp_w;
  float interp_weight;

  // -- interpolate pixel value --
// #pragma unroll
  for (int ix=0;ix<2;ix++){
// #pragma unroll
    for (int jx=0;jx<2;jx++){

      // -- interpolation weight --
      h_interp = __float2int_rz(hi+ix);
      interp_h = max(0.,1-fabs(h_interp-hi));
      interp_h = (interp_h < eps) ? 0 : interp_h;
      interp_h = (interp_h > (1-eps)) ? 1 : interp_h;
      
      w_interp = __float2int_rz(wi+jx);
      interp_w = max(0.,1-fabs(w_interp-wi));
      interp_w = (interp_w < eps) ? 0 : interp_w;
      interp_w = (interp_w > (1-eps)) ? 1 : interp_w;

      interp_weight = interp_h * interp_w;

      // -- round down when very small --
      // if (interp_weight < eps){
      //   interp_weight = 0;
      // }else if (interp_weight > (1-eps)){
      //   interp_weight = 1.;
      // }

      // -- ensure legal bounds --
      if (not check_bound(h_interp,H)){ continue;}
      if (not check_bound(w_interp,W)){ continue;}
      // h_interp = bounds(h_interp,H);
      // w_interp = bounds(w_interp,W);

      // -- get destination coordinate --
      float* dest_pix = dest + F*(h_interp*W + w_interp);
      float* cnts_pix = cnts + (h_interp*W + w_interp);

      // -- video grad --
      atomicAdd(cnts_pix,interp_weight);
      for (int fidx = 0; fidx < F; fidx++){
        atomicAdd(&(dest_pix[fidx]),interp_weight*src_pix[fidx]);
      }
    }
  }

}

