#include <stdio.h>
#include <math.h>
#include <cuda.h>
#include <cuda_runtime.h>
#include <cmath>
#include <cuda/std/type_traits>

__device__  __host__
int midpoint(int a, int b)
{
    return a + (b-a)/2;
}

__device__ __host__
int eval(int A[], int i, int val, int imin, int imax)
{

    int low = (A[i] <= val);
    int high = (A[i+1] > val);

    if (low && high) {
        return 0;
    } else if (low) {
        return -1;
    } else {
        return 1;
    }
}

__device__ __host__
int binary_search(int A[], int val, int imin, int imax){
    while (imax >= imin) {
        int imid = midpoint(imin, imax);
        int e = eval(A, imid, val, imin, imax);
        if(e == 0) {
            return imid;
        } else if (e < 0) {
            imin = imid;
        } else {         
            imax = imid;
        }
    }

    return -1;
}



template<bool mode> __global__
void relabel_spix(int* spix, int* ids, int npix, int num_ids){

  // -- filling superpixel params into image --
  // extern __shared__ int buff[];
  int ix = threadIdx.x + blockIdx.x * blockDim.x;  
  if (ix>=npix) return; 

  // -- offset super pixels --
  int spix_ix = *(spix + ix);
  int new_id = -1;

  // -- offset of kx -- [binary search not needed]
  for (int kx=0; kx<num_ids; kx++){
    if (ids[kx] == spix_ix){
      new_id = kx;
      break;
    }
  }
  (spix + ix)[0] = new_id;
}



// -- templating --
template void __global__ relabel_spix<false>(int*, int*, int, int);
template void __global__ relabel_spix<true>(int*, int*, int, int);
