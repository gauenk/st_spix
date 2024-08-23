
#include "cuda.h"
#include "cuda_runtime.h"
#include "device_launch_parameters.h"


template<bool mode> __global__
void relabel_spix(int* spix, int* ids, int npix, int nspix);


