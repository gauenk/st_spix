#include "cuda.h"
#include "cuda_runtime.h"
#include "device_launch_parameters.h"

__host__ int CudaInitSeg(int* seg_cpu, int* seg_gpu, int* split_merge_pairs,  int nPts,
                         int sz, int nbatch, int xdim, int ydim, bool use_hex);
__global__ void InitHexCenter(double* centers, double H, double width, int max_nPts,
                              int max_num_sp_x, int xdim, int ydim);
__global__ void InitHexSeg(int* seg, double* centers, int K, int nPts, int xdim);
__global__ void InitSquareSeg(int* seg, int nPts, int sz, int xdim, int ydim);
__global__ void InitSplitMerge(int* split_merge_pairs, int nPts);
