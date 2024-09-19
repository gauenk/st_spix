#include <string>
#include <iostream>
#include <fstream>
#include <sstream>
#include <float.h>
#include <filesystem>
using namespace std;

#define THREADS_PER_BLOCK 512

#include "../share/gpu_utils.h"
#include "init_seg.h"

#include <stdio.h>

__host__ int CudaInitSeg(int* seg_cpu, int* seg_gpu, int* split_merge_pairs, int nPts,
                         int sz, int nbatch, int xdim, int ydim, bool use_hex){	
  // -- todo; add nbatch --

  	dim3 ThreadPerBlock(THREADS_PER_BLOCK,1);
    use_hex = true;
    int num_block_pixel = ceil(double(nPts+1) / double(THREADS_PER_BLOCK));
    dim3 BlockPerGrid_pixel(num_block_pixel,nbatch);
	if (!use_hex){
      InitSquareSeg<<<BlockPerGrid_pixel,ThreadPerBlock>>>(seg_gpu,nPts,sz,
                                                           nbatch, xdim, ydim);
        cudaMemcpy(seg_cpu, seg_gpu, nPts*sizeof(int), cudaMemcpyDeviceToHost);

	}else{
      std::stringstream batch_str, xdim_str, ydim_str, sz_str;
      batch_str << nbatch;
        xdim_str << xdim;
        ydim_str << ydim;
        sz_str << sz;     
        std::string root =  ".bass_cache";
        std::string file_path = root + "/" + batch_str.str() + "_" + xdim_str.str() \
          + "_" + ydim_str.str() + "_" + sz_str.str() + ".bin";

        // length of each side   
        double H = sqrt( double(pow(sz, 2)) / (1.5 *sqrt(3.0)) );
        double w = sqrt(3.0) * H;
        //printf("%1f \n", H);
        //printf("%1f \n", w);

        //calculate how many hexagons are on x and y axis
        int max_num_sp_x = (int) floor(double(xdim)/w) + 1;
        int max_num_sp_y = (int) floor(double(ydim)/(1.5*H)) + 1;

        int max_nSPs = max_num_sp_x * max_num_sp_y * 4; //Roy -Change

        //printf("%d \n", max_num_sp_x);
       // printf("%d \n", max_num_sp_y);
       // printf("%d \n", max_nSPs);

        // -- create directory --
        if (not std::filesystem::is_directory(root)){
          std::filesystem::create_directory(root);
        }

        if (loadArray(seg_cpu, nPts, file_path)){
            cudaMemcpy(seg_gpu,seg_cpu,nPts*sizeof(int),cudaMemcpyHostToDevice);
        }else{
        // if (true){

            int num_block_sp =  ceil(double(max_nSPs) /double(THREADS_PER_BLOCK));
            dim3 BlockPerGrid_sp(num_block_sp,nbatch);
            double* centers;
            cudaMalloc((void**) &centers, 2*max_nSPs*sizeof(double));
            InitHexCenter<<<BlockPerGrid_sp,ThreadPerBlock>>>(centers, \
                              H, w, max_nSPs, max_num_sp_x, nbatch, xdim, ydim); 
            InitHexSeg<<<BlockPerGrid_pixel,ThreadPerBlock>>>(seg_gpu, \
                                            centers, max_nSPs, nPts, nbatch, xdim);
            cudaFree(centers);
            cudaMemcpy(seg_cpu, seg_gpu, nPts*sizeof(int), cudaMemcpyDeviceToHost);
            // saveArray(seg_cpu, nPts, file_path);


            // // -- Remove Small Edges --
            // RemoveSmallEdges<<<BlockPerGrid_pixel,ThreadPerBlock>>>(seg_gpu, \
            //                                 centers, max_nSPs, nPts, nbatch, xdim);


        }
        
	}

    int nSPs = get_max(seg_cpu, nPts)+1;
	return nSPs;
}



// __global__ void RemoveSmallEdges(double* centers, double H, double w, int max_nPts,
//                                  int max_num_sp_x, int nbatch, int xdim, int ydim){

// }


__global__ void InitHexCenter(double* centers, double H, double w, int max_nPts,
                              int max_num_sp_x, int nbatch, int xdim, int ydim){
  // todo; add batch
	int idx = threadIdx.x + blockIdx.x * blockDim.x; 
	if (idx >= max_nPts) return;

    int x = idx % max_num_sp_x; 
    int y = idx / max_num_sp_x; 

    double xx = double(x) * w;
    double yy = double(y) * 1.5 *H; 
    
    if (y%2 == 0){
        xx = xx + 0.5*w;
    }
    
    centers[2*idx]  = xx;
    centers[2*idx+1]  = yy;    
}




__global__ void InitHexSeg(int* seg, double* centers, int K, int nPts,
                           int nbatch, int xdim){
  // todo ;add nbatch [just copy this rather than add nbatch...]
	int idx = threadIdx.x + blockIdx.x * blockDim.x; 	
	if (idx >= nPts) return;

    int x = idx % xdim;
    int y = idx / xdim;   

    double dx,dy,d2;
    double D2 = DBL_MAX; 
    for (int j=0; j < K;  j++){
        dx = (x - centers[j*2+0]);
        dy = (y - centers[j*2+1]);
        d2 = dx*dx + dy*dy;
        if ( d2 <= D2){
              D2 = d2;  
              seg[idx]=j+1;
        }           
    } 
    return;	
}



// for everypixel, assign it to a superptxel
__global__ void  InitSquareSeg(int* seg, int nPts, int sz,
                               int nbatch, int xdim, int ydim){
	int t = threadIdx.x + blockIdx.x * blockDim.x; 
	if (t>=nPts) return;
	
    float side_x = float(xdim)/float(ceil(float(xdim)/float(sz)));
    float side_y = float(ydim)/float(ceil(float(ydim)/float(sz)));
    //side_x += xdim/ side_x;
    //side_y += xdim/ side_x;

	//how many superpixels per col
	//int sp_y = (ydim%side_y<1)? int(ydim/side_y) : ( (int)floor(int(ydim/side_y)));

	int x = t % xdim;  
	int y =  t / xdim;

	//int i = (x%sz==0)? int(x/side_x): ((int) floor(x/side_x)); // which col
    //int j = (y%sz==0)? int(y/side_y): ((int) floor(y/side_y)); //which row
    int i = floor(float(x)/side_x);
    int j = floor(float(y)/side_y);
	seg[t] = j + i*float(ceil(float(ydim)/float(sz))) +1;  
}

