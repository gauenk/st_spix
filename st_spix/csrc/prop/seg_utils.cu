
#ifndef OUT_OF_BOUNDS_LABEL
#define OUT_OF_BOUNDS_LABEL -1
#endif
#define THREADS_PER_BLOCK 512

#include <assert.h>
#include "seg_utils.h"

/**********************************************
***********************************************


            Find Border Pixels


***********************************************
**********************************************/

__host__ void CudaFindBorderPixels(const int* seg, bool* border, const int nPixels, const int nbatch,const int xdim,const int ydim, const int single_border){
    int num_block = ceil( double(nPixels) / double(THREADS_PER_BLOCK) ); 
    dim3 ThreadPerBlock(THREADS_PER_BLOCK,1);
    dim3 BlockPerGrid(num_block,nbatch);
    find_border_pixels<<<BlockPerGrid,ThreadPerBlock>>>(seg,border,nPixels,
                                                        nbatch, xdim, ydim,
                                                        single_border);
}

__global__  void find_border_pixels(const int* seg, bool* border, const int nPixels,
                                    const int nbatch, const int xdim, const int ydim,
                                    const int single_border){   
    int idx = threadIdx.x + blockIdx.x * blockDim.x;  
    if (idx>=nPixels) return; 

    border[idx]=0;  // init        
    // todo; add batch info here
    int x = idx % xdim;
    int y = idx / xdim;

    int C =  __ldg(&seg[idx]); // center 
    int N,S,E,W; // north, south, east,west            

    // -- check out of bounds --
    if ((y<1)||(x<1)||(y>=(ydim-1))||(x>=(xdim-1)))
    {
        border[idx] = 1;
        return;
    }
    N = __ldg(&seg[idx-xdim]); // above
    W = __ldg(&seg[idx-1]);  // left
    S = __ldg(&seg[idx+xdim]); // below
    E = __ldg(&seg[idx+1]);  // right
           
    // bool check0 = (C == N) and (N == W) and (W == S);
    if ( (C!=N) || (C!=S) || (C!=E) || (C!=W) ){
            border[idx]=1;  
    }


    return;        
}


/**********************************************
***********************************************


         Find Border Pixels @ END


***********************************************
**********************************************/


__host__ void CudaFindBorderPixels_end(const int* seg, bool* border, const int nPixels, const int nbatch,const int xdim,const int ydim, const int single_border){   
    int num_block = ceil( double(nPixels) / double(THREADS_PER_BLOCK) ); 
    dim3 ThreadPerBlock(THREADS_PER_BLOCK,1);
    dim3 BlockPerGrid(num_block,nbatch);
    find_border_pixels_end<<<BlockPerGrid,ThreadPerBlock>>>(seg,border,nPixels,
                                                            nbatch, xdim, ydim,
                                                            single_border);
}


__global__  void find_border_pixels_end(const int* seg, bool* border, const int nPixels,
                                        const int nbatch, const int xdim, const int ydim,
                                        const int single_border){   
    int idx = threadIdx.x + blockIdx.x * blockDim.x;  
    if (idx>=nPixels) return; 

    border[idx]=0;  // init        
    
    // todo; add batch info here
    int x = idx % xdim;
    int y = idx / xdim;

    int C = seg[idx]; // center 
    int N,S,E,W; // north, south, east,west            
    N=S=W=E=OUT_OF_BOUNDS_LABEL; // init 
    
    if (y>1){
        N = seg[idx-xdim]; // above
    }          
    if (x>1){
        W = seg[idx-1];  // left
    }
    if (y<ydim-1){
        S = seg[idx+xdim]; // below
    }   
    if (x<xdim-1){
        E = seg[idx+1];  // right
    }       
   
    // If the nbr is different from the central pixel and is not out-of-bounds,
    // then it is a border pixel.
    if ( (N>=0 && C!=N) || (S>=0 && C!=S) || (E>=0 && C!=E) || (W>=0 && C!=W) ){
            if (N>=0 && C>N) border[idx]=1; 
            if (S>=0 && C>S) border[idx]=1;
            if (E>=0 && C>E) border[idx]=1;
            if (W>=0 && C>W) border[idx]=1;
    }

    return;        
}

