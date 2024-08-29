#include "cuda.h"
#include "cuda_runtime.h"
#include "device_launch_parameters.h"

/* #include "../bass/share/my_sp_struct.h" */
#ifndef MY_SP_STRUCT
#define MY_SP_STRUCT
#include "../bass/share/my_sp_struct.h"
#endif

__global__ void warm_up_gpu();
__host__ void warm_up();

__host__ void CudaFindBorderPixels( const int* seg, bool* border, const int nPixels, const int nbatch, const int xdim, const int ydim, const int single_border);
__host__ void CudaFindBorderPixels_end( const int* seg, bool* border, const int nPixels, const int nbatch, const int xdim, const int ydim, const int single_border);

__global__  void find_border_pixels( const int* seg, bool* border, const int nPixels, const int nbatch, const int xdim, const int ydim, const int single_border);
__global__  void find_border_pixels_end(const int* seg, bool* border, const int nPixels, const int nbatch, const int xdim, const int ydim, const int single_border);

__host__ void update_prop_seg(float* img, int* seg,int* seg_potts_label, bool* border,
                              superpixel_params* sp_params,
                              superpixel_params* sp_params_prev,
                              superpixel_GPU_helper* sp_gpu_helper,
                              const float3 J_i, const float logdet_Sigma_i,
                              bool cal_cov, float i_std, int s_std, int nInnerIters,
                              const int nPixels, const int nSPs, int nSPs_buffer,
                              int nbatch, int xdim, int ydim, int nftrs,
                              float beta_potts_term, bool use_transition,
                              float* debug_seg);

__global__  void update_prop_seg_subset(
    float* img, int* seg,
    int* seg_potts_label, bool* border,
    superpixel_params* sp_params,
    superpixel_params* sp_params_prev,
    superpixel_GPU_helper* sp_gpu_helper,
    const float3 J_i, const float logdet_Sigma_i,
    bool cal_cov, float i_std, int s_std,
    const int nPts,const int nSuperpixels,
    int nbatch, int xdim, int ydim, int nftrs,
    const int xmod3, const int ymod3,
    const float betta_potts_term,
    const bool use_transition, float* debug_seg);

/* __global__ void update_seg_label(int* seg, int* seg_potts_label,const int nPts); */
/* __global__  void cal_posterior( float* img, int* seg, bool* border, superpixel_params* sp_params, float3 J_i, float logdet_Sigma_i, float i_std, int s_std, int* changes, int nPts , int xdim); */



__device__ inline
void _pix_lab_to_rgb(float* image_gpu_double, float* img_gpu){


    double L = image_gpu_double[0] *(-100);//* 100;
	double La = image_gpu_double[1]*100;//* 100;
	double Lb = image_gpu_double[2]*100; //*100 ;

    double fy = (L+16) / 116;
	double fx = La/500 + fy;
	double fz = fy-Lb/200;

	double x,y,z;
	double xcube = powf(fx,3);
	double ycube = powf(fy,3);
	double zcube = powf(fz,3);
	if (ycube>0.008856)	y = ycube;
	else				y = (fy-16.0/116.0)/7.787;
	if (xcube>0.008856)	x = xcube;
	else				x = (fx - 16.0/116.0)/7.787;
	if (zcube>0.008856)	z = zcube;
	else				z = (fz - 16.0/116.0)/7.787;

	double X = 0.950456 * x;
	double Y = 1.000 * y;
	double Z = 1.088754 * z;

	//convert from XYZ to rgb
	double R = X *  3.2406 + Y * (-1.5372) + Z * (-0.4986);
	double G = X * -0.9689 + Y * 1.8758 + Z *  0.0415;
	double B = X *  0.0557 + Y * (-0.2040) + Z * 1.0570;

	double r,g,b;
	if (R>0.0031308) r = 1.055 * (powf(R,(1.0/2.4))) - 0.055;
	else             r = 12.92 * R;
	if (G>0.0031308) g = 1.055 * ( powf(G,(1.0/2.4))) - 0.055;
	else             g = 12.92 * G;
	if (B>0.0031308) b = 1.055 * (powf(B, (1.0/2.4))) - 0.055;
	else             b = 12.92 * B;

    img_gpu[0] = b;
    img_gpu[1] = g;
    img_gpu[2] = r;

}

__device__ inline float2 cal_posterior_prop(
    float* imgC, int* seg, int x, int y,
    superpixel_params* sp_params,
    superpixel_params* sp_params_prev,
    int seg_idx, float3 J_i,
    float logdet_Sigma_i, float i_std, int s_std,
    float potts, float beta, float2 res_max,
    float xfer_a, float xfer_b, float* debugC){

    // -- init res --
    float res = -1000; // some large negative number
    /* float* imgC = img + idx * 3; */

    // -- compute color/spatial differences --
    const float x0 = __ldg(&imgC[0])-__ldg(&sp_params[seg_idx].mu_i.x);
    const float x1 = __ldg(&imgC[1])-__ldg(&sp_params[seg_idx].mu_i.y);
    const float x2 = __ldg(&imgC[2])-__ldg(&sp_params[seg_idx].mu_i.z);
    const int d0 = x - __ldg(&sp_params[seg_idx].mu_s.x);
    const int d1 = y - __ldg(&sp_params[seg_idx].mu_s.y);

    // -- color component --
    const float J_i_x = J_i.x;
    const float J_i_y = J_i.y;
    const float J_i_z = J_i.z;
    const float sigma_s_x = __ldg(&sp_params[seg_idx].sigma_s.x);
    const float sigma_s_y = __ldg(&sp_params[seg_idx].sigma_s.y);
    const float sigma_s_z = __ldg(&sp_params[seg_idx].sigma_s.z);
    const float logdet_sigma_s = __ldg(&sp_params[seg_idx].logdet_Sigma_s);

    // -- color component --
    res = res - (x0*x0*J_i_x + x1*x1*J_i_y + x2*x2*J_i_z);
    //res = -calc_squared_mahal_3d(imgC,mu_i,J_i);
    res = res - logdet_Sigma_i; // okay; log p(x,y) = -log detSigma

    // -- space component --
    res = res - d0*d0*sigma_s_x - d1*d1*sigma_s_z - 2*d0*d1*sigma_s_y;
    res = res - logdet_sigma_s;
    res = res - beta*potts;

    // -- fill debug --
    /* debugC[0] = -(x0*x0*J_i_x + x1*x1*J_i_y + x2*x2*J_i_z); */
    /* debugC[1] = -d0*d0*sigma_s_x - d1*d1*sigma_s_z - 2*d0*d1*sigma_s_y; */
    /* debugC[2] = -beta*potts; */
    /* debugC[3] = res; */
    /* debugC[4] = 1.*seg_idx; */
    /* _pix_lab_to_rgb(imgC, debugC+5); */
    /* /\* debugC[5] = imgC[0]; *\/ */
    /* /\* debugC[6] = imgC[1]; *\/ */
    /* /\* debugC[7] = imgC[2]; *\/ */
    /* debugC[8] = sp_params[seg_idx].mu_i.x; */
    /* debugC[9] = sp_params[seg_idx].mu_i.y; */
    /* debugC[10] = sp_params[seg_idx].mu_i.z; */


/*     res = res - (x0*x0*J_i_x + x1*x1*J_i_y + x2*x2*J_i_z); */
/*     //res = -calc_squared_mahal_3d(imgC,mu_i,J_i); */
/*     res = res - logdet_Sigma_i; */

/*     // -- space component -- */
/*     res = res - d0*d0*sigma_s_x; */
/*     res = res - d1*d1*sigma_s_z; */
/*     res = res -  2*d0*d1*sigma_s_y; */
/*     // res -= calc_squared_mahal_2d(pt,mu_s,J_s); */
/*     res = res -  logdet_sigma_s; */
/*     res = res -beta*potts; */
/*     /\*if (res > atomicMaxFloat2(&post_changes[idx].post[4],res)) */
/*     { */

    /* if( (res+xfer_a)>(res_max.x+xfer_b)) // include transition terms */
    if(res>res_max.x) // include transition terms
    {
      res_max.x = res;
      res_max.y = seg_idx;
    }

    return res_max;
}


