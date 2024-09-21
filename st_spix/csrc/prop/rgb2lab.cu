
// -- imports --
#include "rgb2lab.h"
#include <stdio.h>
#include <math.h>
#include "pch.h"

// #include <torch/types.h>
// #include <torch/extension.h>

// -- define --
using namespace std;
#define THREADS_PER_BLOCK 512
#define CHECK_CUDA(x) TORCH_CHECK(x.device().is_cuda(), #x " must be a CUDA tensor")
#define CHECK_CONTIGUOUS(x) TORCH_CHECK(x.is_contiguous(), #x " must be contiguous")
#define CHECK_INPUT(x) CHECK_CUDA(x); CHECK_CONTIGUOUS(x)

__host__ void rgb2lab(float* img_rgb, float* img_lab, int npix, int nbatch){
	int num_block = ceil( double(npix) / double(THREADS_PER_BLOCK) ); 
	dim3 ThreadPerBlock(THREADS_PER_BLOCK,1);
	dim3 BlockPerGrid(num_block,nbatch);
	rgb_to_lab<<<BlockPerGrid,ThreadPerBlock>>>(img_rgb,img_lab,npix);
}

__host__ void lab2rgb(float* img_rgb, float* img_lab, int npix, int nbatch){
	int num_block = ceil( double(npix) / double(THREADS_PER_BLOCK) ); 
	dim3 ThreadPerBlock(THREADS_PER_BLOCK,1);
	dim3 BlockPerGrid(num_block,nbatch);
	lab_to_rgb<<<BlockPerGrid,ThreadPerBlock>>>(img_rgb,img_lab,npix);
}

__global__ void rgb_to_lab(float* img_rgb, float* img_lab, int npix) {

    // -- get pixel index --
	int t = threadIdx.x + blockIdx.x * blockDim.x;  
	if (t>=npix) return;
    t = t + npix*blockIdx.y; // offset via batch

	// double sB = (double)img_rgb[3*t];
	// double sG = (double)img_rgb[3*t+1];
	// double sR = (double)img_rgb[3*t+2];
	double sR = (double)img_rgb[3*t];
	double sG = (double)img_rgb[3*t+1];
	double sB = (double)img_rgb[3*t+2];

	if (sR!=sR || sG!=sG || sB!=sB) return; // ??

	//RGB (D65 illuninant assumption) to XYZ conversion
	double R = sR;
	double G = sG;
	double B = sB;

    // -- info --
	double r, g, b;
	if(R <= 0.04045)	r = R/12.92;
	else				r = powf((R+0.055)/1.055,2.4);
	if(G <= 0.04045)	g = G/12.92;
	else				g = powf((G+0.055)/1.055,2.4);
	if(B <= 0.04045)	b = B/12.92;
	else				b = powf((B+0.055)/1.055,2.4);

	double X = r*0.4124564 + g*0.3575761 + b*0.1804375;
	double Y = r*0.2126729 + g*0.7151522 + b*0.0721750;
	double Z = r*0.0193339 + g*0.1191920 + b*0.9503041;

	
	//convert from XYZ to LAB

	double epsilon = 0.008856;	//actual CIE standard
	double kappa   = 903.3;		//actual CIE standard

	double Xr = 0.950456;	//reference white
	double Yr = 1.0;		//reference white
	double Zr = 1.088754;	//reference white

	double xr = X/Xr;
	double yr = Y/Yr;
	double zr = Z/Zr;

	double fx, fy, fz;
	if(xr > epsilon)	fx = powf(xr, 1.0/3.0);
	else				fx = (kappa*xr + 16.0)/116.0;
	if(yr > epsilon)	fy = powf(yr, 1.0/3.0);
	else				fy = (kappa*yr + 16.0)/116.0;
	if(zr > epsilon)	fz = powf(zr, 1.0/3.0);
	else				fz = (kappa*zr + 16.0)/116.0;

	float lval = 116.0*fy-16.0;
	float aval = 500.0*(fx-fy);
	float bval = 200.0*(fy-fz);

	img_lab[3*t] = lval/(-100);
	img_lab[3*t+1] = aval/100;
	img_lab[3*t+2] = bval/100;
}

__global__ void lab_to_rgb(float* img_rgb, float* img_lab, int npix) {

    // -- get pixel index --
	int t = threadIdx.x + blockIdx.x * blockDim.x;  
	if (t>=npix) return;
    t = t + npix*blockIdx.y; // offset via batch

    double L = img_lab[3*t] *(-100);//* 100;
	double La = img_lab[3*t+1]*100;//* 100;
	double Lb = img_lab[3*t+2]*100; //*100 ;
	
	if (L!=L || La!=La || Lb!=Lb) return;

    //convert from LAB to XYZ
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

	float3 p;
	
	p.x =  min(1.0, b * 1.0);
	p.y =  min(1.0, g * 1.0);
	p.z =  min(1.0, r * 1.0);
	p.x =  max(0.0, double(p.x));
    p.y =  max(0.0, double(p.y));
    p.z =  max(0.0, double(p.z));

    // -- save image [RGB(z,y,x) not BGR(x,y,z)]  --
    img_rgb[3*t] = p.z;
    img_rgb[3*t+1] = p.y;
    img_rgb[3*t+2] = p.x;
}




/**********************************************************

             -=-=-=-=- Python API  -=-=-=-=-=-

***********************************************************/

torch::Tensor run_rgb_to_lab(torch::Tensor img_rgb){

    // -- check --
    CHECK_INPUT(img_rgb);

    // -- unpack --
    int nbatch = img_rgb.size(0);
    int height = img_rgb.size(1);
    int width = img_rgb.size(2);
    int nftrs = img_rgb.size(3);
    int npix = height*width;
    auto img_lab = img_rgb.clone();

    // -- run rgb to lab --
    rgb2lab(img_rgb.data<float>(), img_lab.data<float>(), npix, nbatch);

    return img_lab;
}

torch::Tensor run_lab_to_rgb(torch::Tensor img_lab){

    // -- check --
    CHECK_INPUT(img_lab);

    // -- unpack --
    int nbatch = img_lab.size(0);
    int height = img_lab.size(1);
    int width = img_lab.size(2);
    int nftrs = img_lab.size(3);
    int npix = height*width;
    auto img_rgb = img_lab.clone();

    // -- run rgb to lab --
    lab2rgb(img_rgb.data<float>(), img_lab.data<float>(), npix, nbatch);

    return img_rgb;
}


void init_rgb2lab(py::module &m){
  m.def("rgb_to_lab", &run_rgb_to_lab,"change colors");
  m.def("lab_to_rgb", &run_lab_to_rgb,"change colors");
}

