#ifdef WIN32
#include <direct.h>
#define GetCurrentDir _getcwd
#else
#include <unistd.h>
#define GetCurrentDir getcwd
#endif

// #include <opencv2/core/core.hpp>
// #include <opencv2/highgui/highgui.hpp>
#include <iostream>
#include <cmath>
// #include <thrust/system_error.h>
// #include <thrust/system/cuda/error.h>

// using namespace cv;
using namespace std;

float calc_gamma_function(float x);
std::string get_current_dir();
std::string get_curr_work_dir();
// void save_image(cv::Mat img, String img_name, String img_path);
// void writeMatToFile(cv::Mat& m, const char* filename);
