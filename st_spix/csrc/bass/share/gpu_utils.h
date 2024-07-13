/* #ifdef WIN32 */
/* #include <direct.h> */
/* #define GetCurrentDir _getcwd */
/* #else */
/* #include <unistd.h> */
/* #define GetCurrentDir getcwd */
/* #endif */

#include "cuda.h"
#include "cuda_runtime.h"
#include "device_launch_parameters.h"

__host__ int myCeil(int a, int b);
__host__ int get_max(int* seg, int nPts);
__host__ bool saveArray( int* pdata, size_t length, const std::string& file_path );
__host__ bool loadArray( int* pdata, size_t length, const std::string& file_path);
