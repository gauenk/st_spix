#include <string>
#include <iostream>
#include <fstream>
#include <sstream>
#include <float.h>
using namespace std;

#define THREADS_PER_BLOCK 512


// b = 5; 1-5 ->1, 6-10 ->2, 11-15 ->3
__host__ int myCeil(int a, int b){
    if (a%b==0) return a/b;
    else return ceil(double(a)/double(b));
}


/* find the maximum value in the seg_arr */
__host__ int get_max(int* seg, int nPts){
    int max_val = 0;
    for (int i = 0; i<nPts; i++){
        if (seg[i]>max_val){
            max_val = seg[i];
        }
    }
    return max_val;
}


__host__ bool saveArray( int* pdata, size_t length, const std::string& file_path )
{
    std::ofstream os(file_path.c_str(), std::ios::binary | std::ios::out);
    if ( !os.is_open() )
        return false;
    os.write(reinterpret_cast<const char*>(pdata), std::streamsize(length*sizeof(int)));
    os.close();
    return true;
}

__host__ bool loadArray( int* pdata, size_t length, const std::string& file_path)
{
    std::ifstream is(file_path.c_str(), std::ios::binary | std::ios::in);
    if ( !is.is_open() )
        return false;
    is.read(reinterpret_cast<char*>(pdata), std::streamsize(length*sizeof(int)));
    is.close();
    return true;
}


