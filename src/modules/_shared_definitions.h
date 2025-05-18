
//some generic functions should be executable on GPU
// if file is built by nvcc, then the attributes are defined, 
// if by anything else - then not
#ifdef __CUDACC__
    #ifndef _SHARED_DEFINITIONS_CUDA
    #define _SHARED_DEFINITIONS_CUDA

    #define __shared_func__ __host__ __device__
    #define vec3  float3
    #define screen_coords uint2

    /// @brief clamp from std <algorithms>
    /// @note Because I couldn't find CUDA implementation of one.
    __shared_func__ inline float clamp(float x, float min, float max)
    {
        const float t = x < min ? min : x;
        return t > max ? max : t;
    }

    #endif
#else
    #ifndef _SHARED_DEFINITIONS_CPU
    #define _SHARED_DEFINITIONS_CPU

    #define __shared_func__
    #define vec3 vertex
    #define screen_coords matrix_coord
    using namespace std;
    #include <algorithm>

    #endif
#endif
