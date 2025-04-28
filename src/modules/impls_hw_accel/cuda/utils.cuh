#ifndef UTILS_CUH
#define UTILS_CUH

#include "image_tools.h"
#include "cublas_v2.h"
#include <string>
#include <nvjpeg.h>

#define cuda_log(result) cuda_log_detailed(result, __FILE__, __LINE__);

__host__ __device__ void cuda_log_detailed(cudaError_t err, const char *file, int line);
__host__ __device__ void cuda_log_detailed(cublasStatus_t err, const char *file, int line);
__host__ __device__ void cuda_log_detailed(nvjpegStatus_t err, const char *file, int line);

void transferMatrixToDevice(matrix* d_m, unsigned char* d_arr_interlaced, matrix* h_m);
void transferMatrixDataToHost(matrix* h_m, matrix* d_m, bool do_free = true);

#endif
