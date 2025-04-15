#include "utils.cuh"

__host__ __device__ void cuda_log_detailed(cudaError_t err, const char *file, int line)
{
    if (err != cudaSuccess)
    {
        const char* desc = cudaGetErrorString(err);
        printf("[%s:%d] CUDA ERROR %d: %s\n", file, line, err, desc);
    }
}

__host__ __device__ void cuda_log_detailed(cublasStatus_t err, const char *file, int line)
{
    if (err != CUBLAS_STATUS_SUCCESS)
    {
        printf("[%s:%d] CUBLAS ERROR %d\n", file, line, err);
    }
}

__host__ __device__ void cuda_log_detailed(nvjpegStatus_t status, const char *file, int line)
{
    if (status != NVJPEG_STATUS_SUCCESS)
    {
        printf("[%s:%d] NVJPEG ERROR %d\n", file, line, status);
    }
}

void transferMatrixToDevice(matrix* d_m, unsigned char* d_arr_interlaced, matrix* h_m)
{
    unsigned char* h_arr = h_m->get_arr_interlaced();

    cuda_log(cudaMemcpy(d_arr_interlaced, h_arr, h_m->size_interlaced(), cudaMemcpyHostToDevice));
    h_m->set_arr_interlaced(d_arr_interlaced);

    cuda_log(cudaMemcpy(d_m, h_m, sizeof(matrix), cudaMemcpyHostToDevice));

    h_m->set_arr_interlaced(h_arr);
}

void transferMatrixDataToHost(matrix* h_m, matrix* d_m, bool do_free)
{
    unsigned char* h_arr = h_m->get_arr_interlaced();

    cuda_log(cudaMemcpy(h_m, d_m, sizeof(matrix), cudaMemcpyDeviceToHost));
    cuda_log(cudaMemcpy(h_arr, h_m->get_arr_interlaced(), h_m->size_interlaced(), cudaMemcpyDeviceToHost));

    if (do_free)
    {
        cuda_log(cudaFree(h_m->get_arr_interlaced()));
        cuda_log(cudaFree(d_m));
    }

    h_m->set_arr_interlaced(h_arr);
}
