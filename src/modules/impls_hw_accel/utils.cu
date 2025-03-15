#include "utils.cuh"

void cuda_log(cudaError_t err)
{
    LAST_CUDA_ERROR = err;
    LAST_CUDA_ERROR_DESC = cudaGetErrorString(err);
}

matrix* transferMatrixToDevice(matrix* h_m)
{
    matrix* d_m;
    unsigned char* h_arr = h_m->get_arr_interlaced();
    unsigned char* d_arr;

    cuda_log(cudaMalloc(&d_m,  sizeof(matrix)));
    cuda_log(cudaMalloc(&d_arr, h_m->size_interlaced()));
    cuda_log(cudaMemcpy(d_arr, h_arr, h_m->size_interlaced(), cudaMemcpyHostToDevice));
    h_m->set_arr_interlaced(d_arr);

    cuda_log(cudaMemcpy(d_m, h_m, sizeof(matrix), cudaMemcpyHostToDevice));

    h_m->set_arr_interlaced(h_arr);

    return d_m;
}

void transferMatrixDataToHost(matrix* h_m, matrix* d_m, bool do_free)
{
    unsigned char* h_arr = h_m->get_arr_interlaced();

    cuda_log(cudaMemcpy(h_m, d_m, sizeof(matrix), cudaMemcpyDeviceToHost));
    cuda_log(cudaMemcpy(h_arr, h_m->get_arr_interlaced(), h_m->size_interlaced(), cudaMemcpyDeviceToHost));

    if (do_free)
    {
        cudaFree(h_m->get_arr_interlaced());
        cudaFree(d_m);
    }

    h_m->set_arr_interlaced(h_arr);
}
