#include "image_tools.h"
#include <stdio.h>
#include "utils.cuh"

__global__ void kernel_fillInterlaced(matrix* m, unsigned char* components)
{
    unsigned int x = threadIdx.x + blockDim.x * blockIdx.x;
    unsigned int y = threadIdx.y + blockDim.y * blockIdx.y;

    if (x >= m->width || y >= m->height)
        return;

   // printf("Index: %d (Block (%d, %d), Thread (%d, %d)), Component element: %d\n", t_id, blockIdx.x, blockIdx.y, threadIdx.x, threadIdx.y, components[threadIdx.y]);

    m->get(x, y)[threadIdx.z] = components[threadIdx.z];
}

/// @brief Расположение элементов будет - по оси x = ось x матрицы, по оси y = ось y матрицы, по оси z = компоненты каждого элемента матрицы
/// @param m 
/// @param components Значение которое нужно установить
void matrix::fill(unsigned char *value)
{
    int blocksize_2d = (int)(1024/this->components_num);
    int blocksize_1d = (int)sqrt(blocksize_2d);

    int blocksnum_x = (int)(this->width / blocksize_1d + 1);
    int blocksnum_y = (int)(this->height / blocksize_1d + 1);

    dim3 blockSize(blocksize_1d, blocksize_1d, this->components_num);
    dim3 gridSize(blocksnum_x, blocksnum_y);

    matrix* d_m = transferMatrixToDevice(this);

    unsigned char* d_vals;
    cuda_log(cudaMalloc(&d_vals, this->components_num * sizeof(unsigned char)));
    cuda_log(cudaMemcpy(d_vals, value, this->components_num * sizeof(unsigned char), cudaMemcpyHostToDevice));

    kernel_fillInterlaced<<<gridSize, blockSize>>>(d_m, d_vals);
    cuda_log(cudaDeviceSynchronize());
    fflush(stdout);

    transferMatrixDataToHost(this, d_m);

    cuda_log(cudaFree(d_vals));
}
