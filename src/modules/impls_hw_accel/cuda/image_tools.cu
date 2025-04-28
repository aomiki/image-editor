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
    unsigned total_blocksize = 32;
    if (size() >= 4480)
    {
        total_blocksize = 128;
    }

    if (size() >= 8960)
    {
        total_blocksize = 256;
    }

    if (size() >= 17920)
    {
        total_blocksize = 512;
    }

    if (size() >= 35840)
    {
        total_blocksize = 1024;
    }

    int blocksize_2d = (int)(total_blocksize/this->components_num);
    int blocksize_1d = (int)sqrt(blocksize_2d);

    int blocksnum_x = (int)(this->width / blocksize_1d + 1);
    int blocksnum_y = (int)(this->height / blocksize_1d + 1);

    dim3 blockSize(blocksize_1d, blocksize_1d, this->components_num);
    dim3 gridSize(blocksnum_x, blocksnum_y);

    matrix* d_m;
    const unsigned d_m_bytes = sizeof(matrix);

    unsigned char* d_vals;
    const unsigned d_vals_bytes = this->components_num * sizeof(unsigned char);

    char* d_membuf;
    cuda_log(cudaMalloc(
        &d_membuf,
        d_m_bytes +
        d_vals_bytes
    ));

    unsigned mem_offset = 0;

    d_m = (matrix*)d_membuf;
    mem_offset += d_m_bytes;

    d_vals = (unsigned char*)(d_membuf + mem_offset);
    mem_offset += d_vals_bytes;

    cuda_log(cudaMemcpy(d_m, (matrix*)this, sizeof(matrix), cudaMemcpyHostToDevice));
    cuda_log(cudaMemcpy(d_vals, value, d_vals_bytes, cudaMemcpyHostToDevice));

    kernel_fillInterlaced<<<gridSize, blockSize>>>(d_m, d_vals);
    cuda_log(cudaDeviceSynchronize());
    fflush(stdout);

    cuda_log(cudaFree(d_membuf));
}

matrix::matrix(unsigned int components_num, unsigned width, unsigned height)
{
    this->height = 0;
    this->width = 0;
    this->components_num = components_num;
    set_arr_interlaced(nullptr);

    resize(width, height);
}

void matrix::resize(unsigned width, unsigned height)
{
    unsigned int old_size = size_interlaced();

    this->width = width;
    this->height = height;

    unsigned char* d_arr_interlaced;
    const unsigned d_arr_interlaced_bytes = size_interlaced();

    char* d_membuf;
    cuda_log(cudaMalloc(
        &d_membuf,
        d_arr_interlaced_bytes
    ));

    unsigned mem_offset = 0;

    d_arr_interlaced = (unsigned char*)(d_membuf + mem_offset);
    mem_offset += d_arr_interlaced_bytes;

    if (old_size != 0)
    {
        cuda_log(cudaFree(get_arr_interlaced()));
    }

    set_arr_interlaced(d_arr_interlaced);
}

matrix::~matrix()
{
    if (size_interlaced() != 0)
    {
        cuda_log(cudaFree(get_arr_interlaced()));
    }
}
