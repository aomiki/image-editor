#include "image_filters.h"
#include "utils.cuh"
#include <cmath>
#include <cstring>

__global__ void kernel_grayscale(matrix* img, float3 rgb_coeffs)
{
    unsigned int x = threadIdx.x + blockDim.x * blockIdx.x;
    unsigned int y = threadIdx.y + blockDim.y * blockIdx.y;

    if (x >= img->width || y >= img->height)
    {
        return;
    }

    unsigned char* pixel = img->get(x, y);

    float gray_float = 
        pixel[0] * rgb_coeffs.x +  // R 
        pixel[1] * rgb_coeffs.y +  // G 
        pixel[2] * rgb_coeffs.z;   // B 
    
    unsigned char gray = static_cast<unsigned char>(gray_float + 0.5f);
    
    memset(pixel, gray, 3);
}

void grayscale(matrix &img) {

    const float3 rgb_coeffs{
        0.2126f,
        0.7152f,
        0.0722f
    };

    if (img.components_num < 3) return;

    unsigned total_blocksize = 32;
    if (img.size() >= 4480)
    {
        total_blocksize = 128;
    }

    if (img.size() >= 8960)
    {
        total_blocksize = 256;
    }

    if (img.size() >= 17920)
    {
        total_blocksize = 512;
    }

    if (img.size() >= 35840)
    {
        total_blocksize = 1024;
    }

    int blocksize_2d = (int)(total_blocksize/img.components_num);
    int blocksize_1d = (int)sqrt(blocksize_2d);

    int blocksnum_x = (int)(img.width / blocksize_1d + 1);
    int blocksnum_y = (int)(img.height / blocksize_1d + 1);

    dim3 blockSize(blocksize_1d, blocksize_1d, img.components_num);
    dim3 gridSize(blocksnum_x, blocksnum_y);

    matrix* d_img;
    cuda_log(cudaMalloc(&d_img, sizeof(matrix)));
    cuda_log(cudaMemcpy(d_img, &img, sizeof(matrix), cudaMemcpyHostToDevice));

    kernel_grayscale<<<gridSize, blockSize>>>(d_img, rgb_coeffs);

    cuda_log(cudaFree(d_img));
}
