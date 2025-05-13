#include "image_filters.h"
#include "_shared_definitions.h"
#include "utils.cuh"
#include <cmath>
#include <cstring>

__global__ void kernel_grayscale(matrix* img, float3 rgb_coeffs)
{
    const unsigned int x = threadIdx.x + blockDim.x * blockIdx.x;
    const unsigned int y = threadIdx.y + blockDim.y * blockIdx.y;

    if (x >= img->width || y >= img->height)
    {
        return;
    }

    unsigned char* pixel = img->get(x, y);

    float gray_float = 
        pixel[0] * rgb_coeffs.x +  // R 
        pixel[1] * rgb_coeffs.y +  // G 
        pixel[2] * rgb_coeffs.z;   // B 
    
    const unsigned char gray = static_cast<unsigned char>(gray_float + 0.5f);
    
    memset(pixel, gray, 3);
}

__global__ void kernel_gaussian_blur(matrix* img, matrix* img_temp, float sigma, float radius)
{
    const unsigned int x = threadIdx.x + blockDim.x * blockIdx.x;
    const unsigned int y = threadIdx.y + blockDim.y * blockIdx.y;

    if (x >= img->width || y >= img->height)
    {
        return;
    }

    float r = 0, g = 0, b = 0;
    float weight_sum = 0;

    // Применяем гауссово ядро
    for (int dy = -radius; dy <= radius; ++dy) {
        for (int dx = -radius; dx <= radius; ++dx) {
            int nx = clamp(static_cast<int>(x) + dx, 0, static_cast<int>(img->width)-1);
            int ny = clamp(static_cast<int>(y) + dy, 0, static_cast<int>(img->height)-1);

            // Вычисляем вес
            const float weight = exp(-(dx*dx + dy*dy)/(2 * sigma * sigma));

            const unsigned char* p = img_temp->get(nx, ny);
            r += p[0] * weight;
            g += p[1] * weight;
            b += p[2] * weight;
            weight_sum += weight;
        }
    }

    // Нормализация и запись результата
    unsigned char* dst = img->get(x, y);
    dst[0] = static_cast<unsigned char>(r / weight_sum);
    dst[1] = static_cast<unsigned char>(g / weight_sum);
    dst[2] = static_cast<unsigned char>(b / weight_sum);
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

void gaussian_blur(matrix& img, float sigma) {
    if (img.components_num < 3) return;
    
    const int radius = static_cast<int>(3 * sigma);

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

    unsigned char* d_temp_arr;
    cuda_log(cudaMalloc(&d_temp_arr, sizeof(unsigned char) * img.size_interlaced()));
    cuda_log(cudaMemcpy(d_temp_arr, img.get_arr_interlaced(), sizeof(unsigned char) * img.size_interlaced(), cudaMemcpyDeviceToDevice));

    matrix* d_temp_img;
    cuda_log(cudaMalloc(&d_temp_img, sizeof(matrix)));
    unsigned char* src_arr = img.get_arr_interlaced();
    img.set_arr_interlaced(d_temp_arr);
    cuda_log(cudaMemcpy(d_temp_img, &img, sizeof(matrix), cudaMemcpyHostToDevice));
    img.set_arr_interlaced(src_arr);

    kernel_gaussian_blur<<<gridSize, blockSize>>>(d_img, d_temp_img, sigma, radius);

    cuda_log(cudaFree(d_temp_arr));
    cuda_log(cudaFree(d_img));
    cuda_log(cudaFree(d_temp_img));
}
