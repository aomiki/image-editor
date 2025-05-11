#include "image_transforms.h"
#include <cstring>
#include "utils.cuh"
using namespace std;

__global__ void kernel_crop_memcpy(matrix* img_src, matrix* img_dest, uint2 src_offset)
{
    unsigned int x = threadIdx.x + blockDim.x * blockIdx.x;
    unsigned int y = threadIdx.y + blockDim.y * blockIdx.y;

    if (x >= img_dest->width || y >= img_dest->height)
    {
        return;
    }

    unsigned char* old_pixel = img_src->get(x + src_offset.x, y + src_offset.y);
    unsigned char* new_pixel = img_dest->get(x, y);
    for (size_t i = 0; i < img_src->components_num; i++)
    {
        new_pixel[i] = old_pixel[i];
    }
}

__global__ void kernel_rotate_memcpy(matrix* img_src, matrix* img_dest, unsigned angle)
{
    unsigned int x = threadIdx.x + blockDim.x * blockIdx.x;
    unsigned int y = threadIdx.y + blockDim.y * blockIdx.y;

    if (x >= img_src->width || y >= img_src->height)
    {
        return;
    }

    unsigned char* old_pixel = img_src->get(x, y); 
    unsigned char* new_pixel = nullptr;  
    
    switch (angle) {
        case 90:
            new_pixel = img_dest->get(img_dest->width - y - 1, x);
            break;
        case 180:
            new_pixel = img_dest->get(img_dest->width - x - 1, img_dest->height - y -1);
            break;
        case 270:
            new_pixel = img_dest->get(y, img_dest->height - x - 1);
            break;
    }

    for (size_t i = 0; i < img_src->components_num; i++)
    {
        new_pixel[i] = old_pixel[i];
    }
}

__global__ void kernel_reflect_memcpy(matrix* img_src, matrix* img_dest, bool horizontal, bool vertical)
{
    unsigned int x = threadIdx.x + blockDim.x * blockIdx.x;
    unsigned int y = threadIdx.y + blockDim.y * blockIdx.y;

    if (x >= img_src->width || y >= img_src->height)
    {
        return;
    }

    unsigned target_x = vertical ? (img_src->width - 1 - x) : x;
    unsigned target_y = horizontal ? (img_src->height - 1 - y) : y;

    unsigned char* src = img_src->get(x, y);
    unsigned char* dest = img_dest->get(target_x, target_y);

    for (size_t i = 0; i < img_src->components_num; i++)
    {
        dest[i] = src[i];
    }
}

__global__ void kernel_shear_memcpy(matrix* img_src, matrix* img_dest, float2 sh, float2 old_center, float2 new_center)
{
    unsigned int x = threadIdx.x + blockDim.x * blockIdx.x;
    unsigned int y = threadIdx.y + blockDim.y * blockIdx.y;

    if (x >= img_src->width || y >= img_src->height)
    {
        return;
    }

    float2 src_coords {
        old_center.x + (x - new_center.x) - sh.x*(y - new_center.y),
        old_center.y + (y - new_center.y) - sh.y*(x - new_center.x)
    };

    if (src_coords.x >= 0 && src_coords.x < img_src->width && src_coords.y >= 0 && src_coords.y < img_src->height) {
        bilinear_interpolate(*img_src, src_coords.x, src_coords.y, img_dest->get(x, y));
    }
}

void crop(matrix& img, unsigned crop_left, unsigned crop_top, unsigned crop_right, unsigned crop_bottom) 
{
    unsigned new_width = img.width - crop_left - crop_right;
    unsigned new_height = img.height - crop_top - crop_bottom;
    unsigned new_interlaced_size = new_width * new_height * img.components_num;

    if (new_width <= 0 || new_height <= 0) 
    {
        return;
    }

    if (crop_left + crop_right > img.width || crop_top + crop_bottom > img.height) 
    {
        return;
    }

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

    matrix* d_img;
    cuda_log(cudaMalloc(&d_img, sizeof(matrix)));
    cuda_log(cudaMemcpy(d_img, &img, sizeof(matrix), cudaMemcpyHostToDevice));

    unsigned char* d_cropped_arr;
    cuda_log(cudaMalloc(&d_cropped_arr, sizeof(unsigned char) * new_interlaced_size));
    img.set_arr_interlaced(d_cropped_arr, new_width, new_height);

    matrix* d_img_cropped;
    cuda_log(cudaMalloc(&d_img_cropped, sizeof(matrix)));
    cuda_log(cudaMemcpy(d_img_cropped, &img, sizeof(matrix), cudaMemcpyHostToDevice));

    int blocksize_2d = (int)(total_blocksize/img.components_num);
    int blocksize_1d = (int)sqrt(blocksize_2d);

    int blocksnum_x = (int)(new_width / blocksize_1d + 1);
    int blocksnum_y = (int)(new_height / blocksize_1d + 1);

    dim3 blockSize(blocksize_1d, blocksize_1d, img.components_num);
    dim3 gridSize(blocksnum_x, blocksnum_y);

    kernel_crop_memcpy<<<gridSize, blockSize>>>(d_img, d_img_cropped, uint2 { crop_right, crop_top });

    cuda_log(cudaMemcpy(&img, d_img_cropped, sizeof(matrix), cudaMemcpyDeviceToHost));
    cuda_log(cudaFree(d_img));
    cuda_log(cudaFree(d_img_cropped));
}

void rotate(matrix& img, unsigned angle) 
{
    angle = angle % 360;  
    if (angle == 0) return; 

    unsigned new_width = (angle == 90 || angle == 270) ? img.height : img.width;
    unsigned new_height = (angle == 90 || angle == 270) ? img.width : img.height;

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

    unsigned char* d_rotated_arr;
    cuda_log(cudaMalloc(&d_rotated_arr, sizeof(unsigned char) * img.size_interlaced()));
    unsigned char* d_src_arr = img.get_arr_interlaced();
    img.set_arr_interlaced(d_rotated_arr, new_width, new_height);

    matrix* d_img_rotated;
    cuda_log(cudaMalloc(&d_img_rotated, sizeof(matrix)));
    cuda_log(cudaMemcpy(d_img_rotated, &img, sizeof(matrix), cudaMemcpyHostToDevice));

    kernel_rotate_memcpy<<<gridSize, blockSize>>>(d_img, d_img_rotated, angle);

    //cudaMemcpy(&img, d_img_rotated, sizeof(matrix), cudaMemcpyDeviceToHost);
    cuda_log(cudaFree(d_img));
    cuda_log(cudaFree(d_src_arr));
    cuda_log(cudaFree(d_img_rotated));
}

void reflect(matrix& img, bool horizontal, bool vertical) 
{
    if (!horizontal && !vertical) return;
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

    unsigned char* d_dest_arr;
    cuda_log(cudaMalloc(&d_dest_arr, sizeof(unsigned char) * img.size_interlaced()));
    unsigned char* d_src_arr = img.get_arr_interlaced();
    img.set_arr_interlaced(d_dest_arr, img.width, img.height);

    matrix* d_dest_img;
    cuda_log(cudaMalloc(&d_dest_img, sizeof(matrix)));
    cuda_log(cudaMemcpy(d_dest_img, &img, sizeof(matrix), cudaMemcpyHostToDevice));

    kernel_reflect_memcpy<<<gridSize, blockSize>>>(d_img, d_dest_img, horizontal, vertical);

    //cudaMemcpy(&img, d_dest_img, sizeof(matrix), cudaMemcpyDeviceToHost);
    cuda_log(cudaFree(d_img));
    cuda_log(cudaFree(d_src_arr));
    cuda_log(cudaFree(d_dest_img));
}

void shear(matrix& img, float shx, float shy) {
    unsigned new_width = static_cast<unsigned>(img.width + 2*std::abs(shy)*img.height);
    unsigned new_height = static_cast<unsigned>(img.height + 2*std::abs(shx)*img.width);

    float2 new_center {
        new_width / 2.0f,
        new_height / 2.0f
    };

    float2 old_center {
        img.width / 2.0f,
        img.height / 2.0f
    };

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

    unsigned char* d_dest_arr;
    cuda_log(cudaMalloc(&d_dest_arr, sizeof(unsigned char) * img.size_interlaced()));
    unsigned char* d_src_arr = img.get_arr_interlaced();
    img.set_arr_interlaced(d_dest_arr, img.width, img.height);

    matrix* d_dest_img;
    cuda_log(cudaMalloc(&d_dest_img, sizeof(matrix)));
    cuda_log(cudaMemcpy(d_dest_img, &img, sizeof(matrix), cudaMemcpyHostToDevice));

    kernel_shear_memcpy<<<gridSize, blockSize>>>(d_img, d_dest_img, float2 { shx, shy }, old_center, new_center);

    cuda_log(cudaFree(d_img));
    cuda_log(cudaFree(d_src_arr));
    cuda_log(cudaFree(d_dest_img));
}
