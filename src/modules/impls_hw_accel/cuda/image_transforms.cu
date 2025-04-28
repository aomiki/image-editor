#include "image_transforms.h"
#include <cstring>
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
    cudaMalloc(&d_img, sizeof(matrix));
    cudaMemcpy(d_img, &img, sizeof(matrix), cudaMemcpyHostToDevice);

    unsigned char* d_cropped_arr;
    cudaMalloc(&d_cropped_arr, sizeof(unsigned char) * new_interlaced_size);
    img.set_arr_interlaced(d_cropped_arr, new_width, new_height);

    matrix* d_img_cropped;
    cudaMalloc(&d_img_cropped, sizeof(matrix));
    cudaMemcpy(d_img_cropped, &img, sizeof(matrix), cudaMemcpyHostToDevice);

    int blocksize_2d = (int)(total_blocksize/img.components_num);
    int blocksize_1d = (int)sqrt(blocksize_2d);

    int blocksnum_x = (int)(new_width / blocksize_1d + 1);
    int blocksnum_y = (int)(new_height / blocksize_1d + 1);

    dim3 blockSize(blocksize_1d, blocksize_1d, img.components_num);
    dim3 gridSize(blocksnum_x, blocksnum_y);

    kernel_crop_memcpy<<<gridSize, blockSize>>>(d_img, d_img_cropped, uint2 { crop_right, crop_top });

    cudaMemcpy(&img, d_img_cropped, sizeof(matrix), cudaMemcpyDeviceToHost);
    cudaFree(d_img);
    cudaFree(d_img_cropped);
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
    cudaMalloc(&d_img, sizeof(matrix));
    cudaMemcpy(d_img, &img, sizeof(matrix), cudaMemcpyHostToDevice);

    unsigned char* d_rotated_arr;
    cudaMalloc(&d_rotated_arr, sizeof(unsigned char) * img.size_interlaced());
    img.set_arr_interlaced(d_rotated_arr, new_width, new_height);

    matrix* d_img_rotated;
    cudaMalloc(&d_img_rotated, sizeof(matrix));
    cudaMemcpy(d_img_rotated, &img, sizeof(matrix), cudaMemcpyHostToDevice);

    kernel_rotate_memcpy<<<gridSize, blockSize>>>(d_img, d_img_rotated, angle);

    cudaMemcpy(&img, d_img_rotated, sizeof(matrix), cudaMemcpyDeviceToHost);
    cudaFree(d_img);
    cudaFree(d_img_rotated);
}
