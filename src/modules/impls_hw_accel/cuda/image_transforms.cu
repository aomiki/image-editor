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

__global__ void kernel_shear_memcpy(matrix* img_src, matrix* img_dest, float2 sh, float2 offset)
{
    unsigned int x = threadIdx.x + blockDim.x * blockIdx.x;
    unsigned int y = threadIdx.y + blockDim.y * blockIdx.y;

    if (x >= img_dest->width || y >= img_dest->height)
    {
        return;
    }

    float2 src_coords {
        (x - offset.x) - sh.y*(y - offset.y),
        (y - offset.y) - sh.x*(x - offset.x)
    };

    if (src_coords.x >= 0 && src_coords.x < img_src->width && src_coords.y >= 0 && src_coords.y < img_src->height) {
        bilinear_interpolate(*img_src, src_coords.x, src_coords.y, img_dest->get(x, y));
    }
}

__global__ void kernel_rotate_initBatchPtrs(
    float** batch_ptrs_d_R_inv, float** batch_ptrs_new_coords, float** batch_ptrs_src_coords,
    float * d_R_inv, float* batch_new_coords, float* batch_src_coords,
    float new_half_width, float new_half_height,
    unsigned width, unsigned height)
{
    const unsigned int x = threadIdx.x + blockDim.x * blockIdx.x;
    const unsigned int y = threadIdx.y + blockDim.y * blockIdx.y;

    if (x >= width || y >= height)
    {
        return;
    }

    const unsigned i = y*width + x;

    batch_ptrs_d_R_inv[i] = d_R_inv;
    batch_ptrs_new_coords[i] = batch_new_coords + (i*2);
    batch_ptrs_src_coords[i] = batch_src_coords + (i*2);

    batch_ptrs_new_coords[i][0] = x - new_half_width;
    batch_ptrs_new_coords[i][1] = y - new_half_height;
}

__global__ void kernel_rotate_setPixels(matrix* img_src, matrix* img_dest, float** batch_ptrs_src_coords, float half_width, float half_height)
{
    const unsigned int x = threadIdx.x + blockDim.x * blockIdx.x;
    const unsigned int y = threadIdx.y + blockDim.y * blockIdx.y;

    if (x >= img_dest->width || y >= img_dest->height)
    {
        return;
    }

    const unsigned i = y*img_dest->width + x;

    const float2 src_coords {
        batch_ptrs_src_coords[i][0] + half_width,
        batch_ptrs_src_coords[i][1] + half_height
    };

    if (src_coords.x >= 0 && src_coords.x <= img_src->width-1 && 
        src_coords.y >= 0 && src_coords.y <= img_src->height-1) {
        unsigned char* pixel = img_dest->get(x, y);

        bilinear_interpolate(*img_src, src_coords.x, src_coords.y, pixel);
    } else {
        unsigned char* pixel = img_dest->get(x, y);
        memset(pixel, 255, img_src->components_num);
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
    const unsigned new_width = static_cast<unsigned>(img.width + std::abs(shy)*img.height);
    const unsigned new_height = static_cast<unsigned>(img.height + std::abs(shx)*img.width);

    const float2 offset {
        (shy > 0) ? 0 : std::abs(shy)*img.height,
        (shx > 0) ? 0 : std::abs(shx)*img.width
    };

    const unsigned target_size = new_width * new_height * img.components_num;
    unsigned total_blocksize = 32;
    if (target_size >= 4480)
    {
        total_blocksize = 128;
    }

    if (target_size >= 8960)
    {
        total_blocksize = 256;
    }

    if (target_size >= 17920)
    {
        total_blocksize = 512;
    }

    if (target_size >= 35840)
    {
        total_blocksize = 1024;
    }

    int blocksize_2d = (int)(total_blocksize/img.components_num);
    int blocksize_1d = (int)sqrt(blocksize_2d);

    int blocksnum_x = (int)(new_width / blocksize_1d + 1);
    int blocksnum_y = (int)(new_height / blocksize_1d + 1);

    dim3 blockSize(blocksize_1d, blocksize_1d, img.components_num);
    dim3 gridSize(blocksnum_x, blocksnum_y);

    matrix* d_img;
    cuda_log(cudaMalloc(&d_img, sizeof(matrix)));
    cuda_log(cudaMemcpy(d_img, &img, sizeof(matrix), cudaMemcpyHostToDevice));

    unsigned char* d_dest_arr;
    cuda_log(cudaMalloc(&d_dest_arr, sizeof(unsigned char) * target_size));
    unsigned char* d_src_arr = img.get_arr_interlaced();
    img.set_arr_interlaced(d_dest_arr, new_width, new_height);

    matrix* d_dest_img;
    cuda_log(cudaMalloc(&d_dest_img, sizeof(matrix)));
    cuda_log(cudaMemcpy(d_dest_img, &img, sizeof(matrix), cudaMemcpyHostToDevice));

    kernel_shear_memcpy<<<gridSize, blockSize>>>(d_img, d_dest_img, float2 { shx, shy }, offset);

    cuda_log(cudaFree(d_img));
    cuda_log(cudaFree(d_src_arr));
    cuda_log(cudaFree(d_dest_img));
}

void rotate(matrix& img, float angle) {
    cublasHandle_t handle;
    cuda_log(cublasCreate(&handle));
    cuda_log(cublasSetPointerMode(handle, CUBLAS_POINTER_MODE_HOST));

    float radians = angle * M_PI / 180.0f;
    float cos_theta = cos(radians);
    float sin_theta = sin(radians);

    const float R[] = {
        //column 1
        cos_theta,
        sin_theta,

        //column 2
        -sin_theta,
        cos_theta
    };
    float* d_R;
    const size_t R_bytes_size = 2*2 * sizeof(float);

    cuda_log(cudaMalloc(&d_R, R_bytes_size));
    cuda_log(cudaMemcpy(d_R, R, R_bytes_size, cudaMemcpyHostToDevice));

    const float half_height = img.height * 0.5f;
    const float half_width = img.width * 0.5f;

    //2 columns, 4 rows
    const float corners[] = {
        //column 1
        -half_width, 
         half_width, 
        -half_width, 
         half_width, 

         //column 2
        -half_height,
        -half_height,
        half_height,
        half_height
    };
    float* d_corners;
    const size_t corners_bytes_size = 4 * 2 * sizeof(float);

    cuda_log(cudaMalloc(&d_corners, corners_bytes_size));
    cuda_log(cudaMemcpy(d_corners, corners, corners_bytes_size, cudaMemcpyHostToDevice));

    const float alpha = 1.0f;
    const float beta = 0.0f;

    float* d_rotated_corners;
    cuda_log(cudaMalloc(&d_rotated_corners, corners_bytes_size));

    cuda_log(cublasSgemm(handle, CUBLAS_OP_N, CUBLAS_OP_N, 4, 2, 2, &alpha, d_corners, 4, d_R, 2, &beta, d_rotated_corners, 4));

    float rotated_corners[8];
    cuda_log(cudaMemcpy(rotated_corners, d_rotated_corners, corners_bytes_size, cudaMemcpyDeviceToHost));

    float2 min {
        rotated_corners[0],
        rotated_corners[1]
    };

    float2 max = min;

    for (int i = 1; i < 4; ++i) {
        min.x = std::min(min.x, rotated_corners[i*2]);
        max.x = std::max(max.x, rotated_corners[i*2]);
        min.y = std::min(min.y, rotated_corners[i*2+1]);
        max.y = std::max(max.y, rotated_corners[i*2+1]);
    }

    const unsigned new_width = static_cast<unsigned>(round(max.x - min.x));
    const unsigned new_height = static_cast<unsigned>(round(max.y - min.y));

    matrix* d_img;
    cuda_log(cudaMalloc(&d_img, sizeof(matrix)));
    cuda_log(cudaMemcpy(d_img, &img, sizeof(matrix), cudaMemcpyHostToDevice));

    unsigned char* d_dest_arr;
    cuda_log(cudaMalloc(&d_dest_arr, sizeof(unsigned char) * new_width * new_height * img.components_num));
    unsigned char* d_src_arr = img.get_arr_interlaced();
    img.set_arr_interlaced(d_dest_arr, new_width, new_height);

    matrix* d_dest_img;
    cuda_log(cudaMalloc(&d_dest_img, sizeof(matrix)));
    cuda_log(cudaMemcpy(d_dest_img, &img, sizeof(matrix), cudaMemcpyHostToDevice));

    const float new_half_width = new_width * 0.5f;
    const float new_half_height = new_height * 0.5f;

    const float R_inv[4] = {
        //column 1
        cos_theta,  
        -sin_theta, 

        //column 2
        sin_theta,
        cos_theta
    };
    float* d_R_inv;

    cuda_log(cudaMalloc(&d_R_inv, R_bytes_size));
    cuda_log(cudaMemcpy(d_R_inv, R_inv, R_bytes_size, cudaMemcpyHostToDevice));

    const unsigned batch_size = new_height* new_width;

    //array of input coords
    float* batch_new_coords;
    cuda_log(cudaMalloc(&batch_new_coords, sizeof(float) * 2 * batch_size));

    //array of resulting coords
    float* batch_src_coords;
    cuda_log(cudaMalloc(&batch_src_coords, sizeof(float) * 2 * batch_size));

    //array of the same pointer d_R_inv
    float** batch_ptrs_d_R_inv;
    cuda_log(cudaMalloc(&batch_ptrs_d_R_inv, sizeof(float*) * batch_size));

    //array of input coords
    float** batch_ptrs_new_coords;
    cuda_log(cudaMalloc(&batch_ptrs_new_coords, sizeof(float*) * batch_size));

    //array of resulting coords
    float** batch_ptrs_src_coords;
    cuda_log(cudaMalloc(&batch_ptrs_src_coords, sizeof(float*) * batch_size));

    const unsigned batch_iteration_num = new_height * new_width;
    unsigned poly_total_blocksize = 32;
    if (batch_iteration_num >= 4480)
    {
        poly_total_blocksize = 128;
    }

    if (batch_iteration_num >= 8960)
    {
        poly_total_blocksize = 256;
    }

    if (batch_iteration_num >= 17920)
    {
        poly_total_blocksize = 512;
    }

    if (batch_iteration_num >= 35840)
    {
        poly_total_blocksize = 1024;
    }

    unsigned blocksize_1d = (unsigned)sqrtf(poly_total_blocksize);

    unsigned blocknum_x = (unsigned)((new_width/blocksize_1d) +1);
    unsigned blocknum_y = (unsigned)((new_height/blocksize_1d) +1);

    dim3 blocksize(blocksize_1d, blocksize_1d);
    dim3 blocknum(blocknum_x, blocknum_y);

    //initialize all arrays for batched gemv
    kernel_rotate_initBatchPtrs<<<blocknum, blocksize>>>(batch_ptrs_d_R_inv, batch_ptrs_new_coords, batch_ptrs_src_coords, d_R_inv, batch_new_coords, batch_src_coords, new_half_width, new_half_height, new_width, new_height);
    cuda_log(cudaGetLastError());
    cuda_log(cudaDeviceSynchronize());

    cuda_log(cublasSgemvBatched(handle, CUBLAS_OP_N, 2, 2, &alpha, batch_ptrs_d_R_inv, 2, batch_ptrs_new_coords, 1, &beta, batch_ptrs_src_coords, 1, batch_size));

    kernel_rotate_setPixels<<<blocknum, blocksize>>>(d_img, d_dest_img, batch_ptrs_src_coords, half_width, half_height);
    cuda_log(cudaDeviceSynchronize());
    cuda_log(cudaGetLastError());

    cuda_log(cudaFree(d_src_arr));
    cuda_log(cudaFree(d_img));
    cuda_log(cudaFree(d_dest_img));

    cuda_log(cudaFree(d_R));
    cuda_log(cudaFree(d_R_inv));
    cuda_log(cudaFree(d_rotated_corners));
    cuda_log(cudaFree(d_corners));

    cuda_log(cudaFree(batch_new_coords));
    cuda_log(cudaFree(batch_src_coords));

    cuda_log(cudaFree(batch_ptrs_d_R_inv));
    cuda_log(cudaFree(batch_ptrs_new_coords));
    cuda_log(cudaFree(batch_ptrs_src_coords));

    cublasDestroy(handle);
}
