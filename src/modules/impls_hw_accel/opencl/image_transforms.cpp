#include "image_transforms.h"
#include <CL/opencl.hpp>
#include <iostream>
#include <vector>
#include <cmath>
#include <algorithm>
#include <string.h>
#include "impls_hw_accel/opencl/image_codec_cl.h"

// OpenCL helper functions
bool initialize_opencl(cl_context& context, cl_command_queue& queue, cl_program& program, 
                     cl_kernel& crop_kernel, cl_kernel& rotate_kernel, 
                     cl_kernel& reflect_kernel, cl_kernel& shear_kernel);
void cleanup_opencl(cl_context& context, cl_command_queue& queue, cl_program& program, 
                   cl_kernel& crop_kernel, cl_kernel& rotate_kernel, 
                   cl_kernel& reflect_kernel, cl_kernel& shear_kernel,
                   cl_mem& input_buffer, cl_mem& output_buffer);
bool create_buffers(cl_context& context, cl_mem& input_buffer, cl_mem& output_buffer, size_t size);

// OpenCL kernel sources
const char* rotate_kernel_source = R"(
__kernel void rotate_image(__global unsigned char* input,
                         __global unsigned char* output,
                         const int in_width,
                         const int in_height,
                         const int out_width,
                         const int out_height,
                         const int components,
                         const float cos_theta,
                         const float sin_theta)
{
    int x = get_global_id(0);
    int y = get_global_id(1);
    
    if (x >= out_width || y >= out_height)
        return;
    
    // Convert output coordinates to centered coordinates
    float new_x = x - out_width / 2.0f;
    float new_y = y - out_height / 2.0f;
    
    // Apply inverse rotation to find source pixel
    float src_x = cos_theta * new_x + sin_theta * new_y + in_width / 2.0f;
    float src_y = -sin_theta * new_x + cos_theta * new_y + in_height / 2.0f;
    
    // Check if source coordinates are within bounds
    if (src_x >= 0 && src_x < in_width && src_y >= 0 && src_y < in_height) {
        // Get integer and fractional parts for bilinear interpolation
        int x0 = (int)src_x;
        int y0 = (int)src_y;
        int x1 = min(x0 + 1, in_width - 1);
        int y1 = min(y0 + 1, in_height - 1);
        
        float dx = src_x - x0;
        float dy = src_y - y0;
        
        float w00 = (1-dx) * (1-dy);
        float w10 = dx * (1-dy);
        float w01 = (1-dx) * dy;
        float w11 = dx * dy;
        
        // Calculate output pixel index
        int out_idx = (y * out_width + x) * components;
        
        // For each color component
        for (int c = 0; c < components; c++) {
            // Get four surrounding pixels
            unsigned char p00 = input[(y0 * in_width + x0) * components + c];
            unsigned char p10 = input[(y0 * in_width + x1) * components + c];
            unsigned char p01 = input[(y1 * in_width + x0) * components + c];
            unsigned char p11 = input[(y1 * in_width + x1) * components + c];
            
            // Bilinear interpolation
            float val = w00 * p00 + w10 * p10 + w01 * p01 + w11 * p11;
            output[out_idx + c] = (unsigned char)(val + 0.5f);
        }
    } else {
        // Out of bounds - set to white
        int out_idx = (y * out_width + x) * components;
        for (int c = 0; c < components; c++) {
            output[out_idx + c] = 255;
        }
    }
}
)";

const char* reflect_kernel_source = R"(
__kernel void reflect_image(__global unsigned char* input,
                          __global unsigned char* output,
                          const int width,
                          const int height,
                          const int components,
                          const int horizontal,
                          const int vertical)
{
    int x = get_global_id(0);
    int y = get_global_id(1);
    
    if (x >= width || y >= height)
        return;
    
    // Calculate source coordinates based on reflection flags
    int src_x = vertical ? (width - 1 - x) : x;
    int src_y = horizontal ? (height - 1 - y) : y;
    
    // Calculate input and output indices
    int in_idx = (src_y * width + src_x) * components;
    int out_idx = (y * width + x) * components;
    
    // Copy all components
    for (int c = 0; c < components; c++) {
        output[out_idx + c] = input[in_idx + c];
    }
}
)";

const char* shear_kernel_source = R"(
__kernel void shear_image(__global unsigned char* input,
                        __global unsigned char* output,
                        const int in_width,
                        const int in_height,
                        const int out_width,
                        const int out_height,
                        const int components,
                        const float shx,
                        const float shy,
                        const float offset_x,
                        const float offset_y)
{
    int x = get_global_id(0);
    int y = get_global_id(1);
    
    if (x >= out_width || y >= out_height)
        return;
    
    // Apply the shear transformation (reverse mapping)
    float src_x = (x - offset_x) - shy * (y - offset_y);
    float src_y = (y - offset_y) - shx * (x - offset_x);
    
    // Check if source coordinates are within bounds
    if (src_x >= 0 && src_x < in_width && src_y >= 0 && src_y < in_height) {
        // Get integer and fractional parts for bilinear interpolation
        int x0 = (int)src_x;
        int y0 = (int)src_y;
        int x1 = min(x0 + 1, in_width - 1);
        int y1 = min(y0 + 1, in_height - 1);
        
        float dx = src_x - x0;
        float dy = src_y - y0;
        
        float w00 = (1-dx) * (1-dy);
        float w10 = dx * (1-dy);
        float w01 = (1-dx) * dy;
        float w11 = dx * dy;
        
        // Calculate output pixel index
        int out_idx = (y * out_width + x) * components;
        
        // For each color component
        for (int c = 0; c < components; c++) {
            // Get four surrounding pixels
            unsigned char p00 = input[(y0 * in_width + x0) * components + c];
            unsigned char p10 = input[(y0 * in_width + x1) * components + c];
            unsigned char p01 = input[(y1 * in_width + x0) * components + c];
            unsigned char p11 = input[(y1 * in_width + x1) * components + c];
            
            // Bilinear interpolation
            float val = w00 * p00 + w10 * p10 + w01 * p01 + w11 * p11;
            output[out_idx + c] = (unsigned char)(val + 0.5f);
        }
    } else {
        // Out of bounds - set to white
        int out_idx = (y * out_width + x) * components;
        for (int c = 0; c < components; c++) {
            output[out_idx + c] = 255;
        }
    }
}
)";

// OpenCL resources
static cl_context context = nullptr;
static cl_command_queue queue = nullptr;
static cl_program program = nullptr;
static cl_kernel crop_kernel = nullptr;
static cl_kernel rotate_kernel = nullptr;
static cl_kernel reflect_kernel = nullptr;
static cl_kernel shear_kernel = nullptr;
static cl_mem input_buffer = nullptr;
static cl_mem output_buffer = nullptr;
static bool opencl_initialized = false;

namespace opencl_impl {

// Forward declarations
void reflect_gpu(matrix& img, bool horizontal, bool vertical);
void shear_gpu(matrix& img, float shx, float shy);
void rotate_gpu(matrix& img, float angle);

void crop_gpu(matrix& img, unsigned crop_left, unsigned crop_top, unsigned crop_right, unsigned crop_bottom) {
    std::cout << "[OpenCL Transforms] Cropping image using CPU implementation" << std::endl;

    // Calculate new dimensions
    unsigned new_width = img.width - crop_left - crop_right;
    unsigned new_height = img.height - crop_top - crop_bottom;

    // Validate crop parameters
    if (new_width <= 0 || new_height <= 0) {
        std::cerr << "[OpenCL Transforms] Invalid crop dimensions" << std::endl;
        return;
    }

    if (crop_left + crop_right > img.width || crop_top + crop_bottom > img.height) {
        std::cerr << "[OpenCL Transforms] Crop exceeds image dimensions" << std::endl;
        return;
    }

    // CPU implementation
    unsigned char* newArr = new unsigned char[new_width * new_height * img.components_num];
    unsigned char* src = img.arr + (crop_top * img.width + crop_left) * img.components_num;
    unsigned char* dst = newArr;

    for (unsigned i = 0; i < new_height; ++i) {
        memcpy(dst, src, new_width * img.components_num);
        src += img.width * img.components_num;
        dst += new_width * img.components_num;
    }

    delete[] img.arr;
    img.set_arr_interlaced(newArr, new_width, new_height);
    
    std::cout << "[OpenCL Transforms] Crop complete, new dimensions: "
            << img.width << "x" << img.height << std::endl;
}

void rotate_gpu(matrix& img, float angle) {
    std::cout << "[OpenCL Transforms] Rotating image" << std::endl;
    
    // Calculate rotation parameters
    float radians = angle * 3.14159265358979323846f / 180.0f;
    float cos_theta = std::cos(radians);
    float sin_theta = std::sin(radians);
    
    float half_height = img.height * 0.5f;
    float half_width = img.width * 0.5f;
    
    // Calculate bounds
    float corners[4][2] = {
        {-half_width, -half_height},
        { half_width, -half_height},
        {-half_width,  half_height},
        { half_width,  half_height}
    };
    
    float rotated_corners[8];
    for (int i = 0; i < 4; ++i) {
        float x = corners[i][0];
        float y = corners[i][1];
        rotated_corners[i*2] = cos_theta * x - sin_theta * y;
        rotated_corners[i*2+1] = sin_theta * x + cos_theta * y;
    }
    
    float min_x = rotated_corners[0], max_x = rotated_corners[0];
    float min_y = rotated_corners[1], max_y = rotated_corners[1];
    
    for (int i = 1; i < 4; ++i) {
        min_x = std::min(min_x, rotated_corners[i*2]);
        max_x = std::max(max_x, rotated_corners[i*2]);
        min_y = std::min(min_y, rotated_corners[i*2+1]);
        max_y = std::max(max_y, rotated_corners[i*2+1]);
    }
    
    unsigned new_width = static_cast<unsigned>(std::round(max_x - min_x));
    unsigned new_height = static_cast<unsigned>(std::round(max_y - min_y));
    
    // Try to use OpenCL
    if (!opencl_initialized) {
        if (!initialize_opencl(context, queue, program, crop_kernel, rotate_kernel, reflect_kernel, shear_kernel)) {
            std::cout << "[OpenCL Transforms] Failed to initialize OpenCL, falling back to CPU" << std::endl;
            
            // CPU fallback implementation
            unsigned char* newArr = new unsigned char[new_width * new_height * img.components_num];
            std::fill(newArr, newArr + new_width * new_height * img.components_num, 255);
            
            float new_half_width = new_width * 0.5f;
            float new_half_height = new_height * 0.5f;
            
            for (unsigned y = 0; y < new_height; ++y) {
                for (unsigned x = 0; x < new_width; ++x) {
                    float coords[2] = {
                        x - new_half_width,
                        y - new_half_height
                    };
                    
                    // Inverse transform
                    float src_x = cos_theta * coords[0] + sin_theta * coords[1] + half_width;
                    float src_y = -sin_theta * coords[0] + cos_theta * coords[1] + half_height;
                    
                    if (src_x >= 0 && src_x < img.width && 
                        src_y >= 0 && src_y < img.height) {
                        unsigned char* pixel = &newArr[(y * new_width + x) * img.components_num];
                        
                        // Bilinear interpolation
                        const int x0 = static_cast<int>(src_x);
                        const int y0 = static_cast<int>(src_y);
                        const int x1 = std::min(x0 + 1, static_cast<int>(img.width) - 1);
                        const int y1 = std::min(y0 + 1, static_cast<int>(img.height) - 1);
                        
                        float dx = src_x - x0;
                        float dy = src_y - y0;
                        float w00 = (1-dx)*(1-dy);
                        float w10 = dx*(1-dy);
                        float w01 = (1-dx)*dy;
                        float w11 = dx*dy;
                        
                        for (unsigned c = 0; c < img.components_num; ++c) {
                            float v00 = img.get(x0, y0)[c];
                            float v10 = img.get(x1, y0)[c];
                            float v01 = img.get(x0, y1)[c];
                            float v11 = img.get(x1, y1)[c];
                            pixel[c] = static_cast<unsigned char>(v00*w00 + v10*w10 + v01*w01 + v11*w11);
                        }
                    }
                }
            }
            
            delete[] img.arr;
            img.set_arr_interlaced(newArr, new_width, new_height);
            
            std::cout << "[OpenCL Transforms] Rotation complete, new dimensions: "
                    << img.width << "x" << img.height << std::endl;
            return;
        }
    }
    
    // Allocate memory for the new image
    unsigned char* newArr = new unsigned char[new_width * new_height * img.components_num];
    
    // Create buffers if needed
    size_t input_size = img.width * img.height * img.components_num;
    size_t output_size = new_width * new_height * img.components_num;
    
    if (!create_buffers(context, input_buffer, output_buffer, std::max(input_size, output_size))) {
        std::cout << "[OpenCL Transforms] Failed to create buffers, falling back to CPU" << std::endl;
        cleanup_opencl(context, queue, program, crop_kernel, rotate_kernel, reflect_kernel, shear_kernel, input_buffer, output_buffer);
        opencl_initialized = false;
        
        // Call ourselves to use CPU fallback
        rotate_gpu(img, angle);
        return;
    }
    
    // Copy input data to device
    cl_int err = clEnqueueWriteBuffer(queue, input_buffer, CL_TRUE, 0, input_size, img.arr, 0, nullptr, nullptr);
    if (err != CL_SUCCESS) {
        std::cout << "[OpenCL Transforms] Failed to write input data, falling back to CPU" << std::endl;
        cleanup_opencl(context, queue, program, crop_kernel, rotate_kernel, reflect_kernel, shear_kernel, input_buffer, output_buffer);
        opencl_initialized = false;
        
        rotate_gpu(img, angle);
        return;
    }
    
    // Set kernel arguments
    int in_width = img.width;
    int in_height = img.height;
    int out_width = new_width;
    int out_height = new_height;
    int components = img.components_num;
    
    err = clSetKernelArg(rotate_kernel, 0, sizeof(cl_mem), &input_buffer);
    err |= clSetKernelArg(rotate_kernel, 1, sizeof(cl_mem), &output_buffer);
    err |= clSetKernelArg(rotate_kernel, 2, sizeof(int), &in_width);
    err |= clSetKernelArg(rotate_kernel, 3, sizeof(int), &in_height);
    err |= clSetKernelArg(rotate_kernel, 4, sizeof(int), &out_width);
    err |= clSetKernelArg(rotate_kernel, 5, sizeof(int), &out_height);
    err |= clSetKernelArg(rotate_kernel, 6, sizeof(int), &components);
    err |= clSetKernelArg(rotate_kernel, 7, sizeof(float), &cos_theta);
    err |= clSetKernelArg(rotate_kernel, 8, sizeof(float), &sin_theta);
    
    if (err != CL_SUCCESS) {
        std::cout << "[OpenCL Transforms] Failed to set kernel arguments, falling back to CPU" << std::endl;
        cleanup_opencl(context, queue, program, crop_kernel, rotate_kernel, reflect_kernel, shear_kernel, input_buffer, output_buffer);
        opencl_initialized = false;
        
        rotate_gpu(img, angle);
        return;
    }
    
    // Execute the kernel
    size_t global_work_size[2] = {new_width, new_height};
    err = clEnqueueNDRangeKernel(queue, rotate_kernel, 2, nullptr, global_work_size, nullptr, 0, nullptr, nullptr);
    
    if (err != CL_SUCCESS) {
        std::cout << "[OpenCL Transforms] Failed to execute kernel, falling back to CPU" << std::endl;
        cleanup_opencl(context, queue, program, crop_kernel, rotate_kernel, reflect_kernel, shear_kernel, input_buffer, output_buffer);
        opencl_initialized = false;
        
        rotate_gpu(img, angle);
        return;
    }
    
    // Read back the result
    err = clEnqueueReadBuffer(queue, output_buffer, CL_TRUE, 0, output_size, newArr, 0, nullptr, nullptr);
    
    if (err != CL_SUCCESS) {
        std::cout << "[OpenCL Transforms] Failed to read output data, falling back to CPU" << std::endl;
        delete[] newArr;
        cleanup_opencl(context, queue, program, crop_kernel, rotate_kernel, reflect_kernel, shear_kernel, input_buffer, output_buffer);
        opencl_initialized = false;
        
        rotate_gpu(img, angle);
        return;
    }
    
    // Update image with new data
    delete[] img.arr;
    img.set_arr_interlaced(newArr, new_width, new_height);
    
    std::cout << "[OpenCL Transforms] Rotation complete using OpenCL, new dimensions: "
            << img.width << "x" << img.height << std::endl;
}

void reflect_gpu(matrix& img, bool horizontal, bool vertical) {
    std::cout << "[OpenCL Transforms] Reflecting image" << std::endl;
    
    if (!horizontal && !vertical) {
        std::cout << "[OpenCL Transforms] No reflection specified, image unchanged" << std::endl;
        return;
    }
    
    // Try to use OpenCL
    if (!opencl_initialized) {
        if (!initialize_opencl(context, queue, program, crop_kernel, rotate_kernel, reflect_kernel, shear_kernel)) {
            std::cout << "[OpenCL Transforms] Failed to initialize OpenCL, falling back to CPU" << std::endl;
            
            // CPU fallback implementation
            unsigned char* newArr = new unsigned char[img.width * img.height * img.components_num];
            
            for (unsigned y = 0; y < img.height; ++y) {
                for (unsigned x = 0; x < img.width; ++x) {
                    unsigned target_x = vertical ? (img.width - 1 - x) : x;
                    unsigned target_y = horizontal ? (img.height - 1 - y) : y;
                    
                    unsigned char* src = img.get(x, y);
                    unsigned char* dst = &newArr[(target_y * img.width + target_x) * img.components_num];
                    memcpy(dst, src, img.components_num);
                }
            }
            
            delete[] img.arr;
            img.arr = newArr;
            
            std::cout << "[OpenCL Transforms] Reflection complete using CPU" << std::endl;
            return;
        }
    }
    
    // Allocate memory for the new image
    unsigned char* newArr = new unsigned char[img.width * img.height * img.components_num];
    
    // Create buffers if needed
    size_t buffer_size = img.width * img.height * img.components_num;
    
    if (!create_buffers(context, input_buffer, output_buffer, buffer_size)) {
        std::cout << "[OpenCL Transforms] Failed to create buffers, falling back to CPU" << std::endl;
        cleanup_opencl(context, queue, program, crop_kernel, rotate_kernel, reflect_kernel, shear_kernel, input_buffer, output_buffer);
        opencl_initialized = false;
        
        // Call ourselves to use CPU fallback
        reflect_gpu(img, horizontal, vertical);
        return;
    }
    
    // Copy input data to device
    cl_int err = clEnqueueWriteBuffer(queue, input_buffer, CL_TRUE, 0, buffer_size, img.arr, 0, nullptr, nullptr);
    if (err != CL_SUCCESS) {
        std::cout << "[OpenCL Transforms] Failed to write input data, falling back to CPU" << std::endl;
        delete[] newArr;
        cleanup_opencl(context, queue, program, crop_kernel, rotate_kernel, reflect_kernel, shear_kernel, input_buffer, output_buffer);
        opencl_initialized = false;
        
        reflect_gpu(img, horizontal, vertical);
        return;
    }
    
    // Set kernel arguments
    int width = img.width;
    int height = img.height;
    int components = img.components_num;
    int h_flag = horizontal ? 1 : 0;
    int v_flag = vertical ? 1 : 0;
    
    err = clSetKernelArg(reflect_kernel, 0, sizeof(cl_mem), &input_buffer);
    err |= clSetKernelArg(reflect_kernel, 1, sizeof(cl_mem), &output_buffer);
    err |= clSetKernelArg(reflect_kernel, 2, sizeof(int), &width);
    err |= clSetKernelArg(reflect_kernel, 3, sizeof(int), &height);
    err |= clSetKernelArg(reflect_kernel, 4, sizeof(int), &components);
    err |= clSetKernelArg(reflect_kernel, 5, sizeof(int), &h_flag);
    err |= clSetKernelArg(reflect_kernel, 6, sizeof(int), &v_flag);
    
    if (err != CL_SUCCESS) {
        std::cout << "[OpenCL Transforms] Failed to set kernel arguments, falling back to CPU" << std::endl;
        delete[] newArr;
        cleanup_opencl(context, queue, program, crop_kernel, rotate_kernel, reflect_kernel, shear_kernel, input_buffer, output_buffer);
        opencl_initialized = false;
        
        reflect_gpu(img, horizontal, vertical);
        return;
    }
    
    // Execute the kernel
    size_t global_work_size[2] = {img.width, img.height};
    err = clEnqueueNDRangeKernel(queue, reflect_kernel, 2, nullptr, global_work_size, nullptr, 0, nullptr, nullptr);
    
    if (err != CL_SUCCESS) {
        std::cout << "[OpenCL Transforms] Failed to execute kernel, falling back to CPU" << std::endl;
        delete[] newArr;
        cleanup_opencl(context, queue, program, crop_kernel, rotate_kernel, reflect_kernel, shear_kernel, input_buffer, output_buffer);
        opencl_initialized = false;
        
        reflect_gpu(img, horizontal, vertical);
        return;
    }
    
    // Read back the result
    err = clEnqueueReadBuffer(queue, output_buffer, CL_TRUE, 0, buffer_size, newArr, 0, nullptr, nullptr);
    
    if (err != CL_SUCCESS) {
        std::cout << "[OpenCL Transforms] Failed to read output data, falling back to CPU" << std::endl;
        delete[] newArr;
        cleanup_opencl(context, queue, program, crop_kernel, rotate_kernel, reflect_kernel, shear_kernel, input_buffer, output_buffer);
        opencl_initialized = false;
        
        reflect_gpu(img, horizontal, vertical);
        return;
    }
    
    // Update image with new data
    delete[] img.arr;
    img.arr = newArr;
    
    std::cout << "[OpenCL Transforms] Reflection complete using OpenCL" << std::endl;
}

void shear_gpu(matrix& img, float shx, float shy) {
    std::cout << "[OpenCL Transforms] Shearing image" << std::endl;
    
    if (shx == 0.0f && shy == 0.0f) {
        std::cout << "[OpenCL Transforms] No shearing applied, image unchanged" << std::endl;
        return;
    }
    
    // Calculate new dimensions
    unsigned new_width = static_cast<unsigned>(img.width + std::abs(shy)*img.height);
    unsigned new_height = static_cast<unsigned>(img.height + std::abs(shx)*img.width);
    
    // Calculate offsets for different shear directions
    float offset_x = (shy > 0) ? 0 : std::abs(shy)*img.height;
    float offset_y = (shx > 0) ? 0 : std::abs(shx)*img.width;
    
    // Try to use OpenCL
    if (!opencl_initialized) {
        if (!initialize_opencl(context, queue, program, crop_kernel, rotate_kernel, reflect_kernel, shear_kernel)) {
            std::cout << "[OpenCL Transforms] Failed to initialize OpenCL, falling back to CPU" << std::endl;
            
            // CPU fallback implementation
            unsigned char* newArr = new unsigned char[new_width * new_height * img.components_num]();
            std::fill(newArr, newArr + new_width*new_height*img.components_num, 255);
            
            for (unsigned y = 0; y < new_height; ++y) {
                for (unsigned x = 0; x < new_width; ++x) {
                    // Proper inverse coordinate transformation
                    float src_x = (x - offset_x) - shy*(y - offset_y);
                    float src_y = (y - offset_y) - shx*(x - offset_x);
                    
                    if (src_x >= 0 && src_x < img.width && src_y >= 0 && src_y < img.height) {
                        // Bilinear interpolation
                        const int x0 = static_cast<int>(src_x);
                        const int y0 = static_cast<int>(src_y);
                        const int x1 = std::min(x0 + 1, static_cast<int>(img.width) - 1);
                        const int y1 = std::min(y0 + 1, static_cast<int>(img.height) - 1);
                        
                        float dx = src_x - x0;
                        float dy = src_y - y0;
                        float w00 = (1-dx)*(1-dy);
                        float w10 = dx*(1-dy);
                        float w01 = (1-dx)*dy;
                        float w11 = dx*dy;
                        
                        unsigned char* dst = &newArr[(y*new_width + x)*img.components_num];
                        for (unsigned c = 0; c < img.components_num; ++c) {
                            float v00 = img.get(x0, y0)[c];
                            float v10 = img.get(x1, y0)[c];
                            float v01 = img.get(x0, y1)[c];
                            float v11 = img.get(x1, y1)[c];
                            dst[c] = static_cast<unsigned char>(v00*w00 + v10*w10 + v01*w01 + v11*w11);
                        }
                    }
                }
            }
            
            delete[] img.arr;
            img.set_arr_interlaced(newArr, new_width, new_height);
            
            std::cout << "[OpenCL Transforms] Shearing complete using CPU, new dimensions: "
                    << new_width << "x" << new_height << std::endl;
            return;
        }
    }
    
    // Allocate memory for the new image
    unsigned char* newArr = new unsigned char[new_width * new_height * img.components_num];
    std::fill(newArr, newArr + new_width * new_height * img.components_num, 255);
    // Create buffers if needed
    size_t input_size = img.width * img.height * img.components_num;
    size_t output_size = new_width * new_height * img.components_num;
    
    if (!create_buffers(context, input_buffer, output_buffer, std::max(input_size, output_size))) {
        std::cout << "[OpenCL Transforms] Failed to create buffers, falling back to CPU" << std::endl;
        delete[] newArr;
        cleanup_opencl(context, queue, program, crop_kernel, rotate_kernel, reflect_kernel, shear_kernel, input_buffer, output_buffer);
        opencl_initialized = false;
        
        // Call ourselves to use CPU fallback
        shear_gpu(img, shx, shy);
        return;
    }
    
    // Copy input data to device
    cl_int err = clEnqueueWriteBuffer(queue, input_buffer, CL_TRUE, 0, input_size, img.arr, 0, nullptr, nullptr);
    if (err != CL_SUCCESS) {
        std::cout << "[OpenCL Transforms] Failed to write input data, falling back to CPU" << std::endl;
        delete[] newArr;
        cleanup_opencl(context, queue, program, crop_kernel, rotate_kernel, reflect_kernel, shear_kernel, input_buffer, output_buffer);
        opencl_initialized = false;
        
        shear_gpu(img, shx, shy);
        return;
    }
    
    // Set kernel arguments
    int in_width = img.width;
    int in_height = img.height;
    int out_width = new_width;
    int out_height = new_height;
    int components = img.components_num;
    
    err = clSetKernelArg(shear_kernel, 0, sizeof(cl_mem), &input_buffer);
    err |= clSetKernelArg(shear_kernel, 1, sizeof(cl_mem), &output_buffer);
    err |= clSetKernelArg(shear_kernel, 2, sizeof(int), &in_width);
    err |= clSetKernelArg(shear_kernel, 3, sizeof(int), &in_height);
    err |= clSetKernelArg(shear_kernel, 4, sizeof(int), &out_width);
    err |= clSetKernelArg(shear_kernel, 5, sizeof(int), &out_height);
    err |= clSetKernelArg(shear_kernel, 6, sizeof(int), &components);
    err |= clSetKernelArg(shear_kernel, 7, sizeof(float), &shx);
    err |= clSetKernelArg(shear_kernel, 8, sizeof(float), &shy);
    err |= clSetKernelArg(shear_kernel, 9, sizeof(float), &offset_x);
    err |= clSetKernelArg(shear_kernel, 10, sizeof(float), &offset_y);
    
    if (err != CL_SUCCESS) {
        std::cout << "[OpenCL Transforms] Failed to set kernel arguments, falling back to CPU" << std::endl;
        delete[] newArr;
        cleanup_opencl(context, queue, program, crop_kernel, rotate_kernel, reflect_kernel, shear_kernel, input_buffer, output_buffer);
        opencl_initialized = false;
        
        shear_gpu(img, shx, shy);
        return;
    }
    
    // Execute the kernel
    size_t global_work_size[2] = {new_width, new_height};
    err = clEnqueueNDRangeKernel(queue, shear_kernel, 2, nullptr, global_work_size, nullptr, 0, nullptr, nullptr);
    
    if (err != CL_SUCCESS) {
        std::cout << "[OpenCL Transforms] Failed to execute kernel, falling back to CPU" << std::endl;
        delete[] newArr;
        cleanup_opencl(context, queue, program, crop_kernel, rotate_kernel, reflect_kernel, shear_kernel, input_buffer, output_buffer);
        opencl_initialized = false;
        
        shear_gpu(img, shx, shy);
        return;
    }
    
    // Read back the result
    err = clEnqueueReadBuffer(queue, output_buffer, CL_TRUE, 0, output_size, newArr, 0, nullptr, nullptr);
    
    if (err != CL_SUCCESS) {
        std::cout << "[OpenCL Transforms] Failed to read output data, falling back to CPU" << std::endl;
        delete[] newArr;
        cleanup_opencl(context, queue, program, crop_kernel, rotate_kernel, reflect_kernel, shear_kernel, input_buffer, output_buffer);
        opencl_initialized = false;
        
        shear_gpu(img, shx, shy);
        return;
    }
    
    // Update image with new data
    delete[] img.arr;
    img.set_arr_interlaced(newArr, new_width, new_height);
    
    std::cout << "[OpenCL Transforms] Shearing complete using OpenCL, new dimensions: "
            << new_width << "x" << new_height << std::endl;
}

} // namespace opencl_impl

// OpenCL helper functions implementation
bool initialize_opencl(cl_context& context, cl_command_queue& queue, cl_program& program, 
                     cl_kernel& crop_kernel, cl_kernel& rotate_kernel,
                     cl_kernel& reflect_kernel, cl_kernel& shear_kernel) {
    if (opencl_initialized) return true;

    cl_int err;
    cl_uint num_platforms;
    cl_platform_id platform_id;
    cl_device_id device_id;
    
    // Get platform
    err = clGetPlatformIDs(1, &platform_id, &num_platforms);
    if (err != CL_SUCCESS || num_platforms == 0) {
        std::cerr << "Failed to find OpenCL platforms" << std::endl;
        return false;
    }
    
    // Get device (prefer GPU)
    err = clGetDeviceIDs(platform_id, CL_DEVICE_TYPE_GPU, 1, &device_id, nullptr);
    if (err != CL_SUCCESS) {
        // Try to get CPU device if GPU is not available
        err = clGetDeviceIDs(platform_id, CL_DEVICE_TYPE_CPU, 1, &device_id, nullptr);
        if (err != CL_SUCCESS) {
            std::cerr << "Failed to find OpenCL devices" << std::endl;
            return false;
        }
        std::cout << "[OpenCL Transforms] Using CPU device" << std::endl;
    } else {
        std::cout << "[OpenCL Transforms] Using GPU device" << std::endl;
    }
    
    // Create context
    context = clCreateContext(nullptr, 1, &device_id, nullptr, nullptr, &err);
    if (err != CL_SUCCESS) {
        std::cerr << "Failed to create OpenCL context" << std::endl;
        return false;
    }
    
    // Create command queue
    queue = clCreateCommandQueue(context, device_id, 0, &err);
    if (err != CL_SUCCESS) {
        std::cerr << "Failed to create OpenCL command queue" << std::endl;
        clReleaseContext(context);
        return false;
    }
    
    // Create program from source
    std::string combined_source = std::string(rotate_kernel_source) + reflect_kernel_source + shear_kernel_source;
    const char* source = combined_source.c_str();
    size_t source_size = combined_source.size();
    
    program = clCreateProgramWithSource(context, 1, &source, &source_size, &err);
    if (err != CL_SUCCESS) {
        std::cerr << "Failed to create OpenCL program" << std::endl;
        clReleaseCommandQueue(queue);
        clReleaseContext(context);
        return false;
    }
    
    // Build program
    err = clBuildProgram(program, 1, &device_id, nullptr, nullptr, nullptr);
    if (err != CL_SUCCESS) {
        size_t log_size;
        clGetProgramBuildInfo(program, device_id, CL_PROGRAM_BUILD_LOG, 0, nullptr, &log_size);
        char* log = new char[log_size];
        clGetProgramBuildInfo(program, device_id, CL_PROGRAM_BUILD_LOG, log_size, log, nullptr);
        std::cerr << "OpenCL program build failed: " << log << std::endl;
        delete[] log;
        
        clReleaseProgram(program);
        clReleaseCommandQueue(queue);
        clReleaseContext(context);
        return false;
    }
    
    // Create kernels
    rotate_kernel = clCreateKernel(program, "rotate_image", &err);
    if (err != CL_SUCCESS) {
        std::cerr << "Failed to create rotate kernel" << std::endl;
        clReleaseProgram(program);
        clReleaseCommandQueue(queue);
        clReleaseContext(context);
        return false;
    }
    
    reflect_kernel = clCreateKernel(program, "reflect_image", &err);
    if (err != CL_SUCCESS) {
        std::cerr << "Failed to create reflect kernel" << std::endl;
        clReleaseKernel(rotate_kernel);
        clReleaseProgram(program);
        clReleaseCommandQueue(queue);
        clReleaseContext(context);
        return false;
    }
    
    shear_kernel = clCreateKernel(program, "shear_image", &err);
    if (err != CL_SUCCESS) {
        std::cerr << "Failed to create shear kernel" << std::endl;
        clReleaseKernel(rotate_kernel);
        clReleaseKernel(reflect_kernel);
        clReleaseProgram(program);
        clReleaseCommandQueue(queue);
        clReleaseContext(context);
        return false;
    }
    
    opencl_initialized = true;
    return true;
}

void cleanup_opencl(cl_context& context, cl_command_queue& queue, cl_program& program, 
                   cl_kernel& crop_kernel, cl_kernel& rotate_kernel, 
                   cl_kernel& reflect_kernel, cl_kernel& shear_kernel,
                   cl_mem& input_buffer, cl_mem& output_buffer) {
    if (input_buffer) clReleaseMemObject(input_buffer);
    if (output_buffer) clReleaseMemObject(output_buffer);
    if (crop_kernel) clReleaseKernel(crop_kernel);
    if (rotate_kernel) clReleaseKernel(rotate_kernel);
    if (reflect_kernel) clReleaseKernel(reflect_kernel);
    if (shear_kernel) clReleaseKernel(shear_kernel);
    if (program) clReleaseProgram(program);
    if (queue) clReleaseCommandQueue(queue);
    if (context) clReleaseContext(context);
    
    input_buffer = nullptr;
    output_buffer = nullptr;
    crop_kernel = nullptr;
    rotate_kernel = nullptr;
    reflect_kernel = nullptr;
    shear_kernel = nullptr;
    program = nullptr;
    queue = nullptr;
    context = nullptr;
    
    opencl_initialized = false;
}

bool create_buffers(cl_context& context, cl_mem& input_buffer, cl_mem& output_buffer, size_t size) {
    cl_int err;
    
    // Release old buffers if they exist
    if (input_buffer) {
        clReleaseMemObject(input_buffer);
        input_buffer = nullptr;
    }
    
    if (output_buffer) {
        clReleaseMemObject(output_buffer);
        output_buffer = nullptr;
    }
    
    // Create input buffer
    input_buffer = clCreateBuffer(context, CL_MEM_READ_ONLY, size, nullptr, &err);
    if (err != CL_SUCCESS) {
        std::cerr << "Failed to create input buffer" << std::endl;
        return false;
    }
    
    // Create output buffer
    output_buffer = clCreateBuffer(context, CL_MEM_WRITE_ONLY, size, nullptr, &err);
    if (err != CL_SUCCESS) {
        std::cerr << "Failed to create output buffer" << std::endl;
        clReleaseMemObject(input_buffer);
        input_buffer = nullptr;
        return false;
    }
    
    return true;
}
