#include "image_transforms.h"
#include <CL/cl.h>
#include <iostream>
#include <vector>
#include <cmath>
#include <algorithm>
#include <string.h>
#include "impls_hw_accel/opencl/image_codec_cl.h"

// OpenCL helper functions
bool initialize_opencl(cl_context& context, cl_command_queue& queue, cl_program& program, cl_kernel& crop_kernel, cl_kernel& rotate_kernel);
void cleanup_opencl(cl_context& context, cl_command_queue& queue, cl_program& program, cl_kernel& crop_kernel, cl_kernel& rotate_kernel, cl_mem& input_buffer, cl_mem& output_buffer);
bool create_buffers(cl_context& context, cl_mem& input_buffer, cl_mem& output_buffer, size_t size);

// OpenCL resources
static cl_context context = nullptr;
static cl_command_queue queue = nullptr;
static cl_program program = nullptr;
static cl_kernel crop_kernel = nullptr;
static cl_kernel rotate_kernel = nullptr;
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
    std::cout << "[OpenCL Transforms] Rotating image using CPU implementation" << std::endl;
    
    // CPU implementation for arbitrary angle
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
}

void reflect_gpu(matrix& img, bool horizontal, bool vertical) {
    std::cout << "[OpenCL Transforms] Reflecting image using CPU implementation" << std::endl;
    
    if (!horizontal && !vertical) {
        std::cout << "[OpenCL Transforms] No reflection specified, image unchanged" << std::endl;
        return;
    }
    
    // CPU implementation
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
    
    std::cout << "[OpenCL Transforms] Reflection complete" << std::endl;
}

void shear_gpu(matrix& img, float shx, float shy) {
    std::cout << "[OpenCL Transforms] Shearing image using CPU implementation" << std::endl;
    
    if (shx == 0.0f && shy == 0.0f) {
        std::cout << "[OpenCL Transforms] No shearing applied, image unchanged" << std::endl;
        return;
    }
    
    // Calculate new dimensions
    unsigned new_width = static_cast<unsigned>(img.width + 2*std::abs(shy)*img.height);
    unsigned new_height = static_cast<unsigned>(img.height + 2*std::abs(shx)*img.width);
    
    // CPU implementation
    unsigned char* newArr = new unsigned char[new_width * new_height * img.components_num];
    std::fill(newArr, newArr + new_width*new_height*img.components_num, 255);
    
    float center_x = new_width / 2.0f;
    float center_y = new_height / 2.0f;
    float img_center_x = img.width / 2.0f;
    float img_center_y = img.height / 2.0f;
    
    for (unsigned y = 0; y < new_height; ++y) {
        for (unsigned x = 0; x < new_width; ++x) {
            float src_x = img_center_x + (x - center_x) - shx*(y - center_y);
            float src_y = img_center_y + (y - center_y) - shy*(x - center_x);
            
            if (src_x >= 0 && src_x < img.width && src_y >= 0 && src_y < img.height) {
                unsigned char* pixel = &newArr[(y*new_width + x)*img.components_num];
                
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
    
    std::cout << "[OpenCL Transforms] Shearing complete, new dimensions: "
            << new_width << "x" << new_height << std::endl;
}

} // namespace opencl_impl

// OpenCL implementation functions are accessed through namespace opencl_impl
// Global functions are defined elsewhere

// OpenCL helper functions implementation (minimal stubs)
bool initialize_opencl(cl_context& context, cl_command_queue& queue, cl_program& program, cl_kernel& crop_kernel, cl_kernel& rotate_kernel) {
    // For simplicity, we're not implementing actual OpenCL initialization
    // This would normally set up the OpenCL context, program, and kernels
    return false; // Always fall back to CPU implementation
}

void cleanup_opencl(cl_context& context, cl_command_queue& queue, cl_program& program, cl_kernel& crop_kernel, cl_kernel& rotate_kernel, cl_mem& input_buffer, cl_mem& output_buffer) {
    // Stub for cleanup
}

bool create_buffers(cl_context& context, cl_mem& input_buffer, cl_mem& output_buffer, size_t size) {
    // Stub for buffer creation
    return false;
}