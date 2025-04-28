#ifdef OPENCL_IMPL

#include "image_transforms.h"
#include <CL/cl.h>
#include <iostream>
#include <vector>
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

namespace opencl_impl {

void crop(matrix& img, unsigned crop_left, unsigned crop_top, unsigned crop_right, unsigned crop_bottom) {
    std::cout << "[OpenCL Transforms] Cropping image using CPU implementation" << std::endl;
    
    // CPU реализация обрезки (можно позже заменить на GPU)
    unsigned new_width = img.width - crop_left - crop_right;
    unsigned new_height = img.height - crop_top - crop_bottom;
    
    unsigned char* newArr = new unsigned char[new_width * new_height * img.components_num];
    
    for (unsigned y = 0; y < new_height; ++y) {
        for (unsigned x = 0; x < new_width; ++x) {
            unsigned oldX = x + crop_left;
            unsigned oldY = y + crop_top;
            
            unsigned char* old_pixel = img.get(oldX, oldY);
            unsigned char* new_pixel = &newArr[(y * new_width + x) * img.components_num];
            
            memcpy(new_pixel, old_pixel, img.components_num);
        }
    }
    
    delete[] img.arr;
    img.set_arr_interlaced(newArr, new_width, new_height);
    
    std::cout << "[OpenCL Transforms] Crop complete, new dimensions: " 
              << img.width << "x" << img.height << std::endl;
}

void rotate(matrix& img, unsigned angle) {
    std::cout << "[OpenCL Transforms] Rotating image by " << angle << " degrees using GPU" << std::endl;
    
    // Создаем экземпляр OpenCL кодека
    image_codec_cl cl_codec;
    
    // Применяем поворот на GPU
    if (!cl_codec.rotate_on_gpu(&img, angle)) {
        std::cerr << "[OpenCL Transforms] GPU rotation failed, falling back to CPU implementation" << std::endl;
        
        // Если GPU обработка не удалась, выполняем CPU реализацию
        angle = angle % 360;
        if (angle == 0) return;
        
        unsigned new_width = (angle == 90 || angle == 270) ? img.height : img.width;
        unsigned new_height = (angle == 90 || angle == 270) ? img.width : img.height;
        
        unsigned char* newArr = new unsigned char[new_width * new_height * img.components_num];
        
        for (unsigned y = 0; y < img.height; ++y) {
            for (unsigned x = 0; x < img.width; ++x) {
                unsigned char* old_pixel = img.get(x, y);
                unsigned char* new_pixel = nullptr;
                
                switch (angle) {
                    case 90:
                        new_pixel = &newArr[(x * new_width + (new_width - y - 1)) * img.components_num];
                        break;
                    case 180:
                        new_pixel = &newArr[((new_height - y - 1) * new_width + (new_width - x - 1)) * img.components_num];
                        break;
                    case 270:
                        new_pixel = &newArr[((new_height - x - 1) * new_width + y) * img.components_num];
                        break;
                }
                
                if (new_pixel) {
                    memcpy(new_pixel, old_pixel, img.components_num);
                }
            }
        }
        
        delete[] img.arr;
        img.set_arr_interlaced(newArr, new_width, new_height);
    }
    
    std::cout << "[OpenCL Transforms] Rotation complete, new dimensions: " 
              << img.width << "x" << img.height << std::endl;
}

} // namespace opencl_impl

// Глобальные функции для OpenCL-реализации
void crop(matrix& img, unsigned crop_left, unsigned crop_top, unsigned crop_right, unsigned crop_bottom) {
    opencl_impl::crop(img, crop_left, crop_top, crop_right, crop_bottom);
}

void rotate(matrix& img, unsigned angle) {
    opencl_impl::rotate(img, angle);
}

// OpenCL helper functions implementation
bool initialize_opencl(cl_context& context, cl_command_queue& queue, cl_program& program, cl_kernel& crop_kernel, cl_kernel& rotate_kernel) {
    cl_int err;

    // Get platform
    cl_platform_id platform;
    err = clGetPlatformIDs(1, &platform, NULL);
    if (err != CL_SUCCESS) {
        std::cerr << "Failed to get platform" << std::endl;
        return false;
    }

    // Get device
    cl_device_id device;
    err = clGetDeviceIDs(platform, CL_DEVICE_TYPE_GPU, 1, &device, NULL);
    if (err != CL_SUCCESS) {
        std::cerr << "Failed to get device" << std::endl;
        return false;
    }

    // Create context
    context = clCreateContext(NULL, 1, &device, NULL, NULL, &err);
    if (err != CL_SUCCESS) {
        std::cerr << "Failed to create context" << std::endl;
        return false;
    }

    // Create command queue
    queue = clCreateCommandQueue(context, device, 0, &err);
    if (err != CL_SUCCESS) {
        std::cerr << "Failed to create command queue" << std::endl;
        return false;
    }

    // Create program with kernels
    const char* kernel_source = R"(
        __kernel void crop(
            __global const unsigned char* input,
            __global unsigned char* output,
            unsigned width,
            unsigned height,
            unsigned new_width,
            unsigned crop_left,
            unsigned crop_top,
            unsigned channels
        ) {
            int x = get_global_id(0);
            int y = get_global_id(1);
            
            if (x < new_width && y < height - crop_top - crop_bottom) {
                int src_x = x + crop_left;
                int src_y = y + crop_top;
                int src_idx = (src_y * width + src_x) * channels;
                int dst_idx = (y * new_width + x) * channels;
                
                for (int c = 0; c < channels; c++) {
                    output[dst_idx + c] = input[src_idx + c];
                }
            }
        }

        __kernel void rotate(
            __global const unsigned char* input,
            __global unsigned char* output,
            unsigned width,
            unsigned height,
            unsigned new_width,
            unsigned new_height,
            unsigned angle,
            unsigned channels
        ) {
            int x = get_global_id(0);
            int y = get_global_id(1);
            
            if (x < new_width && y < new_height) {
                int src_x, src_y;
                
                switch (angle) {
                    case 90:
                        src_x = y;
                        src_y = width - 1 - x;
                        break;
                    case 180:
                        src_x = width - 1 - x;
                        src_y = height - 1 - y;
                        break;
                    case 270:
                        src_x = height - 1 - y;
                        src_y = x;
                        break;
                    default:
                        src_x = x;
                        src_y = y;
                }
                
                int src_idx = (src_y * width + src_x) * channels;
                int dst_idx = (y * new_width + x) * channels;
                
                for (int c = 0; c < channels; c++) {
                    output[dst_idx + c] = input[src_idx + c];
                }
            }
        }
    )";

    program = clCreateProgramWithSource(context, 1, &kernel_source, NULL, &err);
    if (err != CL_SUCCESS) {
        std::cerr << "Failed to create program" << std::endl;
        return false;
    }

    // Build program
    err = clBuildProgram(program, 1, &device, NULL, NULL, NULL);
    if (err != CL_SUCCESS) {
        std::cerr << "Failed to build program" << std::endl;
        return false;
    }

    // Create kernels
    crop_kernel = clCreateKernel(program, "crop", &err);
    if (err != CL_SUCCESS) {
        std::cerr << "Failed to create crop kernel" << std::endl;
        return false;
    }

    rotate_kernel = clCreateKernel(program, "rotate", &err);
    if (err != CL_SUCCESS) {
        std::cerr << "Failed to create rotate kernel" << std::endl;
        return false;
    }

    return true;
}

void cleanup_opencl(cl_context& context, cl_command_queue& queue, cl_program& program, cl_kernel& crop_kernel, cl_kernel& rotate_kernel, cl_mem& input_buffer, cl_mem& output_buffer) {
    if (crop_kernel) clReleaseKernel(crop_kernel);
    if (rotate_kernel) clReleaseKernel(rotate_kernel);
    if (program) clReleaseProgram(program);
    if (input_buffer) clReleaseMemObject(input_buffer);
    if (output_buffer) clReleaseMemObject(output_buffer);
    if (queue) clReleaseCommandQueue(queue);
    if (context) clReleaseContext(context);
}

bool create_buffers(cl_context& context, cl_mem& input_buffer, cl_mem& output_buffer, size_t size) {
    cl_int err;

    // Create or resize input buffer
    if (input_buffer) clReleaseMemObject(input_buffer);
    input_buffer = clCreateBuffer(context, CL_MEM_READ_ONLY, size, NULL, &err);
    if (err != CL_SUCCESS) {
        std::cerr << "Failed to create input buffer" << std::endl;
        return false;
    }

    // Create or resize output buffer
    if (output_buffer) clReleaseMemObject(output_buffer);
    output_buffer = clCreateBuffer(context, CL_MEM_WRITE_ONLY, size, NULL, &err);
    if (err != CL_SUCCESS) {
        std::cerr << "Failed to create output buffer" << std::endl;
        return false;
    }

    return true;
}

#endif 