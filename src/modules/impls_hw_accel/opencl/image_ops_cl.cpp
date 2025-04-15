#include "image_ops_cl.h"
#include <fstream>
#include <iostream>
#include <string>
#include <cstring>

// OpenCL kernel source code
const char* kernelSource = R"(
    __kernel void rgb_to_gray(__global const uchar* input,
                             __global uchar* output,
                             const int width,
                             const int height) {
        int x = get_global_id(0);
        int y = get_global_id(1);
        
        if (x < width && y < height) {
            int idx = (y * width + x) * 3;
            uchar r = input[idx];
            uchar g = input[idx + 1];
            uchar b = input[idx + 2];
            
            // Convert RGB to grayscale using luminance formula
            output[y * width + x] = (uchar)(0.299f * r + 0.587f * g + 0.114f * b);
        }
    }

    __kernel void gray_to_rgb(__global const uchar* input,
                             __global uchar* output,
                             const int width,
                             const int height) {
        int x = get_global_id(0);
        int y = get_global_id(1);
        
        if (x < width && y < height) {
            int gray_idx = y * width + x;
            int rgb_idx = gray_idx * 3;
            
            uchar gray = input[gray_idx];
            output[rgb_idx] = gray;     // R
            output[rgb_idx + 1] = gray; // G
            output[rgb_idx + 2] = gray; // B
        }
    }
)";

OpenCLImageOps::OpenCLImageOps() : initialized(false) {
    // Initialize OpenCL context and device
    cl_platform_id platform_id = NULL;
    cl_uint ret_num_devices;
    cl_uint ret_num_platforms;
    cl_int ret = clGetPlatformIDs(1, &platform_id, &ret_num_platforms);
    checkError(ret, "Failed to get platform ID");

    ret = clGetDeviceIDs(platform_id, CL_DEVICE_TYPE_GPU, 1, &device_id, &ret_num_devices);
    checkError(ret, "Failed to get device ID");

    context = clCreateContext(NULL, 1, &device_id, NULL, NULL, &ret);
    checkError(ret, "Failed to create context");

    command_queue = clCreateCommandQueue(context, device_id, 0, &ret);
    checkError(ret, "Failed to create command queue");

    program = createProgram(kernelSource);
    
    // Create kernels
    rgb_to_gray_kernel = clCreateKernel(program, "rgb_to_gray", &ret);
    checkError(ret, "Failed to create rgb_to_gray kernel");
    
    gray_to_rgb_kernel = clCreateKernel(program, "gray_to_rgb", &ret);
    checkError(ret, "Failed to create gray_to_rgb kernel");

    initialized = true;
}

OpenCLImageOps::~OpenCLImageOps() {
    if (initialized) {
        clReleaseKernel(rgb_to_gray_kernel);
        clReleaseKernel(gray_to_rgb_kernel);
        clReleaseProgram(program);
        clReleaseCommandQueue(command_queue);
        clReleaseContext(context);
    }
}

cl_program OpenCLImageOps::createProgram(const char* source) {
    cl_int ret;
    cl_program program = clCreateProgramWithSource(context, 1, &source, NULL, &ret);
    checkError(ret, "Failed to create program");

    ret = clBuildProgram(program, 1, &device_id, NULL, NULL, NULL);
    if (ret != CL_SUCCESS) {
        size_t log_size;
        clGetProgramBuildInfo(program, device_id, CL_PROGRAM_BUILD_LOG, 0, NULL, &log_size);
        char* log = new char[log_size];
        clGetProgramBuildInfo(program, device_id, CL_PROGRAM_BUILD_LOG, log_size, log, NULL);
        std::cerr << "Build log:\n" << log << std::endl;
        delete[] log;
        checkError(ret, "Failed to build program");
    }

    return program;
}

void OpenCLImageOps::checkError(cl_int error, const char* message) {
    if (error != CL_SUCCESS) {
        std::cerr << "Error: " << message << " (Error code: " << error << ")" << std::endl;
        throw std::runtime_error(message);
    }
}

bool OpenCLImageOps::convertRGBtoGray(const matrix_rgb& input, matrix_gray& output) {
    if (!initialized) return false;

    cl_int ret;
    size_t input_size = input.width * input.height * 3;  // Calculate size directly
    size_t output_size = input.width * input.height;

    // Create temporary buffer for input data
    std::vector<unsigned char> input_data(input_size);
    // Copy data directly from the array
    memcpy(input_data.data(), input.arr, input_size);

    // Create OpenCL buffers
    cl_mem input_buffer = clCreateBuffer(context, CL_MEM_READ_ONLY, input_size, NULL, &ret);
    checkError(ret, "Failed to create input buffer");
    
    cl_mem output_buffer = clCreateBuffer(context, CL_MEM_WRITE_ONLY, output_size, NULL, &ret);
    checkError(ret, "Failed to create output buffer");

    // Write input data to device
    ret = clEnqueueWriteBuffer(command_queue, input_buffer, CL_TRUE, 0, input_size, input_data.data(), 0, NULL, NULL);
    checkError(ret, "Failed to write input buffer");

    // Set kernel arguments
    ret = clSetKernelArg(rgb_to_gray_kernel, 0, sizeof(cl_mem), &input_buffer);
    ret |= clSetKernelArg(rgb_to_gray_kernel, 1, sizeof(cl_mem), &output_buffer);
    ret |= clSetKernelArg(rgb_to_gray_kernel, 2, sizeof(int), &input.width);
    ret |= clSetKernelArg(rgb_to_gray_kernel, 3, sizeof(int), &input.height);
    checkError(ret, "Failed to set kernel arguments");

    // Execute kernel
    size_t global_work_size[2] = {input.width, input.height};
    ret = clEnqueueNDRangeKernel(command_queue, rgb_to_gray_kernel, 2, NULL, global_work_size, NULL, 0, NULL, NULL);
    checkError(ret, "Failed to execute kernel");

    // Read output data from device
    std::vector<unsigned char> output_data(output_size);
    ret = clEnqueueReadBuffer(command_queue, output_buffer, CL_TRUE, 0, output_size, output_data.data(), 0, NULL, NULL);
    checkError(ret, "Failed to read output buffer");

    // Clean up
    clReleaseMemObject(input_buffer);
    clReleaseMemObject(output_buffer);

    // Set output data
    output.resize(input.width, input.height);
    output.set_arr_interlaced(output_data.data());

    return true;
}

bool OpenCLImageOps::convertGraytoRGB(const matrix_gray& input, matrix_rgb& output) {
    if (!initialized) return false;

    cl_int ret;
    size_t input_size = input.width * input.height;  // Calculate size directly
    size_t output_size = input.width * input.height * 3;

    // Create temporary buffer for input data
    std::vector<unsigned char> input_data(input_size);
    // Copy data directly from the array
    memcpy(input_data.data(), input.arr, input_size);

    // Create OpenCL buffers
    cl_mem input_buffer = clCreateBuffer(context, CL_MEM_READ_ONLY, input_size, NULL, &ret);
    checkError(ret, "Failed to create input buffer");
    
    cl_mem output_buffer = clCreateBuffer(context, CL_MEM_WRITE_ONLY, output_size, NULL, &ret);
    checkError(ret, "Failed to create output buffer");

    // Write input data to device
    ret = clEnqueueWriteBuffer(command_queue, input_buffer, CL_TRUE, 0, input_size, input_data.data(), 0, NULL, NULL);
    checkError(ret, "Failed to write input buffer");

    // Set kernel arguments
    ret = clSetKernelArg(gray_to_rgb_kernel, 0, sizeof(cl_mem), &input_buffer);
    ret |= clSetKernelArg(gray_to_rgb_kernel, 1, sizeof(cl_mem), &output_buffer);
    ret |= clSetKernelArg(gray_to_rgb_kernel, 2, sizeof(int), &input.width);
    ret |= clSetKernelArg(gray_to_rgb_kernel, 3, sizeof(int), &input.height);
    checkError(ret, "Failed to set kernel arguments");

    // Execute kernel
    size_t global_work_size[2] = {input.width, input.height};
    ret = clEnqueueNDRangeKernel(command_queue, gray_to_rgb_kernel, 2, NULL, global_work_size, NULL, 0, NULL, NULL);
    checkError(ret, "Failed to execute kernel");

    // Read output data from device
    std::vector<unsigned char> output_data(output_size);
    ret = clEnqueueReadBuffer(command_queue, output_buffer, CL_TRUE, 0, output_size, output_data.data(), 0, NULL, NULL);
    checkError(ret, "Failed to read output buffer");

    // Clean up
    clReleaseMemObject(input_buffer);
    clReleaseMemObject(output_buffer);

    // Set output data
    output.resize(input.width, input.height);
    output.set_arr_interlaced(output_data.data());

    return true;
} 