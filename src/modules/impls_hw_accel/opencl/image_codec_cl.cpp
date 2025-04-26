#include <CL/opencl.hpp>
#include <vector>
#include <iostream>
#include <fstream>
#include <cstring>
#include "impls_hw_accel/opencl/image_codec_cl.h"

namespace {
    // OpenCL resources
    cl::Context context;
    cl::Device device;
    cl::Program program;
    cl::CommandQueue queue;
    bool initialized = false;
    unsigned width = 0;
    unsigned height = 0;

    // OpenCL kernels for image processing
    const char* codecKernelSource = R"CLC(
    __kernel void decode_kernel(__global const unsigned char* src,
                              __global unsigned char* dst,
                              const unsigned width,
                              const unsigned height)
    {
        int x = get_global_id(0);
        int y = get_global_id(1);
        if (x < width && y < height) {
            int index = (y * width + x) * 3;
            // Direct memory copy for RGB components
            dst[index] = src[index];
            dst[index + 1] = src[index + 1];
            dst[index + 2] = src[index + 2];
        }
    }

    __kernel void encode_kernel(__global const unsigned char* src,
                              __global unsigned char* dst,
                              const unsigned width,
                              const unsigned height)
    {
        int x = get_global_id(0);
        int y = get_global_id(1);
        if (x < width && y < height) {
            int index = (y * width + x) * 3;
            // Direct memory copy for RGB components
            dst[index] = src[index];
            dst[index + 1] = src[index + 1];
            dst[index + 2] = src[index + 2];
        }
    }
    )CLC";
}

image_codec_cl::image_codec_cl() : width(0), height(0), initialized(false) {}

image_codec_cl::~image_codec_cl() {}

bool image_codec_cl::initializeOpenCL() {
    if (initialized) return true;

    try {
        // Get available platforms
        std::vector<cl::Platform> platforms;
        cl::Platform::get(&platforms);
        if (platforms.empty()) {
            std::cerr << "No OpenCL platforms found" << std::endl;
            return false;
        }

        // Select first platform
        cl::Platform platform = platforms[0];

        // Get devices
        std::vector<cl::Device> devices;
        platform.getDevices(CL_DEVICE_TYPE_GPU, &devices);
        if (devices.empty()) {
            std::cerr << "No GPU devices found" << std::endl;
            return false;
        }

        // Select first device
        device = devices[0];

        // Create context and command queue
        context = cl::Context(device);
        queue = cl::CommandQueue(context, device);

        // Create and build program
        cl::Program::Sources sources;
        sources.push_back({codecKernelSource, std::strlen(codecKernelSource)});
        program = cl::Program(context, sources);
        
        if (program.build({device}) != CL_SUCCESS) {
            std::cerr << "Error building program: " 
                     << program.getBuildInfo<CL_PROGRAM_BUILD_LOG>(device) << std::endl;
            return false;
        }

        initialized = true;
        return true;
    } catch (const std::exception& e) {
        std::cerr << "OpenCL error: " << e.what() << std::endl;
        return false;
    }
}

void image_codec_cl::load_image_file(std::vector<unsigned char>* png_buffer, std::string image_filepath) {
    std::ifstream file(image_filepath, std::ios::binary);
    if (!file) {
        std::cerr << "Error opening file: " << image_filepath << std::endl;
        return;
    }

    // Get file size
    file.seekg(0, std::ios::end);
    size_t size = file.tellg();
    file.seekg(0, std::ios::beg);

    // Read data directly into buffer
    png_buffer->resize(size);
    file.read(reinterpret_cast<char*>(png_buffer->data()), size);
    file.close();

    // Set image dimensions (assuming RGB)
    width = 0;  // To be determined from file header
    height = 0; // To be determined from file header
}

ImageInfo image_codec_cl::read_info(std::vector<unsigned char>* img_buffer) {
    ImageInfo info;
    info.width = width;
    info.height = height;
    info.colorScheme = IMAGE_RGB;
    info.bit_depth = 8;
    return info;
}

void image_codec_cl::decode(std::vector<unsigned char>* img_source, matrix* img_matrix, ImageColorScheme colorScheme, unsigned bit_depth) {
    if (!img_source || !img_matrix) return;
    
    if (!initializeOpenCL()) return;

    try {
        // Create OpenCL buffers with direct memory access
        cl::Buffer src_buffer(context, CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR,
                            img_source->size(), img_source->data());
        
        cl::Buffer dst_buffer(context, CL_MEM_WRITE_ONLY,
                            img_matrix->size() * sizeof(unsigned char));

        // Create and configure kernel
        cl::Kernel kernel(program, "decode_kernel");
        kernel.setArg(0, src_buffer);
        kernel.setArg(1, dst_buffer);
        kernel.setArg(2, width);
        kernel.setArg(3, height);

        // Execute kernel
        queue.enqueueNDRangeKernel(kernel, cl::NullRange,
                                 cl::NDRange(width, height),
                                 cl::NullRange);

        // Read results directly into matrix
        queue.enqueueReadBuffer(dst_buffer, CL_TRUE, 0,
                              img_matrix->size() * sizeof(unsigned char),
                              img_matrix->arr);

    } catch (const std::exception& e) {
        std::cerr << "OpenCL error: " << e.what() << std::endl;
    }
}

void image_codec_cl::encode(std::vector<unsigned char>* img_buffer, matrix* img_matrix, ImageColorScheme colorScheme, unsigned bit_depth) {
    if (!img_buffer || !img_matrix) return;
    
    if (!initializeOpenCL()) return;

    try {
        // Create OpenCL buffers with direct memory access
        cl::Buffer src_buffer(context, CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR,
                            img_matrix->size() * sizeof(unsigned char),
                            img_matrix->arr);
        
        cl::Buffer dst_buffer(context, CL_MEM_WRITE_ONLY,
                            img_buffer->size());

        // Create and configure kernel
        cl::Kernel kernel(program, "encode_kernel");
        kernel.setArg(0, src_buffer);
        kernel.setArg(1, dst_buffer);
        kernel.setArg(2, width);
        kernel.setArg(3, height);

        // Execute kernel
        queue.enqueueNDRangeKernel(kernel, cl::NullRange,
                                 cl::NDRange(width, height),
                                 cl::NullRange);

        // Read results directly into buffer
        queue.enqueueReadBuffer(dst_buffer, CL_TRUE, 0,
                              img_buffer->size(),
                              img_buffer->data());

    } catch (const std::exception& e) {
        std::cerr << "OpenCL error: " << e.what() << std::endl;
    }
}

void image_codec_cl::save_image_file(std::vector<unsigned char>* png_buffer, std::string image_filepath) {
    std::ofstream file(image_filepath, std::ios::binary);
    if (!file) {
        std::cerr << "Error opening file for writing: " << image_filepath << std::endl;
        return;
    }

    // Write data directly from buffer
    file.write(reinterpret_cast<const char*>(png_buffer->data()), png_buffer->size());
    file.close();
}
