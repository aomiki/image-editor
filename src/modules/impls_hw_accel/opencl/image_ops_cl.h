#pragma once
#ifndef image_ops_cl_h
#define image_ops_cl_h

#include "CL/cl.h"
#include <vector>
#include "image_tools.h"

class OpenCLImageOps {
private:
    cl_context context;
    cl_command_queue command_queue;
    cl_program program;
    cl_device_id device_id;
    bool initialized;

    // OpenCL kernels
    cl_kernel rgb_to_gray_kernel;
    cl_kernel gray_to_rgb_kernel;

    // Helper functions
    cl_program createProgram(const char* source);
    void checkError(cl_int error, const char* message);

public:
    OpenCLImageOps();
    ~OpenCLImageOps();

    // Initialize OpenCL context and create kernels
    bool initialize();

    // Image conversion operations
    bool convertRGBtoGray(const matrix_rgb& input, matrix_gray& output);
    bool convertGraytoRGB(const matrix_gray& input, matrix_rgb& output);
};

#endif // image_ops_cl_h
