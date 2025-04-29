#define CL_HPP_ENABLE_EXCEPTIONS
#define CL_HPP_TARGET_OPENCL_VERSION 200
#define CL_HPP_MINIMUM_OPENCL_VERSION 120

#include <CL/opencl.hpp>
#include <iostream>
#include <vector>

int main() {
    try {
        std::vector<cl::Platform> platforms;
        cl::Platform::get(&platforms);
        std::cout << "Found " << platforms.size() << " platform(s)." << std::endl;

        for (const auto& platform : platforms) {
            std::string name = platform.getInfo<CL_PLATFORM_NAME>();
            std::cout << "Platform: " << name << std::endl;

            std::vector<cl::Device> devices;
            platform.getDevices(CL_DEVICE_TYPE_GPU, &devices);
            for (const auto& device : devices) {
                std::cout << "  Device: " << device.getInfo<CL_DEVICE_NAME>() << std::endl;
            }
        }
    } catch (cl::Error& e) {
        std::cerr << "OpenCL Error: " << e.what() << " (" << e.err() << ")" << std::endl;
    }
    return 0;
}
