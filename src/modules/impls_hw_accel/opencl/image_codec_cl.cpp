#include <CL/opencl.hpp>
#include <vector>
#include <iostream>
#include <fstream>
#include <cstring>
#include "image_codec.h"

class image_codec_cl : public image_codec {
private:
    unsigned width;
    unsigned height;
    cl::Context context;
    cl::Device device;
    cl::Program program;
    cl::CommandQueue queue;
    bool initialized;

    // Исходный код OpenCL-ядер для декодирования и кодирования
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
            // Простейшее копирование для 3-х компонент RGB
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
            // Простейшее копирование для 3-х компонент RGB
            dst[index] = src[index];
            dst[index + 1] = src[index + 1];
            dst[index + 2] = src[index + 2];
        }
    }
    )CLC";

    bool initializeOpenCL() {
        if (initialized) return true;

        try {
            // Получаем список доступных платформ
            std::vector<cl::Platform> platforms;
            cl::Platform::get(&platforms);
            if (platforms.empty()) {
                std::cerr << "No OpenCL platforms found" << std::endl;
                return false;
            }

            // Выбираем первую платформу
            cl::Platform platform = platforms[0];

            // Получаем список устройств
            std::vector<cl::Device> devices;
            platform.getDevices(CL_DEVICE_TYPE_GPU, &devices);
            if (devices.empty()) {
                std::cerr << "No GPU devices found" << std::endl;
                return false;
            }

            // Выбираем первое устройство
            device = devices[0];

            // Создаём контекст
            context = cl::Context(device);

            // Создаём очередь команд
            queue = cl::CommandQueue(context, device);

            // Создаём и компилируем программу
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

public:
    image_codec_cl() : width(0), height(0), initialized(false) {}
    ~image_codec_cl() {}

    void load_image_file(std::vector<unsigned char>* png_buffer, std::string image_filepath) {
        // Читаем файл напрямую
        std::ifstream file(image_filepath, std::ios::binary);
        if (!file) {
            std::cerr << "Error opening file: " << image_filepath << std::endl;
            return;
        }

        // Получаем размер файла
        file.seekg(0, std::ios::end);
        size_t size = file.tellg();
        file.seekg(0, std::ios::beg);

        // Читаем данные
        png_buffer->resize(size);
        file.read(reinterpret_cast<char*>(png_buffer->data()), size);
        file.close();

        // Устанавливаем размеры изображения (предполагаем RGB)
        width = 0;  // Нужно будет определить из заголовка файла
        height = 0; // Нужно будет определить из заголовка файла
    }

    ImageInfo read_info(std::vector<unsigned char>* img_buffer) {
        ImageInfo info;
        info.width = width;
        info.height = height;
        info.colorScheme = IMAGE_RGB;
        info.bit_depth = 8;
        return info;
    }

    void decode(std::vector<unsigned char>* img_source, matrix* img_matrix, ImageColorScheme colorScheme, unsigned bit_depth) {
        if (!img_source || !img_matrix) return;
        
        if (!initializeOpenCL()) return;

        try {
            // Создаём буферы OpenCL
            cl::Buffer src_buffer(context, CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR,
                                img_source->size(), img_source->data());
            
            // Создаём буфер для результата
            cl::Buffer dst_buffer(context, CL_MEM_WRITE_ONLY,
                                img_matrix->size() * sizeof(unsigned char));

            // Создаём ядро
            cl::Kernel kernel(program, "decode_kernel");

            // Устанавливаем аргументы
            kernel.setArg(0, src_buffer);
            kernel.setArg(1, dst_buffer);
            kernel.setArg(2, width);
            kernel.setArg(3, height);

            // Запускаем ядро
            queue.enqueueNDRangeKernel(kernel, cl::NullRange,
                                     cl::NDRange(width, height),
                                     cl::NullRange);

            // Читаем результат
            queue.enqueueReadBuffer(dst_buffer, CL_TRUE, 0,
                                  img_matrix->size() * sizeof(unsigned char),
                                  img_matrix->arr);

        } catch (const std::exception& e) {
            std::cerr << "OpenCL error: " << e.what() << std::endl;
        }
    }

    void encode(std::vector<unsigned char>* img_buffer, matrix* img_matrix, ImageColorScheme colorScheme, unsigned bit_depth) {
        if (!img_buffer || !img_matrix) return;
        
        if (!initializeOpenCL()) return;

        try {
            // Создаём буферы OpenCL
            cl::Buffer src_buffer(context, CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR,
                                img_matrix->size() * sizeof(unsigned char),
                                img_matrix->arr);
            
            // Создаём буфер для результата
            cl::Buffer dst_buffer(context, CL_MEM_WRITE_ONLY,
                                img_buffer->size());

            // Создаём ядро
            cl::Kernel kernel(program, "encode_kernel");

            // Устанавливаем аргументы
            kernel.setArg(0, src_buffer);
            kernel.setArg(1, dst_buffer);
            kernel.setArg(2, width);
            kernel.setArg(3, height);

            // Запускаем ядро
            queue.enqueueNDRangeKernel(kernel, cl::NullRange,
                                     cl::NDRange(width, height),
                                     cl::NullRange);

            // Читаем результат
            queue.enqueueReadBuffer(dst_buffer, CL_TRUE, 0,
                                  img_buffer->size(),
                                  img_buffer->data());

        } catch (const std::exception& e) {
            std::cerr << "OpenCL error: " << e.what() << std::endl;
        }
    }

    void save_image_file(std::vector<unsigned char>* png_buffer, std::string image_filepath) {
        // Записываем файл напрямую
        std::ofstream file(image_filepath, std::ios::binary);
        if (!file) {
            std::cerr << "Error opening file for writing: " << image_filepath << std::endl;
            return;
        }

        file.write(reinterpret_cast<const char*>(png_buffer->data()), png_buffer->size());
        file.close();
    }
};
