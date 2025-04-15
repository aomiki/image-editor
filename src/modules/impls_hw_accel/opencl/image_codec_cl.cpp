#include <CL/cl2.hpp>
#include <vector>
#include <iostream>
#include <fstream>
#include <cstring>
#include "lodepng.h"
#include "image_codec.h"

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
        dst[index]   = src[index];
        dst[index+1] = src[index+1];
        dst[index+2] = src[index+2];
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
        dst[index]   = src[index];
        dst[index+1] = src[index+1];
        dst[index+2] = src[index+2];
    }
}
)CLC";

class image_codec_cl : public image_codec {
private:
    unsigned width;
    unsigned height;
    cl::Context context;
    cl::CommandQueue queue;
    cl::Program program;

public:
    image_codec_cl() : image_codec(), width(0), height(0) {
        // Получаем платформы OpenCL и выбираем первую доступную
        std::vector<cl::Platform> platforms;
        cl::Platform::get(&platforms);
        if (platforms.empty()) {
            std::cerr << "Не найдено OpenCL-платформ" << std::endl;
            exit(1);
        }
        cl::Platform platform = platforms[0];

        // Пытаемся получить GPU-устройство, при отсутствии используем CPU
        std::vector<cl::Device> devices;
        platform.getDevices(CL_DEVICE_TYPE_GPU, &devices);
        if (devices.empty()) {
            std::cerr << "GPU-устройств не найдено, пробуем CPU" << std::endl;
            platform.getDevices(CL_DEVICE_TYPE_CPU, &devices);
            if (devices.empty()) {
                std::cerr << "Не найдено ни GPU, ни CPU-устройств" << std::endl;
                exit(1);
            }
        }
        cl::Device device = devices[0];

        // Создаём OpenCL-контекст и очередь команд
        context = cl::Context(device);
        queue = cl::CommandQueue(context, device);

        // Создаём и компилируем программу на основе исходного кода ядер
        cl::Program::Sources sources;
        sources.push_back({codecKernelSource, std::strlen(codecKernelSource)});
        program = cl::Program(context, sources);
        if (program.build({device}) != CL_SUCCESS) {
            std::cerr << "Ошибка компиляции OpenCL-программы: "
                      << program.getBuildInfo<CL_PROGRAM_BUILD_LOG>(device) << std::endl;
            exit(1);
        }
    }

    ~image_codec_cl() {}

    // Загрузка PNG-файла с использованием lodepng
    void load_image_file(std::vector<unsigned char>* png_buffer, std::string image_filepath) {
        unsigned error = lodepng::decode(*png_buffer, width, height, image_filepath);
        if (error) {
            std::cerr << "Ошибка загрузки изображения: "
                      << lodepng_error_text(error) << std::endl;
        }
    }

    // Считывание информации об изображении
    ImageInfo read_info(std::vector<unsigned char>* img_buffer) {
        ImageInfo info;
        info.width = width;
        info.height = height;
        info.colorScheme = IMAGE_RGB;  // По умолчанию RGB
        info.bit_depth = 8;            // По умолчанию 8-бит
        return info;
    }

    // Декодирование (копирование данных изображения в матрицу) с использованием OpenCL
    void decode(std::vector<unsigned char>* img_source, matrix* img_matrix,
                ImageColorScheme colorScheme, unsigned bit_depth) {
        if (!img_source || !img_matrix) return;

        // Изменяем размер матрицы согласно габаритам изображения
        img_matrix->resize(width, height);

        // Создаём OpenCL-буфер для исходных данных изображения
        cl::Buffer bufferSrc(context, CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR,
                               img_source->size() * sizeof(unsigned char), img_source->data());
        // Подготовка буфера для результата
        size_t totalSize = width * height * 3;
        std::vector<unsigned char> result(totalSize, 0);
        cl::Buffer bufferDst(context, CL_MEM_WRITE_ONLY, totalSize * sizeof(unsigned char));

        // Настраиваем и запускаем ядро для декодирования
        cl::Kernel kernel(program, "decode_kernel");
        kernel.setArg(0, bufferSrc);
        kernel.setArg(1, bufferDst);
        kernel.setArg(2, width);
        kernel.setArg(3, height);

        cl::NDRange global(width, height);
        queue.enqueueNDRangeKernel(kernel, cl::NullRange, global, cl::NullRange);
        queue.finish();

        // Считываем результат из буфера
        queue.enqueueReadBuffer(bufferDst, CL_TRUE, 0, totalSize * sizeof(unsigned char), result.data());

        // Переносим данные в матрицу (здесь предполагается, что matrix обеспечивает метод get(x,y) для доступа к пикселю)
        for (unsigned int y = 0; y < height; y++) {
            for (unsigned int x = 0; x < width; x++) {
                unsigned char* pixel = img_matrix->get(x, y);
                int index = (y * width + x) * 3;
                pixel[0] = result[index];
                pixel[1] = result[index + 1];
                pixel[2] = result[index + 2];
            }
        }
    }

    // Кодирование (копирование данных из матрицы в буфер изображения) с использованием OpenCL
    void encode(std::vector<unsigned char>* img_buffer, matrix* img_matrix,
                ImageColorScheme colorScheme, unsigned bit_depth) {
        if (!img_buffer || !img_matrix) return;

        // Сначала копируем данные из матрицы в вектор (предполагается, что размеры матрицы равны width x height)
        size_t totalSize = width * height * 3;
        std::vector<unsigned char> srcData(totalSize, 0);
        for (unsigned int y = 0; y < height; y++) {
            for (unsigned int x = 0; x < width; x++) {
                unsigned char* pixel = img_matrix->get(x, y);
                int index = (y * width + x) * 3;
                srcData[index]     = pixel[0];
                srcData[index + 1] = pixel[1];
                srcData[index + 2] = pixel[2];
            }
        }

        // Создаём OpenCL-буфер для исходных данных матрицы
        cl::Buffer bufferSrc(context, CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR,
                               srcData.size() * sizeof(unsigned char), srcData.data());
        // Буфер для результата кодирования
        std::vector<unsigned char> result(totalSize, 0);
        cl::Buffer bufferDst(context, CL_MEM_WRITE_ONLY, totalSize * sizeof(unsigned char));

        // Настройка и запуск OpenCL-ядра для кодирования
        cl::Kernel kernel(program, "encode_kernel");
        kernel.setArg(0, bufferSrc);
        kernel.setArg(1, bufferDst);
        kernel.setArg(2, width);
        kernel.setArg(3, height);

        cl::NDRange global(width, height);
        queue.enqueueNDRangeKernel(kernel, cl::NullRange, global, cl::NullRange);
        queue.finish();

        // Считываем результат кодирования
        queue.enqueueReadBuffer(bufferDst, CL_TRUE, 0, totalSize * sizeof(unsigned char), result.data());

        // Записываем данные в вектор для дальнейшего сохранения
        img_buffer->resize(result.size());
        std::copy(result.begin(), result.end(), img_buffer->begin());
    }

    // Сохранение PNG-файла с использованием lodepng
    void save_image_file(std::vector<unsigned char>* png_buffer, std::string image_filepath) {
        unsigned error = lodepng::encode(image_filepath, *png_buffer, width, height);
        if (error) {
            std::cerr << "Ошибка сохранения изображения: "
                      << lodepng_error_text(error) << std::endl;
        }
    }
};
