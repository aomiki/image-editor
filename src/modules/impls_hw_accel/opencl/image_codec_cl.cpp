#include <CL/opencl.hpp>
#include <vector>
#include <iostream>
#include <fstream>
#include <cstring>
#include <sstream>
#include <iomanip>
#include "impls_hw_accel/opencl/image_codec_cl.h"
#include <filesystem>
#include "lodepng.h"
#include "image_edit.h" // For global flags

inline ImageColorScheme LodePNGColorTypeToImageColorScheme(LodePNGColorType color_type)
{
    switch (color_type)
    {
        case LodePNGColorType::LCT_RGBA:
        case LodePNGColorType::LCT_RGB:
            return ImageColorScheme::IMAGE_RGB;
        case LodePNGColorType::LCT_PALETTE:
            return ImageColorScheme::IMAGE_PALETTE;
        case LodePNGColorType::LCT_GREY:
            return ImageColorScheme::IMAGE_GRAY;
        case LodePNGColorType::LCT_GREY_ALPHA:
            return ImageColorScheme::IMAGE_RGB;
        case LodePNGColorType::LCT_MAX_OCTET_VALUE:
        default:
            return ImageColorScheme::IMAGE_RGB;
    }
}

namespace {
    // OpenCL resources
    cl::Context global_context;
    cl::Device global_device;
    cl::Program global_program;
    cl::CommandQueue global_queue;

    // OpenCL kernels for image processing
    const char* codecKernelSource = R"CLC(
    __kernel void decode_kernel(__global const unsigned char* src,
                              __global unsigned char* dst,
                              const unsigned width,
                              const unsigned height,
                              const unsigned components_num)
    {
        int x = get_global_id(0);
        int y = get_global_id(1);
        if (x < width && y < height) {
            int src_idx = (y * width + x) * components_num;
            int dst_idx = (y * width + x) * components_num;

            // Копирование пикселя из src в dst
            for (int c = 0; c < components_num; c++) {
                dst[dst_idx + c] = src[src_idx + c];
            }
        }
    }

    __kernel void encode_kernel(__global const unsigned char* src,
                              __global unsigned char* dst,
                              const unsigned width,
                              const unsigned height,
                              const unsigned components_num)
    {
        int x = get_global_id(0);
        int y = get_global_id(1);
        if (x < width && y < height) {
            int src_idx = (y * width + x) * components_num;
            int dst_idx = (y * width + x) * components_num;

            // Копирование пикселя из src в dst
            for (int c = 0; c < components_num; c++) {
                dst[dst_idx + c] = src[src_idx + c];
            }
        }
    }

    // Простое копирование данных без инверсии
    __kernel void process_image(__global const unsigned char* src,
                             __global unsigned char* dst,
                             const unsigned width,
                             const unsigned height,
                             const unsigned components_num)
    {
        int x = get_global_id(0);
        int y = get_global_id(1);
        if (x < width && y < height) {
            int src_idx = (y * width + x) * components_num;
            int dst_idx = (y * width + x) * components_num;

            // Простое копирование без инверсии цветов
            for (int c = 0; c < components_num; c++) {
                dst[dst_idx + c] = src[src_idx + c];
            }
        }
    }

    // Поворот изображения на 90 градусов по часовой стрелке
    __kernel void rotate_90_cw(__global const unsigned char* src,
                              __global unsigned char* dst,
                              const unsigned src_width,
                              const unsigned src_height,
                              const unsigned components_num)
    {
        int x = get_global_id(0);
        int y = get_global_id(1);

        if (x < src_width && y < src_height) {
            // В повернутом изображении координаты меняются:
            // новый_x = y
            // новый_y = (ширина - 1) - x
            int dst_x = y;
            int dst_y = (src_width - 1) - x;

            // Рассчитываем индексы в буферах
            int src_idx = (y * src_width + x) * components_num;
            int dst_idx = (dst_y * src_height + dst_x) * components_num;

            // Копируем пиксель
            for (int c = 0; c < components_num; c++) {
                dst[dst_idx + c] = src[src_idx + c];
            }
        }
    }

    // Поворот изображения на 180 градусов
    __kernel void rotate_180(__global const unsigned char* src,
                           __global unsigned char* dst,
                           const unsigned width,
                           const unsigned height,
                           const unsigned components_num)
    {
        int x = get_global_id(0);
        int y = get_global_id(1);

        if (x < width && y < height) {
            // В повернутом изображении координаты меняются:
            // новый_x = (ширина - 1) - x
            // новый_y = (высота - 1) - y
            int dst_x = (width - 1) - x;
            int dst_y = (height - 1) - y;

            // Рассчитываем индексы в буферах
            int src_idx = (y * width + x) * components_num;
            int dst_idx = (dst_y * width + dst_x) * components_num;

            // Копируем пиксель
            for (int c = 0; c < components_num; c++) {
                dst[dst_idx + c] = src[src_idx + c];
            }
        }
    }

    // Поворот изображения на 270 градусов по часовой стрелке (90 градусов против часовой)
    __kernel void rotate_270_cw(__global const unsigned char* src,
                              __global unsigned char* dst,
                              const unsigned src_width,
                              const unsigned src_height,
                              const unsigned components_num)
    {
        int x = get_global_id(0);
        int y = get_global_id(1);

        if (x < src_width && y < src_height) {
            // В повернутом изображении координаты меняются:
            // новый_x = (высота - 1) - y
            // новый_y = x
            int dst_x = (src_height - 1) - y;
            int dst_y = x;

            // Рассчитываем индексы в буферах
            int src_idx = (y * src_width + x) * components_num;
            int dst_idx = (dst_y * src_height + dst_x) * components_num;

            // Копируем пиксель
            for (int c = 0; c < components_num; c++) {
                dst[dst_idx + c] = src[src_idx + c];
            }
        }
    }
    )CLC";
}

image_codec_cl::image_codec_cl() : width(0), height(0), initialized(false) {
    std::cout << "[OpenCL] Initializing image_codec_cl" << std::endl;

    // Try to initialize OpenCL right away to catch errors early
    if (!initializeOpenCL()) {
        std::cerr << "[OpenCL] Failed to initialize OpenCL during construction" << std::endl;
        throw std::runtime_error("Failed to initialize OpenCL");
    }
}

image_codec_cl::~image_codec_cl() {
    std::cout << "[OpenCL] Destroying image_codec_cl" << std::endl;
}

bool image_codec_cl::initializeOpenCL() {
    if (initialized) return true;

    try {
        if (g_verbose_enabled) {
            std::cout << "[OpenCL] Initializing OpenCL" << std::endl;
        }

        // Get available platforms
        std::vector<cl::Platform> platforms;
        cl::Platform::get(&platforms);
        if (platforms.empty()) {
            std::cerr << "[OpenCL] No OpenCL platforms found" << std::endl;
            return false;
        }

        if (g_verbose_enabled) {
            std::cout << "[OpenCL] Found " << platforms.size() << " platforms:" << std::endl;
            for (size_t i = 0; i < platforms.size(); i++) {
                std::string platformName = platforms[i].getInfo<CL_PLATFORM_NAME>();
                std::string platformVendor = platforms[i].getInfo<CL_PLATFORM_VENDOR>();
                std::string platformVersion = platforms[i].getInfo<CL_PLATFORM_VERSION>();
                std::cout << "  Platform " << i << ": " << platformName << " (" << platformVendor << "), " << platformVersion << std::endl;
            }
        }

        // Select first platform
        cl::Platform platform = platforms[0];

        if (g_verbose_enabled) {
            std::cout << "[OpenCL] Selected platform: " << platform.getInfo<CL_PLATFORM_NAME>() << std::endl;
        }

        // Get devices
        std::vector<cl::Device> devices;
        platform.getDevices(CL_DEVICE_TYPE_ALL, &devices);

        if (devices.empty()) {
            std::cerr << "[OpenCL] No GPU devices found" << std::endl;

            if (g_force_gpu_enabled) {
                std::cerr << "[OpenCL] Force GPU is enabled, so not falling back to CPU" << std::endl;
                return false;
            }

            std::cerr << "[OpenCL] Trying CPU devices instead" << std::endl;
            platform.getDevices(CL_DEVICE_TYPE_CPU, &devices);
            if (devices.empty()) {
                std::cerr << "[OpenCL] No CPU devices found either" << std::endl;
                return false;
            }
        }

        if (g_verbose_enabled) {
            std::cout << "[OpenCL] Found " << devices.size() << " devices:" << std::endl;
            for (size_t i = 0; i < devices.size(); i++) {
                std::string deviceName = devices[i].getInfo<CL_DEVICE_NAME>();
                std::string deviceVendor = devices[i].getInfo<CL_DEVICE_VENDOR>();
                std::string deviceVersion = devices[i].getInfo<CL_DEVICE_VERSION>();
                cl_device_type deviceType = devices[i].getInfo<CL_DEVICE_TYPE>();

                std::string typeStr;
                if (deviceType == CL_DEVICE_TYPE_CPU) typeStr = "CPU";
                else if (deviceType == CL_DEVICE_TYPE_GPU) typeStr = "GPU";
                else if (deviceType == CL_DEVICE_TYPE_ACCELERATOR) typeStr = "Accelerator";
                else typeStr = "Other";

                std::cout << "  Device " << i << ": " << deviceName << " (" << deviceVendor << "), "
                          << deviceVersion << ", Type: " << typeStr << std::endl;
            }
        }

        // Select first device
        device = devices[0];
        cl_device_type deviceType = device.getInfo<CL_DEVICE_TYPE>();
        bool isGPU = (deviceType == CL_DEVICE_TYPE_GPU);

        if (!isGPU && g_force_gpu_enabled) {
            std::cerr << "[OpenCL] Selected device is not a GPU, but force GPU is enabled. Aborting." << std::endl;
            return false;
        }

        std::string deviceName = device.getInfo<CL_DEVICE_NAME>();
        std::cout << "[OpenCL] Using device: " << deviceName
                  << " (Type: " << (isGPU ? "GPU" : "CPU") << ")" << std::endl;

        // Create context and command queue
        context = cl::Context(device);
        queue = cl::CommandQueue(context, device);

        // Create and build program
        cl::Program::Sources sources;
        sources.push_back({codecKernelSource, std::strlen(codecKernelSource)});
        program = cl::Program(context, sources);

        if (program.build({device}) != CL_SUCCESS) {
            std::cerr << "[OpenCL] Error building program: "
                     << program.getBuildInfo<CL_PROGRAM_BUILD_LOG>(device) << std::endl;
            return false;
        }

        if (g_verbose_enabled) {
            std::cout << "[OpenCL] Program built successfully" << std::endl;
        }

        // Store copies in anonymous namespace for shared access
        global_context = context;
        global_device = device;
        global_program = program;
        global_queue = queue;

        initialized = true;
        return true;
    } catch (const std::exception& e) {
        std::cerr << "[OpenCL] Error during initialization: " << e.what() << std::endl;
        return false;
    }
}

void image_codec_cl::load_image_file(std::vector<unsigned char>* png_buffer, std::string image_filepath) {
    std::cout << "[OpenCL] Loading image file: " << image_filepath << std::endl;

    unsigned error = lodepng::load_file(*png_buffer, image_filepath);

    if (error) {
        std::cerr << "[OpenCL] Error loading file: " << lodepng_error_text(error) << std::endl;
        return;
    }

    std::cout << "[OpenCL] File loaded, buffer size: " << png_buffer->size() << " bytes" << std::endl;
}

ImageInfo image_codec_cl::read_info(std::vector<unsigned char>* img_buffer) {
    LodePNGState state;
    lodepng_state_init(&state);

    ImageInfo img_info;
    lodepng_inspect(&img_info.width, &img_info.height, &state, img_buffer->data(), img_buffer->size());

    img_info.colorScheme = LodePNGColorTypeToImageColorScheme(state.info_png.color.colortype);
    img_info.bit_depth = state.info_png.color.bitdepth;

    return img_info;
}

void image_codec_cl::decode(std::vector<unsigned char>* img_source, matrix* img_matrix, ImageColorScheme colorScheme, unsigned bit_depth) {
    std::cout << "[OpenCL] Decoding image, buffer size: " << img_source->size()
              << ", target matrix: " << img_matrix->width << "x" << img_matrix->height
              << ", components: " << img_matrix->components_num << std::endl;

    if (!img_source || !img_matrix) {
        std::cerr << "[OpenCL] Invalid source or matrix" << std::endl;
        return;
    }

    // Декодируем PNG с помощью LodePNG
    std::vector<unsigned char> decoded_data;
    unsigned w, h;
    LodePNGColorType color_type = (colorScheme == IMAGE_RGB) ? LCT_RGB : LCT_GREY;

    unsigned error = lodepng::decode(decoded_data, w, h, *img_source, color_type, bit_depth);

    if (error) {
        std::cerr << "[OpenCL] Error decoding PNG: " << lodepng_error_text(error) << std::endl;
        return;
    }

    std::cout << "[OpenCL] Image decoded with LodePNG: " << w << "x" << h
              << ", decoded data size: " << decoded_data.size() << " bytes" << std::endl;

    // Обновляем размеры
    width = w;
    height = h;

    // Инициализируем OpenCL для будущих операций
    initializeOpenCL();

    // Проверяем размеры матрицы и изменяем их если нужно
    if (img_matrix->width != width || img_matrix->height != height) {
        std::cout << "[OpenCL] Resizing matrix to match image: " << width << "x" << height << std::endl;
        img_matrix->resize(width, height);
    }

    // Просто копируем декодированные данные в матрицу без поворота
    std::memcpy(img_matrix->get_arr_interlaced(), decoded_data.data(), decoded_data.size());

    std::cout << "[OpenCL] Image copied to matrix" << std::endl;
}

void image_codec_cl::encode(std::vector<unsigned char>* img_buffer, matrix* img_matrix, ImageColorScheme colorScheme, unsigned bit_depth) {
    std::cout << "[OpenCL] Encoding image, matrix: " << img_matrix->width << "x" << img_matrix->height
              << ", components: " << img_matrix->components_num << std::endl;

    if (!img_buffer || !img_matrix) {
        std::cerr << "[OpenCL] Invalid buffer or matrix" << std::endl;
        return;
    }

    // Получаем данные из матрицы
    unsigned w = img_matrix->width;
    unsigned h = img_matrix->height;
    LodePNGColorType color_type = (colorScheme == IMAGE_RGB) ? LCT_RGB : LCT_GREY;

    // Кодируем в PNG с помощью LodePNG
    unsigned error = lodepng::encode(*img_buffer, img_matrix->get_arr_interlaced(), w, h, color_type, bit_depth);

    if (error) {
        std::cerr << "[OpenCL] Error encoding PNG: " << lodepng_error_text(error) << std::endl;
        return;
    }

    std::cout << "[OpenCL] Image encoded to PNG, buffer size: " << img_buffer->size() << " bytes" << std::endl;
}

void image_codec_cl::save_image_file(std::vector<unsigned char>* png_buffer, std::string image_filepath) {
    std::cout << "[OpenCL] Saving image to: " << image_filepath << ", buffer size: " << png_buffer->size() << " bytes" << std::endl;

    // Проверяем существование директории
    std::filesystem::path path(image_filepath);
    std::filesystem::path dir = path.parent_path();

    if (!dir.empty() && !std::filesystem::exists(dir)) {
        std::cout << "[OpenCL] Creating directory: " << dir << std::endl;
        std::filesystem::create_directories(dir);
    }

    // Добавляем расширение .png если его нет
    if (path.extension() != ".png") {
        image_filepath += ".png";
    }

    // Сохраняем файл с помощью LodePNG
    unsigned error = lodepng::save_file(*png_buffer, image_filepath);

    if (error) {
        std::cerr << "[OpenCL] Error saving PNG file: " << lodepng_error_text(error) << std::endl;
        return;
    }

    std::cout << "[OpenCL] File saved successfully as: " << image_filepath << std::endl;
}

// Function for rotating an image on GPU
bool image_codec_cl::rotate_on_gpu(matrix* img_matrix, unsigned angle) {
    if (g_verbose_enabled) {
        std::cout << "[OpenCL] Rotating image on GPU by " << angle << " degrees" << std::endl;
    }

    if (!img_matrix || !img_matrix->get_arr_interlaced()) {
        std::cerr << "[OpenCL] Invalid matrix for rotation" << std::endl;
        return false;
    }

    // Normalize angle to 0-359
    angle = angle % 360;
    if (angle == 0) {
        if (g_verbose_enabled) {
            std::cout << "[OpenCL] No rotation needed (angle=0)" << std::endl;
        }
        return true; // No rotation needed
    }

    // Initialize OpenCL if not initialized
    if (!initializeOpenCL()) {
        std::cerr << "[OpenCL] Failed to initialize OpenCL" << std::endl;
        return false;
    }

    // Check if we're on a GPU
    cl_device_type deviceType = device.getInfo<CL_DEVICE_TYPE>();
    bool isGPU = (deviceType == CL_DEVICE_TYPE_GPU);

    if (!isGPU) {
        std::cout << "[OpenCL] WARNING: Using CPU for OpenCL processing, not GPU!" << std::endl;

        if (g_force_gpu_enabled) {
            std::cerr << "[OpenCL] Force GPU is enabled, aborting CPU execution" << std::endl;
            return false;
        }
    } else {
        std::cout << "[OpenCL] Using GPU for OpenCL processing" << std::endl;
    }

    if (g_verbose_enabled) {
        std::cout << "[OpenCL] OpenCL initialized: " << initialized << std::endl;
    }

    try {
        // Get current matrix dimensions and data
        unsigned w = img_matrix->width;
        unsigned h = img_matrix->height;
        unsigned components = img_matrix->components_num;

        if (g_verbose_enabled) {
            std::cout << "[OpenCL] Input image dimensions: " << w << "x" << h
                    << " with " << components << " components" << std::endl;
        }

        // Create a copy of matrix data for processing
        size_t input_size = w * h * components;
        std::vector<unsigned char> input_data(img_matrix->get_arr_interlaced(),
                                              img_matrix->get_arr_interlaced() + input_size);

        // Determine new dimensions
        unsigned new_width = w;
        unsigned new_height = h;

        // For 90 and 270 degree rotations, swap width and height
        if (angle == 90 || angle == 270) {
            new_width = h;
            new_height = w;
        }

        if (g_verbose_enabled) {
            std::cout << "[OpenCL] Output dimensions will be: " << new_width << "x" << new_height << std::endl;
        }

        // Size of output data
        size_t output_size = new_width * new_height * components;

        // Create OpenCL buffers
        cl_int err = CL_SUCCESS;

        // Print device information
        if (g_verbose_enabled) {
            std::string deviceName = device.getInfo<CL_DEVICE_NAME>();
            std::string deviceVendor = device.getInfo<CL_DEVICE_VENDOR>();
            std::string deviceVersion = device.getInfo<CL_DEVICE_VERSION>();
            std::cout << "[OpenCL] Using device: " << deviceName << " from " << deviceVendor
                      << " (version: " << deviceVersion << ")" << std::endl;
        }

        // Input buffer
        cl::Buffer device_input(
            context,
            CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR,
            input_size,
            input_data.data(),
            &err
        );

        if (err != CL_SUCCESS) {
            std::cerr << "[OpenCL] Error creating input buffer: " << err << std::endl;
            return false;
        }

        // Output buffer
        cl::Buffer device_output(context, CL_MEM_WRITE_ONLY, output_size, nullptr, &err);

        if (err != CL_SUCCESS) {
            std::cerr << "[OpenCL] Error creating output buffer: " << err << std::endl;
            return false;
        }

        // Select kernel based on rotation angle
        std::string kernel_name;
        if (angle == 90) {
            kernel_name = "rotate_90_cw";
        } else if (angle == 180) {
            kernel_name = "rotate_180";
        } else if (angle == 270) {
            kernel_name = "rotate_270_cw";
        } else {
            // If angle is not supported, just copy data
            kernel_name = "process_image";
        }

        if (g_verbose_enabled) {
            std::cout << "[OpenCL] Using kernel: " << kernel_name << std::endl;
        }

        // Create and configure kernel
        cl::Kernel kernel(program, kernel_name.c_str(), &err);

        if (err != CL_SUCCESS) {
            std::cerr << "[OpenCL] Error creating kernel '" << kernel_name << "': " << err << std::endl;
            return false;
        }

        // Set kernel arguments
        err = kernel.setArg(0, device_input);
        err |= kernel.setArg(1, device_output);
        err |= kernel.setArg(2, w);
        err |= kernel.setArg(3, h);
        err |= kernel.setArg(4, components);

        if (err != CL_SUCCESS) {
            std::cerr << "[OpenCL] Error setting kernel arguments: " << err << std::endl;
            return false;
        }

        if (g_verbose_enabled) {
            std::cout << "[OpenCL] Executing kernel..." << std::endl;
        }

        // Execute the kernel
        err = queue.enqueueNDRangeKernel(
            kernel,
            cl::NullRange,
            cl::NDRange(w, h),
            cl::NullRange
        );

        if (err != CL_SUCCESS) {
            std::cerr << "[OpenCL] Error enqueueing kernel: " << err << std::endl;
            return false;
        }

        // Make sure all operations on the queue are finished
        queue.finish();

        if (g_verbose_enabled) {
            std::cout << "[OpenCL] Kernel execution completed" << std::endl;
        }

        // Read result
        std::vector<unsigned char> rotated_data(output_size);
        err = queue.enqueueReadBuffer(
            device_output,
            CL_TRUE,
            0,
            output_size,
            rotated_data.data()
        );

        if (err != CL_SUCCESS) {
            std::cerr << "[OpenCL] Error reading output buffer: " << err << std::endl;
            return false;
        }

        // Resize matrix if needed
        if (img_matrix->width != new_width || img_matrix->height != new_height) {
            if (g_verbose_enabled) {
                std::cout << "[OpenCL] Resizing matrix for rotated image: " << new_width << "x" << new_height << std::endl;
            }
            img_matrix->resize(new_width, new_height);
        }

        // Copy rotated data back into the matrix
        std::memcpy(img_matrix->get_arr_interlaced(), rotated_data.data(), rotated_data.size());

        std::cout << "[OpenCL] Image rotated successfully using " << (isGPU ? "GPU" : "CPU") << std::endl;
        return true;
    } catch (const std::exception& e) {
        std::cerr << "[OpenCL] Error during rotation: " << e.what() << std::endl;
        return false;
    }
}
