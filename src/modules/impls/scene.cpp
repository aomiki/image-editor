#include "scene.h"
#include <cstring>
#include <iostream>
#include "image_transforms.h"
#include "image_edit.h" // For global flags

// Global flags
extern bool g_force_gpu_enabled;
extern bool g_verbose_enabled;

#ifdef OPENCL_IMPL
#include "impls_hw_accel/opencl/image_codec_cl.h"
#endif

scene::scene()
{
    std::cout << "Initializing scene..." << std::endl;

#ifdef OPENCL_IMPL
    std::cout << "OpenCL support is enabled" << std::endl;
    if (g_force_gpu_enabled) {
        std::cout << "GPU mode is enabled, trying to create OpenCL codec..." << std::endl;
        try {
            codec = new image_codec_cl();
            std::cout << "Successfully created OpenCL implementation for image codec" << std::endl;
        } catch (const std::exception& e) {
            std::cout << "Error creating OpenCL codec: " << e.what() << std::endl;
            std::cout << "Falling back to standard implementation" << std::endl;
            codec = new image_codec();
        } catch (...) {
            std::cout << "Unknown error creating OpenCL codec, falling back to standard implementation" << std::endl;
            codec = new image_codec();
        }
    } else {
        std::cout << "Using standard implementation for image codec (GPU not forced)" << std::endl;
        codec = new image_codec();
    }
#else
    std::cout << "Using standard implementation for image codec (OpenCL not enabled)" << std::endl;
    codec = new image_codec();
#endif

    std::cout << "Initializing other scene resources..." << std::endl;
    img_matrix = nullptr;
    img_buffer = nullptr;
    std::cout << "Scene initialization complete" << std::endl;
}

scene::~scene()
{
    delete codec;
    if (img_matrix != nullptr)
    {
        delete img_matrix;
    }
}

long unsigned scene::get_img_binary_size()
{
    if (img_buffer == nullptr) {
        std::cerr << "Warning: img_buffer is null in get_img_binary_size" << std::endl;
        return 0;
    }
    return img_buffer->size();
}

void scene::get_img_binary(unsigned char* img_buffer)
{
    if (this->img_buffer == nullptr) {
        std::cerr << "Warning: img_buffer is null in get_img_binary" << std::endl;
        return;
    }
    std::memcpy(img_buffer, this->img_buffer->data(), this->img_buffer->size());
}

ImageInfo scene::get_img_info()
{
    if (img_buffer == nullptr) {
        std::cerr << "Warning: img_buffer is null in get_img_info" << std::endl;
        ImageInfo empty_info = {ImageColorScheme::IMAGE_RGB, 8, 0, 0};
        return empty_info;
    }
    return codec->read_info(img_buffer);
}

void scene::load_image_file(std::string inp_filename)
{
    if (img_buffer != nullptr)
    {
        delete img_buffer;
    }

    img_buffer = new std::vector<unsigned char>();

    codec->load_image_file(img_buffer, inp_filename);
}

void scene::save_image_file(std::string inp_filename)
{
    codec->save_image_file(img_buffer, inp_filename);
}

void scene::decode()
{
    ImageInfo img_info = codec->read_info(img_buffer);

    //resize matrix if needed
    if (img_matrix != nullptr && (
        img_info.width != img_matrix->width ||
        img_info.height != img_matrix->height ||
        img_info.colorScheme != colorScheme
    ))
    {
        if(img_info.colorScheme == colorScheme)
        {
            img_matrix->resize(img_info.width, img_info.height);
        }
        else
        {
            delete img_matrix;
            img_matrix = nullptr;
        }
    }

    //allocate matrix if needed
    if (img_matrix == nullptr)
    {
        colorScheme = img_info.colorScheme;
        switch (colorScheme)
        {
            case IMAGE_GRAY:
                img_matrix = new matrix_gray(img_info.width, img_info.height);
                break;
            case IMAGE_RGB:
                img_matrix = new matrix_rgb(img_info.width, img_info.height);
                break;
            default:
                break;
        }
    }

    codec->decode(img_buffer, img_matrix, colorScheme, 8);
}

void scene::encode()
{
    codec->encode(img_buffer, img_matrix, colorScheme, 8);
}

void scene::rotate(float angle)
{
    ::rotate(*img_matrix, angle);
}

void scene::crop(unsigned crop_left, unsigned crop_top, unsigned crop_right, unsigned crop_bottom)
{
    ::crop(*img_matrix, crop_left, crop_top, crop_right, crop_bottom);
}

void scene::reflect(bool horizontal, bool vertical)
{
    ::reflect(*img_matrix, horizontal, vertical);
}

void scene::shear(float shx, float shy)
{
    ::shear(*img_matrix, shx, shy);
}
