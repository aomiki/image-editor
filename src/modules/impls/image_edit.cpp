#include "image_edit.h"
#include "image_transforms.h"
#ifdef OPENCL_IMPL
#include "impls_hw_accel/opencl/image_codec_cl.h"
#endif
#include <vector>
#include <string>
#include <iostream>


void transform_image_crop(std::string filepath, image_codec* codec)
{
    std::vector<unsigned char> img_buffer;

    codec->load_image_file(&img_buffer, input_folder / filepath);
    ImageInfo info = codec->read_info(&img_buffer);

    matrix* mat = nullptr;

    if (info.colorScheme == ImageColorScheme::IMAGE_RGB) 
    {
        mat = new matrix_rgb(info.width, info.height);
    } 
    else if (info.colorScheme == ImageColorScheme::IMAGE_GRAY) 
    {
        mat = new matrix_gray(info.width, info.height);
    }
    else if (info.colorScheme == ImageColorScheme::IMAGE_PALETTE) 
    {
        mat = new matrix_rgb(info.width, info.height);
    }

    codec->decode(&img_buffer, mat, info.colorScheme, info.bit_depth);

    crop(*mat, 200, 200, 200, 200);

    img_buffer.clear();
    codec->encode(&img_buffer, mat, info.colorScheme, info.bit_depth);
    codec->save_image_file(&img_buffer, result_folder / "cropped_result");

    //delete mat;
}

void transform_image_rotate(std::string filepath, image_codec* codec, unsigned angle)
{
    if (g_verbose_enabled) {
        std::cout << "transform_image_rotate with " << (g_force_gpu_enabled ? "force GPU" : "normal") << " mode" << std::endl;
    }
    
    std::vector<unsigned char> img_buffer;

    codec->load_image_file(&img_buffer, input_folder / filepath);
    ImageInfo info = codec->read_info(&img_buffer);

    matrix* mat = nullptr;

    if (info.colorScheme == ImageColorScheme::IMAGE_RGB) 
    {
        mat = new matrix_rgb(info.width, info.height);
    } 
    else if (info.colorScheme == ImageColorScheme::IMAGE_GRAY) 
    {
        mat = new matrix_gray(info.width, info.height);
    }
    else if (info.colorScheme == ImageColorScheme::IMAGE_PALETTE) 
    {
        mat = new matrix_rgb(info.width, info.height);
    }

    codec->decode(&img_buffer, mat, info.colorScheme, info.bit_depth);

#ifdef OPENCL_IMPL
    if (g_verbose_enabled) {
        std::cout << "Using OpenCL implementation for rotation" << std::endl;
    }
    
    // Create OpenCL codec for GPU processing
    image_codec_cl cl_codec;
    
    // Try direct GPU rotation
    if (cl_codec.rotate_on_gpu(mat, angle)) {
        if (g_verbose_enabled) {
            std::cout << "Successfully rotated using OpenCL codec directly" << std::endl;
        }
    } else {
        // If direct rotation failed, use the global rotate function (which might still use OpenCL)
        if (g_verbose_enabled) {
            std::cout << "Direct OpenCL rotation failed, using global rotate function" << std::endl;
        }
        rotate(*mat, angle);
    }
#else
    // Regular CPU rotation
    if (g_verbose_enabled) {
        std::cout << "Using CPU implementation for rotation" << std::endl;
    }
    rotate(*mat, angle); 
#endif

    img_buffer.clear();
    codec->encode(&img_buffer, mat, info.colorScheme, info.bit_depth);
    codec->save_image_file(&img_buffer, result_folder / "rotated_result");

    //delete mat;  
}
