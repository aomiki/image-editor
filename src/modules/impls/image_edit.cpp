#include "image_edit.h"
#include "image_transforms.h"
#include <vector>
#include <string>


void transform_image_crop(std::string filepath, image_codec* codec, 
                         unsigned int crop_left, unsigned int crop_top,
                         unsigned int crop_right, unsigned int crop_bottom)
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

    crop(*mat, crop_left, crop_top, crop_right, crop_bottom);

    img_buffer.clear();
    codec->encode(&img_buffer, mat, info.colorScheme, info.bit_depth);
    codec->save_image_file(&img_buffer, result_folder / "cropped_result");

    //delete mat;
}

void transform_image_rotate(std::string filepath, image_codec* codec, unsigned angle)
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

    rotate(*mat, angle); 

    img_buffer.clear();
    codec->encode(&img_buffer, mat, info.colorScheme, info.bit_depth);
    codec->save_image_file(&img_buffer, result_folder / "rotated_result");

    //delete mat;  
}

void transform_image_reflect(std::string filepath, image_codec* codec, bool horizontal, bool vertical)
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

    reflect(*mat, horizontal, vertical);

    img_buffer.clear();
    codec->encode(&img_buffer, mat, info.colorScheme, info.bit_depth);
    codec->save_image_file(&img_buffer, result_folder / "reflect_result");

    //delete mat;  
}

void transform_image_shear(std::string filepath, image_codec* codec, float shx, float shy)
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

    shear(*mat, shx, shy);
    img_buffer.clear();
    codec->encode(&img_buffer, mat, info.colorScheme, info.bit_depth);
    codec->save_image_file(&img_buffer, result_folder / "shear_result");
    // delete mat;
}