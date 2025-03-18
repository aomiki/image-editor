#include "image_edit.h"
#include "image_transforms.h"
#include <vector>
#include <string>


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
