#include "image_edit.h"
#include "image_transforms.h"
#include <vector>
#include <string>
#include <iostream>

void transform_image_crop(std::string filepath, image_codec* codec, unsigned crop_left, unsigned crop_top, unsigned crop_right, unsigned crop_bottom)
{
    std::cout << "Starting crop transformation of " << filepath << std::endl;
    std::vector<unsigned char> img_buffer;

    codec->load_image_file(&img_buffer, input_folder / filepath);
    std::cout << "Image loaded, buffer size: " << img_buffer.size() << std::endl;
    
    ImageInfo info = codec->read_info(&img_buffer);
    std::cout << "Image info: " << info.width << "x" << info.height << ", color scheme: " << info.colorScheme << std::endl;

    matrix* mat = nullptr;

    if (info.colorScheme == ImageColorScheme::IMAGE_RGB) 
    {
        mat = new matrix_rgb(info.width, info.height);
        std::cout << "Created RGB matrix" << std::endl;
    } 
    else if (info.colorScheme == ImageColorScheme::IMAGE_GRAY) 
    {
        mat = new matrix_gray(info.width, info.height);
        std::cout << "Created Grayscale matrix" << std::endl;
    }
    else if (info.colorScheme == ImageColorScheme::IMAGE_PALETTE) 
    {
        mat = new matrix_rgb(info.width, info.height);
        std::cout << "Created RGB matrix for palette image" << std::endl;
    }

    codec->decode(&img_buffer, mat, info.colorScheme, info.bit_depth);
    std::cout << "Image decoded to matrix" << std::endl;

    std::cout << "Applying crop: " << crop_left << ", " << crop_top << ", " 
              << crop_right << ", " << crop_bottom << std::endl;
    crop(*mat, crop_left, crop_top, crop_right, crop_bottom);
    std::cout << "Crop applied" << std::endl;

    img_buffer.clear();
    std::cout << "Encoding cropped image" << std::endl;
    codec->encode(&img_buffer, mat, info.colorScheme, info.bit_depth);
    std::cout << "Encoded buffer size: " << img_buffer.size() << std::endl;
    
    std::string output_path = (result_folder / "cropped_result").string();
    std::cout << "Saving to: " << output_path << std::endl;
    codec->save_image_file(&img_buffer, output_path);
    std::cout << "Image saved" << std::endl;

    //delete mat;
}

void transform_image_rotate(std::string filepath, image_codec* codec, unsigned angle)
{
    std::cout << "Starting rotation of " << filepath << " by " << angle << " degrees" << std::endl;
    std::vector<unsigned char> img_buffer;

    codec->load_image_file(&img_buffer, input_folder / filepath);
    std::cout << "Image loaded, buffer size: " << img_buffer.size() << std::endl;
    
    ImageInfo info = codec->read_info(&img_buffer);
    std::cout << "Image info: " << info.width << "x" << info.height << ", color scheme: " << info.colorScheme << std::endl;

    matrix* mat = nullptr;

    if (info.colorScheme == ImageColorScheme::IMAGE_RGB) 
    {
        mat = new matrix_rgb(info.width, info.height);
        std::cout << "Created RGB matrix" << std::endl;
    } 
    else if (info.colorScheme == ImageColorScheme::IMAGE_GRAY) 
    {
        mat = new matrix_gray(info.width, info.height);
        std::cout << "Created Grayscale matrix" << std::endl;
    }
    else if (info.colorScheme == ImageColorScheme::IMAGE_PALETTE) 
    {
        mat = new matrix_rgb(info.width, info.height);
        std::cout << "Created RGB matrix for palette image" << std::endl;
    }

    codec->decode(&img_buffer, mat, info.colorScheme, info.bit_depth);
    std::cout << "Image decoded to matrix" << std::endl;

    std::cout << "Applying rotation: " << angle << " degrees" << std::endl;
    rotate(*mat, angle); 
    std::cout << "Rotation applied, new dimensions: " << mat->width << "x" << mat->height << std::endl;

    img_buffer.clear();
    std::cout << "Encoding rotated image" << std::endl;
    codec->encode(&img_buffer, mat, info.colorScheme, info.bit_depth);
    std::cout << "Encoded buffer size: " << img_buffer.size() << std::endl;
    
    std::string output_path = (result_folder / "rotated_result").string();
    std::cout << "Saving to: " << output_path << std::endl;
    codec->save_image_file(&img_buffer, output_path);
    std::cout << "Image saved" << std::endl;

    //delete mat;  
}
