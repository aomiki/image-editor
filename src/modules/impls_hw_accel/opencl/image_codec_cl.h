#pragma once

#include <CL/opencl.hpp>
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

    bool initializeOpenCL();

public:
    image_codec_cl();
    ~image_codec_cl();

    ImgFormat native_format() const {
        return ImgFormat::PNG;
    }

    ImageInfo read_info(std::vector<unsigned char>* img_buffer);
    void encode(std::vector<unsigned char>* img_buffer, matrix* img_matrix, ImageColorScheme colorScheme, unsigned bit_depth);
    void decode(std::vector<unsigned char>* img_source, matrix* img_matrix, ImageColorScheme colorScheme, unsigned bit_depth);
    void load_image_file(std::vector<unsigned char>* png_buffer, std::string image_filepath);
    void save_image_file(std::vector<unsigned char>* png_buffer, std::string image_filepath);

    // Function for rotating an image on the GPU
    bool rotate_on_gpu(matrix* img_matrix, unsigned angle);
}; 