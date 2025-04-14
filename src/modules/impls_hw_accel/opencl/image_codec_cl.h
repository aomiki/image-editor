#pragma once

#include "image_tools.h"
#include <vector>
#include <string>

class image_codec_cl {
public:
    image_codec_cl();
    ~image_codec_cl();

    bool load_image_file(std::vector<unsigned char>* image, const std::string& filename);
    bool read_info(std::vector<unsigned char>* image);
    bool decode(std::vector<unsigned char>* image, matrix* mat, ImageColorScheme scheme, unsigned int components);
    bool encode(std::vector<unsigned char>* image, matrix* mat, ImageColorScheme scheme, unsigned int components);
    bool save_image_file(std::vector<unsigned char>* image, const std::string& filename);

private:
    unsigned width;
    unsigned height;
};
