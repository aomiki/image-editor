#include "image_codec.h"

// Default implementation of virtual methods
ImageInfo image_codec::read_info(std::vector<unsigned char>* img_buffer) {
    ImageInfo info;
    info.width = 0;
    info.height = 0;
    info.colorScheme = IMAGE_RGB;
    info.bit_depth = 8;
    return info;
}

void image_codec::encode(std::vector<unsigned char>* img_buffer, matrix* img_matrix, ImageColorScheme colorScheme, unsigned bit_depth) {
    // Default implementation does nothing
}

void image_codec::decode(std::vector<unsigned char>* img_source, matrix* img_matrix, ImageColorScheme colorScheme, unsigned bit_depth) {
    // Default implementation does nothing
}

void image_codec::load_image_file(std::vector<unsigned char>* png_buffer, std::string image_filepath) {
    // Default implementation does nothing
}

void image_codec::save_image_file(std::vector<unsigned char>* png_buffer, std::string image_filepath) {
    // Default implementation does nothing
} 