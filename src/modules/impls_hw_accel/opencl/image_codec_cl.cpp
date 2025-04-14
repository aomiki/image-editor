#include "image_codec_cl.h"
#include "image_ops_cl.h"
#include <lodepng.h>
#include <iostream>

image_codec_cl::image_codec_cl() : width(0), height(0) {}

image_codec_cl::~image_codec_cl() {}

bool image_codec_cl::load_image_file(std::vector<unsigned char>* image, const std::string& filename) {
    unsigned error = lodepng::decode(*image, width, height, filename);
    if (error) {
        std::cerr << "Error loading image: " << lodepng_error_text(error) << std::endl;
        return false;
    }
    return true;
}

bool image_codec_cl::read_info(std::vector<unsigned char>* image) {
    // For OpenCL implementation, we'll use the image dimensions
    // that were set during load_image_file
    return true;
}

bool image_codec_cl::decode(std::vector<unsigned char>* image, matrix* mat, ImageColorScheme scheme, unsigned int components) {
    if (!image || !mat) return false;
    
    // Create matrix with correct dimensions
    mat->resize(height, width * components);
    
    // Copy data from image to matrix
    for (unsigned y = 0; y < height; y++) {
        for (unsigned x = 0; x < width; x++) {
            unsigned char* pixel = mat->get(x, y);
            for (unsigned c = 0; c < components; c++) {
                pixel[c] = (*image)[(y * width + x) * components + c];
            }
        }
    }
    
    return true;
}

bool image_codec_cl::encode(std::vector<unsigned char>* image, matrix* mat, ImageColorScheme scheme, unsigned int components) {
    if (!image || !mat) return false;
    
    // Resize image vector to hold all pixels
    image->resize(height * width * components);
    
    // Copy data from matrix to image
    for (unsigned y = 0; y < height; y++) {
        for (unsigned x = 0; x < width; x++) {
            unsigned char* pixel = mat->get(x, y);
            for (unsigned c = 0; c < components; c++) {
                (*image)[(y * width + x) * components + c] = pixel[c];
            }
        }
    }
    
    return true;
}

bool image_codec_cl::save_image_file(std::vector<unsigned char>* image, const std::string& filename) {
    unsigned error = lodepng::encode(filename, *image, width, height);
    if (error) {
        std::cerr << "Error saving image: " << lodepng_error_text(error) << std::endl;
        return false;
    }
    return true;
} 