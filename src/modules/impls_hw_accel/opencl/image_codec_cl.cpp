#include "image_codec.h"
#include "image_ops_cl.h"
#include <lodepng.h>
#include <iostream>

class image_codec_cl : public image_codec {
private:
    unsigned width;
    unsigned height;

public:
    image_codec_cl() : image_codec(), width(0), height(0) {}
    ~image_codec_cl() {}

    void load_image_file(std::vector<unsigned char>* png_buffer, std::string image_filepath) {
        unsigned error = lodepng::decode(*png_buffer, width, height, image_filepath);
        if (error) {
            std::cerr << "Error loading image: " << lodepng_error_text(error) << std::endl;
        }
    }

    ImageInfo read_info(std::vector<unsigned char>* img_buffer) {
        ImageInfo info;
        info.width = width;
        info.height = height;
        info.colorScheme = IMAGE_RGB;  // Default to RGB
        info.bit_depth = 8;  // Default to 8-bit
        return info;
    }

    void decode(std::vector<unsigned char>* img_source, matrix* img_matrix, ImageColorScheme colorScheme, unsigned bit_depth) {
        if (!img_source || !img_matrix) return;
        
        // Resize matrix to match image dimensions
        img_matrix->resize(width, height);
        
        // Copy data from image to matrix
        for (unsigned int y = 0; y < height; y++) {
            for (unsigned int x = 0; x < width; x++) {
                unsigned char* pixel = img_matrix->get(x, y);
                for (unsigned int c = 0; c < 3; c++) {  // Always 3 components for RGB
                    pixel[c] = (*img_source)[(y * width + x) * 3 + c];
                }
            }
        }
    }

    void encode(std::vector<unsigned char>* img_buffer, matrix* img_matrix, ImageColorScheme colorScheme, unsigned bit_depth) {
        if (!img_buffer || !img_matrix) return;
        
        // Resize image vector to match matrix dimensions
        img_buffer->resize(width * height * 3);  // Always 3 components for RGB
        
        // Copy data from matrix to image
        for (unsigned int y = 0; y < height; y++) {
            for (unsigned int x = 0; x < width; x++) {
                unsigned char* pixel = img_matrix->get(x, y);
                for (unsigned int c = 0; c < 3; c++) {  // Always 3 components for RGB
                    (*img_buffer)[(y * width + x) * 3 + c] = pixel[c];
                }
            }
        }
    }

    void save_image_file(std::vector<unsigned char>* png_buffer, std::string image_filepath) {
        unsigned error = lodepng::encode(image_filepath, *png_buffer, width, height);
        if (error) {
            std::cerr << "Error saving image: " << lodepng_error_text(error) << std::endl;
        }
    }
}; 