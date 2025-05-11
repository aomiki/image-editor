#include "image_filters.h"

void greyscale(matrix &img) {
    const float r_coeff = 0.2126f;
    const float g_coeff = 0.7152f;
    const float b_coeff = 0.0722f;

    if (img.components_num < 3) return;

    for (unsigned y = 0; y < img.height; ++y) {
        for (unsigned x = 0; x < img.width; ++x) {

            unsigned char* pixel = img.get(x, y);
            
            float gray_float = 
                pixel[0] * r_coeff +  // R 
                pixel[1] * g_coeff +  // G 
                pixel[2] * b_coeff;   // B 
            
            unsigned char gray = static_cast<unsigned char>(gray_float + 0.5f);
            
            memset(pixel, gray, 3);
        }
    }
}