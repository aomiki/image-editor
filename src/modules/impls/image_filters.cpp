#include "image_filters.h"
#include <cmath>
#include <cstring>
#include <algorithm>

void grayscale(matrix &img) {
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

void gaussian_blur(matrix& img, float sigma) {
    if (img.components_num < 3) return;
    
    const int radius = static_cast<int>(3 * sigma);
    std::vector<unsigned char> temp(img.arr, img.arr + img.width * img.height * img.components_num);

    for (unsigned y = 0; y < img.height; ++y) {
        for (unsigned x = 0; x < img.width; ++x) {
            float r = 0, g = 0, b = 0;
            float weight_sum = 0;

            // Применяем гауссово ядро
            for (int dy = -radius; dy <= radius; ++dy) {
                for (int dx = -radius; dx <= radius; ++dx) {
                    int nx = std::clamp(static_cast<int>(x) + dx, 0, static_cast<int>(img.width)-1);
                    int ny = std::clamp(static_cast<int>(y) + dy, 0, static_cast<int>(img.height)-1);
                    
                    // Вычисляем вес
                    const float weight = exp(-(dx*dx + dy*dy)/(2 * sigma * sigma));
                    
                    const unsigned char* p = &temp[(ny * img.width + nx) * img.components_num];
                    r += p[0] * weight;
                    g += p[1] * weight;
                    b += p[2] * weight;
                    weight_sum += weight;
                }
            }

            // Нормализация и запись результата
            unsigned char* dst = img.get(x, y);
            dst[0] = static_cast<unsigned char>(r / weight_sum);
            dst[1] = static_cast<unsigned char>(g / weight_sum);
            dst[2] = static_cast<unsigned char>(b / weight_sum);
        }
    }
}