#include "image_transforms.h"
#include <cstring>
#include <cblas.h>  
#include <cmath>
#include <algorithm>

void rotate(matrix& img, float angle) {
    float radians = angle * M_PI / 180.0f;
    float cos_theta = cos(radians);
    float sin_theta = sin(radians);
    
    float R[4] = {
        cos_theta, -sin_theta,
        sin_theta,  cos_theta
    };
    
    float half_height = img.height * 0.5f;
    float half_width = img.width * 0.5f;
    
    float corners[4][2] = {
        {-half_width, -half_height},
        { half_width, -half_height},
        {-half_width,  half_height},
        { half_width,  half_height}
    };
    
    float rotated_corners[8];
    cblas_sgemm(CblasRowMajor, CblasNoTrans, CblasNoTrans,
                4, 2, 2, 1.0f, 
                &corners[0][0], 2,
                R, 2,
                0.0f, rotated_corners, 2);
    
    float min_x = rotated_corners[0], max_x = rotated_corners[0];
    float min_y = rotated_corners[1], max_y = rotated_corners[1];
    
    for (int i = 1; i < 4; ++i) {
        min_x = std::min(min_x, rotated_corners[i*2]);
        max_x = std::max(max_x, rotated_corners[i*2]);
        min_y = std::min(min_y, rotated_corners[i*2+1]);
        max_y = std::max(max_y, rotated_corners[i*2+1]);
    }
    
    unsigned new_width = static_cast<unsigned>(round(max_x - min_x));
    unsigned new_height = static_cast<unsigned>(round(max_y - min_y));
    
    unsigned char* newArr = new unsigned char[new_width * new_height * img.components_num];
    
    float new_half_width = new_width * 0.5f;
    float new_half_height = new_height * 0.5f;
    
    float R_inv[4] = {
        cos_theta,  sin_theta,
        -sin_theta, cos_theta
    };
    
    for (unsigned y = 0; y < new_height; ++y) {
        for (unsigned x = 0; x < new_width; ++x) {
            float coords[2] = {
                x - new_half_width,
                y - new_half_height
            };
            
            float src_coords[2];
            cblas_sgemv(CblasRowMajor, CblasNoTrans,
                       2, 2, 1.0f,
                       R_inv, 2,
                       coords, 1,
                       0.0f, src_coords, 1);
            
            float src_x = src_coords[0] + half_width;
            float src_y = src_coords[1] + half_height;
            
            if (src_x >= 0 && src_x <= img.width-1 && 
                src_y >= 0 && src_y <= img.height-1) {
                unsigned char* pixel = &newArr[(y * new_width + x) * img.components_num];
                bilinear_interpolate(img, src_x, src_y, pixel);
            } else {
                unsigned char* pixel = &newArr[(y * new_width + x) * img.components_num];
                memset(pixel, 255, img.components_num);
            }
        }
    }
    delete[] img.arr;
    img.set_arr_interlaced(newArr, new_width, new_height);
}