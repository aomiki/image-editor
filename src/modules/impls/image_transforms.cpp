#include "image_transforms.h"
#include <cstring>
#include <cblas.h>  
#include <cmath>
#include <algorithm>
using namespace std;

void crop(matrix& img, unsigned crop_left, unsigned crop_top, unsigned crop_right, unsigned crop_bottom) 
{
    unsigned new_width = img.width - crop_left - crop_right;
    unsigned new_height = img.height - crop_top - crop_bottom;

    
    if (new_width <= 0 || new_height <= 0) 
    {
        return;
    }

    if (crop_left + crop_right > img.width || crop_top + crop_bottom > img.height) 
    {
        return;
    }

    unsigned char* newArr = new unsigned char[new_width * new_height * img.components_num];
    unsigned char* src = img.arr + (crop_top * img.width + crop_left) * img.components_num;
    unsigned char* dst = newArr;

    for (unsigned i = 0; i < new_height; ++i) 
    {
        memcpy(dst, src, new_width * img.components_num);
        src += img.width * img.components_num;  
        dst += new_width * img.components_num;  
    }

    
    delete[] img.arr;

    img.set_arr_interlaced(newArr, new_width, new_height);

}


void rotate(matrix& img, unsigned angle) 
{
    angle = angle % 360;  
    if (angle == 0) return; 

    unsigned new_width = (angle == 90 || angle == 270) ? img.height : img.width;
    unsigned new_height = (angle == 90 || angle == 270) ? img.width : img.height;

    unsigned char* newArr = new unsigned char[new_width * new_height * img.components_num];

    for (unsigned y = 0; y < img.height; ++y) {
        for (unsigned x = 0; x < img.width; ++x) {
            unsigned char* old_pixel = img.get(x, y); 
            unsigned char* new_pixel = nullptr;  
            
            switch (angle) {
                case 90:
                    new_pixel = &newArr[(x * new_width + (new_width - y - 1)) * img.components_num];
                    break;
                case 180:
                    new_pixel = &newArr[((new_height - y - 1) * new_width + (new_width - x - 1)) * img.components_num];
                    break;
                case 270:
                    new_pixel = &newArr[((new_height - x - 1) * new_width + y) * img.components_num];
                    break;
            }

            memcpy(new_pixel, old_pixel, img.components_num);
        }
    }

    delete[] img.arr;
    img.set_arr_interlaced(newArr, new_width, new_height);
}


void reflect(matrix& img, bool horizontal, bool vertical) 
{
    if (!horizontal && !vertical) return;

    unsigned char* newArr = new unsigned char[img.width * img.height * img.components_num];

    for (unsigned y = 0; y < img.height; ++y) {
        for (unsigned x = 0; x < img.width; ++x) {
            unsigned target_x = vertical ? (img.width - 1 - x) : x;
            unsigned target_y = horizontal ? (img.height - 1 - y) : y;
            
            unsigned char* src = img.get(x, y);
            unsigned char* dst = &newArr[(target_y * img.width + target_x) * img.components_num];
            memcpy(dst, src, img.components_num);
        }
    }

    delete[] img.arr;
    img.arr = newArr;
}
void bilinear_interpolate(matrix& img, float x, float y, unsigned char* result) {
    const int max_x = static_cast<int>(img.width) - 1;
    const int max_y = static_cast<int>(img.height) - 1;
    
    x = std::clamp(x, 0.0f, static_cast<float>(max_x));
    y = std::clamp(y, 0.0f, static_cast<float>(max_y));

    int x0 = static_cast<int>(x);
    int y0 = static_cast<int>(y);
    int x1 = std::min(x0 + 1, max_x);  
    int y1 = std::min(y0 + 1, max_y); 

    float dx = x - x0;
    float dy = y - y0;
    float w00 = (1-dx)*(1-dy);
    float w10 = dx*(1-dy);
    float w01 = (1-dx)*dy;
    float w11 = dx*dy;

    for (int c = 0; c < img.components_num; ++c) {
        float v00 = img.get(x0, y0)[c];
        float v10 = img.get(x1, y0)[c];
        float v01 = img.get(x0, y1)[c];
        float v11 = img.get(x1, y1)[c];
        result[c] = static_cast<unsigned char>(v00*w00 + v10*w10 + v01*w01 + v11*w11);
    }
}
void shear(matrix& img, float shx, float shy) {
    unsigned new_width = static_cast<unsigned>(img.width + 2*std::abs(shy)*img.height);
    unsigned new_height = static_cast<unsigned>(img.height + 2*std::abs(shx)*img.width);

    unsigned char* newArr = new unsigned char[new_width * new_height * img.components_num]();
    std::fill(newArr, newArr + new_width*new_height*img.components_num, 255);

    float center_x = new_width / 2.0f;
    float center_y = new_height / 2.0f;
    float img_center_x = img.width / 2.0f;
    float img_center_y = img.height / 2.0f;

    for (unsigned y = 0; y < new_height; ++y) {
        for (unsigned x = 0; x < new_width; ++x) {
            float src_x = img_center_x + (x - center_x) - shx*(y - center_y);
            float src_y = img_center_y + (y - center_y) - shy*(x - center_x);

            if (src_x >= 0 && src_x < img.width && src_y >= 0 && src_y < img.height) {
                bilinear_interpolate(img, src_x, src_y, &newArr[(y*new_width + x)*img.components_num]);
            }
        }
    }

    delete[] img.arr;
    img.set_arr_interlaced(newArr, new_width, new_height);
}

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