#include "image_transforms.h"
#include <cstring>
#include <cblas.h>  
#include <cmath>
#include <algorithm>
using namespace std;

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