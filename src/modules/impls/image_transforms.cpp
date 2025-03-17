#include "image_transforms.h"
#include <cstring>
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

    unsigned char* newArr = new (std::nothrow) unsigned char[new_width * new_height * img.components_num];
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

    unsigned char* newArr = new (std::nothrow) unsigned char[new_width * new_height * img.components_num];

    for (unsigned y = 0; y < img.height; ++y) {
        for (unsigned x = 0; x < img.width; ++x) {
            unsigned old_idx = (y * img.width + x) * img.components_num;
            unsigned new_idx = 0;

            switch (angle) 
            {
                case 90:
                    new_idx = ((x * new_width) + (new_width - y - 1)) * img.components_num;
                    break;
                case 180:
                    new_idx = ((new_height - y - 1) * new_width + (new_width - x - 1)) * img.components_num;
                    break;
                case 270:
                    new_idx = ((new_height - x - 1) * new_width + y) * img.components_num;
                    break;
            }

            memcpy(&newArr[new_idx], &img.arr[old_idx], img.components_num);
        }
    }

    delete[] img.arr;
    img.set_arr_interlaced(newArr, new_width, new_height);
}

