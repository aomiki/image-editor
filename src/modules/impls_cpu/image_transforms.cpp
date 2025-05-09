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
