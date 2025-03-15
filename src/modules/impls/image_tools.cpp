#include "image_tools.h"
#include <cstring>

matrix::matrix(unsigned int components_num, unsigned width, unsigned height)
{
    this->height = 0;
    this->width = 0;
    this->components_num = components_num;

    resize(width, height);
}

matrix::matrix(unsigned int components_num)
{
    this->height = 0;
    this->width = 0;
    this->components_num = components_num;
}

void matrix::resize(unsigned width, unsigned height)
{
    unsigned int old_size = size_interlaced();

    this->width = width;
    this->height = height;

    unsigned char* newArr = new unsigned char[size_interlaced()];

    if (old_size != 0)
    {
        std::memcpy(newArr, arr,  old_size);
        delete [] arr;
    }

    arr = newArr;
}

void matrix_gray::element_to_c_arr(unsigned char* buffer, unsigned char value)
{
    buffer[0] = value;
}

unsigned char matrix_gray::c_arr_to_element(unsigned char *buffer)
{
    return buffer[0];
}


void matrix_rgb::element_to_c_arr(unsigned char* buffer, color_rgb value)
{
    buffer[0] = value.red;
    buffer[1] = value.green;
    buffer[2] = value.blue;
}

color_rgb matrix_rgb::c_arr_to_element(unsigned char *buffer)
{
    return color_rgb(buffer[0], buffer[1], buffer[2]);
}
