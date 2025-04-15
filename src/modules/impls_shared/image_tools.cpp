#include "image_tools.h"

__shared_func__ matrix_coord::matrix_coord(unsigned x, unsigned y)
{
    this->x = x;
    this->y = y;
}

__shared_func__ unsigned int matrix::get_interlaced_index(unsigned int x, unsigned int y)
{
    return (width*y+x)*components_num;
}

__shared_func__ unsigned char* matrix::get(unsigned int x, unsigned int y)
{
    return arr + get_interlaced_index(x, y);
}

__shared_func__ unsigned char* matrix::get_arr_interlaced()
{
    return arr;
}

__shared_func__ unsigned int matrix::size_interlaced()
{
    return width * height * components_num;
}

__shared_func__ unsigned int matrix::size()
{
    return width * height;
}

__shared_func__ void matrix::set_arr_interlaced(unsigned char *arr, unsigned width, unsigned height)
{
    this->arr = arr;
    this->width = width;
    this->height = height;
}

__shared_func__ void matrix::set_arr_interlaced(unsigned char *arr)
{
    this->arr = arr;
}