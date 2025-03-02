#include "image_tools.h"

void matrix_gray::fill(unsigned char value)
{
    for (size_t i = 0; i < width; i++)
    {
        for (size_t j = 0; j < height; j++)
        {
            array.push_back(value);
        }
    }
}

void matrix_gray::set(unsigned x, unsigned y, unsigned char color)
{
    set(matrix_coord(x,y), color);
}

void matrix_gray::set(matrix_coord coord, unsigned char color)
{
    size_t index = width*coord.y+coord.x;
    array[index] = color;
}

unsigned char matrix_gray::get(unsigned x, unsigned y)
{
    size_t index = width*y+x;
    return array[index];
}

void matrix_rgb::fill(color_rgb value)
{
    for (size_t i = 0; i < width; i++)
    {
        for (size_t j = 0; j < height; j++)
        {
            array.push_back(value.red);
            array.push_back(value.green);
            array.push_back(value.blue);
        }
    }
}

void matrix_rgb::set(unsigned x, unsigned y, color_rgb color)
{
    set(matrix_coord(x,y), color);
};

void matrix_rgb::set(matrix_coord coord, color_rgb color)
{
    size_t index = (width*coord.y+coord.x)*3;

    (array)[index] = color.red;
    (array)[index+1] = color.green;
    (array)[index+2] = color.blue;
};

color_rgb matrix_rgb::get(unsigned x, unsigned y)
{
    size_t index = (width*y+x)*3;

    color_rgb color((array)[index], (array)[index+1], (array)[index+2]);

    return color;
};
