#include "image_tools.h"

void matrix::fill(unsigned char *value)
{
    for (size_t i = 0; i < width; i++)
    {
        for (size_t j = 0; j < height; j++)
        {
            unsigned char* cell = get(i, j);
            for (size_t k = 0; k < components_num; k++)
            {
                cell[k] = value[k];
            }
        }
    }
}
