#include "image_tools.h"

void matrix::fill(unsigned char *value) {
    if (!value) return;
    
    for (size_t i = 0; i < size(); i++) {
        arr[i] = value[0];
    }
} 