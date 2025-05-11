#include "image_transforms.h"
#include "_shared_definitions.h"

void bilinear_interpolate(matrix& img, float x, float y, unsigned char* result) {
    const int max_x = static_cast<int>(img.width) - 1;
    const int max_y = static_cast<int>(img.height) - 1;

    x = clamp(x, 0.0f, static_cast<float>(max_x));
    y = clamp(y, 0.0f, static_cast<float>(max_y));

    int x0 = static_cast<int>(x);
    int y0 = static_cast<int>(y);
    int x1 = min(x0 + 1, max_x);  
    int y1 = min(y0 + 1, max_y); 

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
