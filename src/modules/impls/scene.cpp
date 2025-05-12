#include "scene.h"
#include <cstring>
#include "image_transforms.h"
#include "image_filters.h"
scene::scene()
{
    codec = new image_codec();
    img_matrix = nullptr;
    img_buffer = nullptr;
}

scene::~scene()
{
    delete codec;
    if (img_matrix != nullptr)
    {
        delete img_matrix;
    }
}

long unsigned scene::get_img_binary_size()
{
    return img_buffer->size();
}

void scene::get_img_binary(unsigned char* img_buffer)
{
    std::memcpy(img_buffer, this->img_buffer->data(), this->img_buffer->size());
}

ImageInfo scene::get_img_info()
{
    return codec->read_info(img_buffer);
}

void scene::load_image_file(std::string inp_filename)
{
    if (img_buffer != nullptr)
    {
        delete img_buffer;
    }

    img_buffer = new std::vector<unsigned char>();

    codec->load_image_file(img_buffer, inp_filename);
}

void scene::save_image_file(std::string inp_filename)
{
    codec->save_image_file(img_buffer, inp_filename);
}

void scene::decode()
{
    ImageInfo img_info = codec->read_info(img_buffer);

    //resize matrix if needed
    if (img_matrix != nullptr && (
        img_info.width != img_matrix->width ||
        img_info.height != img_matrix->height ||
        img_info.colorScheme != colorScheme
    ))
    {
        if(img_info.colorScheme == colorScheme)
        {
            img_matrix->resize(img_info.width, img_info.height);
        }
        else
        {
            delete img_matrix;
            img_matrix = nullptr;
        }
    }

    //allocate matrix if needed
    if (img_matrix == nullptr)
    {
        colorScheme = img_info.colorScheme;
        switch (colorScheme)
        {
            case IMAGE_GRAY:
                img_matrix = new matrix_gray(img_info.width, img_info.height);
                break;
            case IMAGE_RGB:
                img_matrix = new matrix_rgb(img_info.width, img_info.height);
                break;
            default:
                break;
        }
    }

    codec->decode(img_buffer, img_matrix, colorScheme, 8);
}

void scene::encode()
{
    codec->encode(img_buffer, img_matrix, colorScheme, 8);
}

void scene::rotate(float angle)
{
    ::rotate(*img_matrix, angle);
}

void scene::crop(unsigned crop_left, unsigned crop_top, unsigned crop_right, unsigned crop_bottom)
{
    ::crop(*img_matrix, crop_left, crop_top, crop_right, crop_bottom);
}

void scene::reflect(bool horizontal, bool vertical)
{
    ::reflect(*img_matrix, horizontal, vertical);
}

void scene::shear(float shx, float shy)
{
    ::shear(*img_matrix, shx, shy);
}
void scene::grayscale() {
        ::grayscale(*img_matrix);
}
void scene::gaussian_blur(float sigma) {
    if (img_matrix) {
        ::gaussian_blur(*img_matrix, sigma);
    }
}