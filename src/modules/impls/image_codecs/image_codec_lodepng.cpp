#include "image_codec.h"
#include "lodepng.h"

inline LodePNGColorType ColorSchemeToLodeColorType(ImageColorScheme colorscheme)
{
    switch (colorscheme)
    {
        case ImageColorScheme::IMAGE_GRAY :
            return LodePNGColorType::LCT_GREY;
        case ImageColorScheme::IMAGE_RGB :
            return LodePNGColorType::LCT_RGB;
        default:
            return LodePNGColorType::LCT_MAX_OCTET_VALUE;
    }
}

image_codec::image_codec()
{}

void image_codec::encode(std::vector<unsigned char>* img_source, matrix* img_matrix, ImageColorScheme colorScheme, unsigned bit_depth)
{
    lodepng::encode(*img_source, img_matrix->array, img_matrix->width, img_matrix->height, ColorSchemeToLodeColorType(colorScheme), bit_depth);
}

void image_codec::decode(std::vector<unsigned char>* img_source, matrix* img_matrix, ImageColorScheme colorScheme, unsigned bit_depth)
{
    lodepng::decode(img_matrix->array, img_matrix->width, img_matrix->height, *img_source, ColorSchemeToLodeColorType(colorScheme), bit_depth);
};

void image_codec::load_image_file(std::vector<unsigned char>* png_buffer, std::string image_filepath)
{
    lodepng::load_file(*png_buffer, image_filepath);
}

void image_codec::save_image_file(std::vector<unsigned char>* png_buffer, std::string image_filepath)
{
    lodepng::save_file(*png_buffer, image_filepath+".png");
}

image_codec::~image_codec()
{}