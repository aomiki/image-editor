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

inline ImageColorScheme LodePNGColorTypeToImageColorScheme(LodePNGColorType color_type)
{
    switch (color_type)
    {
        case LodePNGColorType::LCT_RGBA:
            return ImageColorScheme::IMAGE_RGB;
        case LodePNGColorType::LCT_RGB:
            return ImageColorScheme::IMAGE_RGB;
        case LodePNGColorType::LCT_GREY:
            return ImageColorScheme::IMAGE_GRAY;
    }
}

image_codec::image_codec()
{}

ImageInfo image_codec::read_info(std::vector<unsigned char>* img_buffer)
{
    LodePNGState state;
    lodepng_state_init(&state);

    ImageInfo img_info;

    lodepng_inspect(&img_info.width, &img_info.height, &state, img_buffer->data(), img_buffer->size());

    img_info.colorScheme = LodePNGColorTypeToImageColorScheme(state.info_png.color.colortype);
    img_info.bit_depth = state.info_png.color.bitdepth;

    return img_info;
}

void image_codec::encode(std::vector<unsigned char>* img_buffer, matrix* img_matrix, ImageColorScheme colorScheme, unsigned bit_depth)
{
    unsigned char** result_img = new unsigned char*;
    size_t* result_size = new size_t;
    lodepng_encode_memory(result_img, result_size, img_matrix->get_arr_interlaced(), img_matrix->width, img_matrix->height, ColorSchemeToLodeColorType(colorScheme), bit_depth);
    img_buffer->assign(*result_img, *result_img + *result_size);

    delete [] *result_img;
    delete result_img;

    delete result_size;
}

void image_codec::decode(std::vector<unsigned char>* img_source, matrix* img_matrix, ImageColorScheme colorScheme, unsigned bit_depth)
{
    if (img_matrix->size_interlaced() == 0)
    {
        return;
    }

    unsigned char** matrix_buffer = new unsigned char*;
    unsigned int* result_w = new unsigned int;
    unsigned int* result_h = new unsigned int;
    
    lodepng_decode_memory(matrix_buffer, result_w, result_h, img_source->data(),  img_source->size(), ColorSchemeToLodeColorType(colorScheme), bit_depth);
    img_matrix->set_arr_interlaced(*matrix_buffer, img_matrix->width, img_matrix->height);

    delete result_w;
    delete result_h;
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