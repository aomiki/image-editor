#include "image_codec.h"
#include "image_tools.h"
#include <boost/program_options.hpp>
#include <filesystem>
#include <iostream>
#include <string>
#include <vector>
#include "cmd_parser.hpp"
namespace fs = std::filesystem;

const fs::path result_folder("output");
const fs::path input_folder("input");


void decode_encode_img(std::string filepath, image_codec *codec);

namespace po = boost::program_options;
int main(int ac, char *av[]) {
    std::cout << "Shellow from SSAU!" << std::endl;

    try {
        po::variables_map vm = parse_arguments(ac, av);

    } catch (std::exception &e) {
        std::cerr << "error: " << e.what() << "\n";
        return 1;
    } catch (...) {
        std::cerr << "Exception of unknown type!\n";
    }


}

/// @brief Draws border around an image
template <typename E>
void draw_border(matrix_color<E> &img_matrix, E border_color) {
    unsigned int vert_boundary = (int)img_matrix.height / 10;
    unsigned int horiz_boundary = (int)img_matrix.width / 10;

    for (size_t i = 0; i < img_matrix.height; i++) {
        for (size_t j = 0; j < img_matrix.width; j++) {
            if (i < vert_boundary || i > img_matrix.height - vert_boundary) {
                img_matrix.set(j, i, border_color);
            } else if (j < horiz_boundary ||
                       j > img_matrix.width - horiz_boundary) {
                img_matrix.set(j, i, border_color);
            }
        }
    }
}

/// @brief Draws border around an image
void decode_encode_img(std::string filepath, image_codec *codec) {
    std::vector<unsigned char> img_buffer;

    codec->load_image_file(&img_buffer, input_folder / filepath);

    ImageInfo info = codec->read_info(&img_buffer);

    matrix *mat;
    if (info.colorScheme == ImageColorScheme::IMAGE_RGB) {
        matrix_rgb *color_mat = new matrix_rgb(info.width, info.height);
        mat = color_mat;

        codec->decode(&img_buffer, mat, info.colorScheme, info.bit_depth);
        draw_border<color_rgb>(*color_mat, color_rgb(255, 255, 255));
    } else {
        matrix_gray *color_mat = new matrix_gray(info.width, info.height);
        mat = color_mat;

        codec->decode(&img_buffer, mat, info.colorScheme, info.bit_depth);

        draw_border<unsigned char>(*color_mat, 255);
    }

    img_buffer.clear();
    codec->encode(&img_buffer, mat, info.colorScheme, info.bit_depth);
    codec->save_image_file(&img_buffer, result_folder / filepath);

    //delete mat;
}
