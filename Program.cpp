#include <ctime>
#include <filesystem>
#include <iostream>
#include <string>
#include <vector>

#include "image_codec.h"
#include "lodepng.h"

namespace fs = std::filesystem;

const fs::path RESULTFOLDER("output");
const fs::path INPUTFOLDER("input");
const unsigned int TEN = 10;

void DecodeEncodeImg(std::string filepath, image_codec *codec);

int main() {
    std::cout << "Shellow from SSAU!" << std::endl;

    image_codec codec;

    DecodeEncodeImg("shuttle.jpg", &codec);

    std::cout << "that's it" << std::endl;
}

void DecodeEncodeImg(std::string filepath, image_codec *codec) {
    std::vector<unsigned char> img_buffer;

    codec->load_image_file(&img_buffer, input_folder / filepath);

    matrix_rgb img_matrix;
    codec->decode(&img_buffer, &img_matrix, ImageColorScheme::IMAGE_RGB, 8);

    unsigned int vert_boundary = static_cast<unsigned int>(img_matrix.height / TEN);
    unsigned int horiz_boundary = static_cast<unsigned int>(img_matrix.width / TEN);

    for (size_t i = 0; i < img_matrix.height; i++) {
        for (size_t j = 0; j < img_matrix.width; j++) {
            if (i < vert_boundary || i > img_matrix.height - vert_boundary) {
                img_matrix.set(j, i, color_rgb(255, 255, 255));
            } else if (j < horiz_boundary || j > img_matrix.width - horiz_boundary) {
                img_matrix.set(j, i, color_rgb(255, 255, 255));
            }
        }
    }

    img_buffer.clear();
    codec->encode(&img_buffer, &img_matrix, ImageColorScheme::IMAGE_RGB, 8);

    codec->save_image_file(&img_buffer, RESULTFOLDER / filepath);
}
