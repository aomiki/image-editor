#include <filesystem>
#include <iostream>
#include <string>
#include <vector>

#include "image_codec.h"
#include "lodepng.h"

namespace fs = std::filesystem;

const fs::path result_folder("output");
const fs::path input_folder("input");

void decode_encode_img(std::string filepath, image_codec *codec);

int main() {
  std::cout << "Shellow from SSAU!" << std::endl;

  image_codec codec;

  decode_encode_img("shuttle.jpg", &codec);

  std::cout << "that's it" << std::endl;
}

void decode_encode_img(std::string filepath, image_codec *codec) {
  std::vector<unsigned char> img_buffer;

  codec->load_image_file(&img_buffer, input_folder / filepath);

  matrix_rgb img_matrix;
  codec->decode(&img_buffer, &img_matrix, ImageColorScheme::IMAGE_RGB, 8);

  unsigned int vert_boundary = (int)img_matrix.height / 10;
  unsigned int horiz_boundary = (int)img_matrix.width / 10;

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

  codec->save_image_file(&img_buffer, result_folder / filepath);
}
