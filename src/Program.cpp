#include "modules/cmd_parser.h"
#include "modules/image_codec.h"
#include "image_codec.h"
#include "image_tools.h"
#include "image_edit.h"

#include <iostream>
#include <memory>
#include <filesystem>

#include "gui/mainwindow.h"
#include <QApplication>

namespace fs = std::filesystem;

// Global flags for controlling behavior
bool g_verbose_enabled = false;
bool g_force_gpu_enabled = false;

// Make flags accessible to other modules
extern bool g_verbose_enabled;
extern bool g_force_gpu_enabled;


const fs::path result_folder("output");
const fs::path input_folder ("input");


void decode_encode_img(std::string filepath, image_codec* codec);

int main(int argc, char* argv[]) {
    std::cout << "Shellow from SSAU!" << std::endl;
    CmdParser parser;
    parser.parse_arguments(argc, argv);
    CommandType cmdType = parser.get_command_type();

    // Set the global flags
    g_verbose_enabled = parser.is_verbose();
    g_force_gpu_enabled = parser.is_force_gpu();

    if (g_verbose_enabled) {
        std::cout << "Verbose mode enabled" << std::endl;
    }

    if (g_force_gpu_enabled) {
        std::cout << "Forcing GPU mode (will not fall back to CPU)" << std::endl;
    }

    // If no valid command was specified, show help
    if (cmdType == CommandType::NONE) {
        std::cout << "No valid command specified. Use --help for available options.\n";
        return 1;
    }



    image_codec codec;

    switch (cmdType) {
        case CommandType::GUI: {
            QApplication a(argc, argv);
            MainWindow w;
            w.show();
            return a.exec();
        }

        case CommandType::HELP: {
            std::cout << "Help requested\n";
            auto helpData = parser.get_help_command_data();
            return 0;
        }

        case CommandType::DRAW_BORDER: {
            auto drawBorderData = parser.get_draw_border_command_data();
            if (drawBorderData) {
                std::cout << "Processing image: " << drawBorderData->imagePath << "\n";
                parser.decode_encode_img(drawBorderData->imagePath, &codec);
                std::cout << drawBorderData->imagePath << " drawed successfully\n";
            }
            return 0;
        }

        case CommandType::CROP: {
            auto cropData = parser.get_crop_command_data();
            if (cropData) {
                std::cout << "Cropping image: " << cropData->imagePath << "\n";
                transform_image_crop(cropData->imagePath, &codec, cropData->crop_left, cropData->crop_top,
                                                                  cropData->crop_right, cropData->crop_bottom);
                std::cout << cropData->imagePath << " cropped successfully\n";
            }
            return 0;
        }

        case CommandType::ROTATE: {
            auto rotateData = parser.get_rotate_command_data();
            if (rotateData) {
                std::cout << "Processing image: " << rotateData->imagePath << "\n";
                transform_image_rotate(rotateData->imagePath, &codec, rotateData->angle);
                std::cout << rotateData->imagePath << " rotated successfully\n";
            }
            return 0;
        }

        case CommandType::REFLECT: {
            auto reflectData = parser.get_reflect_command_data();
            if (reflectData) {
                std::cout << "Reflecting image: " << reflectData->imagePath << "\n";
                transform_image_reflect(reflectData->imagePath, &codec,
                                     reflectData->horizontal, reflectData->vertical);
                std::cout << reflectData->imagePath << " reflected successfully\n";
            }
            return 0;
        }

        case CommandType::SHEAR: {
            auto shearData = parser.get_shear_command_data();
            if (shearData) {
                std::cout << "Shearing image: " << shearData->imagePath << "\n";
                transform_image_shear(shearData->imagePath, &codec,
                                   shearData->shearX, shearData->shearY);
                std::cout << shearData->imagePath << " sheared successfully\n";
            }
            return 0;
        }

        case CommandType::NONE:
        default:
            std::cout << "No valid command specified. Use --help for available options.\n";
            return 1;
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
