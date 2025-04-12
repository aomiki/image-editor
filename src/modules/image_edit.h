#include "image_codec.h"
#include <filesystem>
namespace fs = std::filesystem;
extern const fs::path input_folder;
extern const fs::path result_folder;
void transform_image_crop(std::string filepath, image_codec* codec, 
                         unsigned int crop_left, unsigned int crop_top,
                         unsigned int crop_right, unsigned int crop_bottom);
void transform_image_rotate(std::string filepath, image_codec* codec, unsigned angle);
void transform_image_reflect(std::string filepath, image_codec* codec, bool horizontal, bool vertical);