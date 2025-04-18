#include "image_codec.h"
#include <filesystem>
namespace fs = std::filesystem;
extern const fs::path input_folder;
extern const fs::path result_folder;
void transform_image_crop(std::string filepath, image_codec* codec);
void transform_image_rotate(std::string filepath, image_codec* codec, unsigned angle);
