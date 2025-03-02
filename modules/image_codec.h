#include <string>
#include <vector>
#include "image_tools.h"

#ifndef image_codec_h
#define image_codec_h

enum ImageColorScheme{
    IMAGE_GRAY,
    IMAGE_RGB
};

class image_codec {
    public:
        image_codec();

        /// @brief Encodes image matrix to supported format
        /// @param[out] img_buffer Buffer for encoded image
        /// @param[in] img_matrix Image matrix 
        /// @param[in] colorScheme Image matrix color scheme
        /// @param[in] bit_depth Image matrix bit depth
        void encode(std::vector<unsigned char>* img_buffer, matrix* img_matrix, ImageColorScheme colorScheme, unsigned bit_depth);

        /// @brief Decodes image from some format to image matrix
        /// @param[in] img_source Image in some format 
        /// @param[out] img_matrix Image matrix buffer
        /// @param[in] colorScheme Image matrix color scheme 
        /// @param[in] bit_depth Image matrix bit depth 
        void decode(std::vector<unsigned char>* img_source, matrix* img_matrix, ImageColorScheme colorScheme, unsigned bit_depth);

        /// @brief Reads image from a file
        /// @param[out] png_buffer Where to read it to
        /// @param[in] image_filepath filepath, with extension
        void load_image_file(std::vector<unsigned char>* png_buffer, std::string image_filepath);
        
        /// @brief Saves image to file
        /// @param[in] png_buffer image data
        /// @param[in] image_filepath filepath, without extension
        void save_image_file(std::vector<unsigned char>* png_buffer, std::string image_filepath);

        ~image_codec();
};

#endif
