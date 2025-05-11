#include <vector>
#include "image_codec.h"
#include "image_tools.h"

/// @brief Manages all necessary recources, data allocations
class scene
{
    private:
        image_codec* codec;
        matrix* img_matrix;
        ImageColorScheme colorScheme;
        std::vector<unsigned char>* img_buffer;

    public:
        scene();
        ~scene();

        long unsigned get_img_binary_size();
        void get_img_binary(unsigned char* img_buffer);

        image_codec* get_codec()
        {
            return codec;
        }

        ImageInfo get_img_info();
        void load_image_file(std::string inp_filename);
        void save_image_file(std::string inp_filename);
        void decode();
        void encode();

        void rotate(float angle);
        void crop(unsigned crop_left, unsigned crop_top, unsigned crop_right, unsigned crop_bottom);
        void reflect(bool horizontal, bool vertical);
        void shear(float shx, float shy);
        void grayscale();
        void gaussian_blur(float sigma);
};
