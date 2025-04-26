#ifndef image_codec_cl_h
#define image_codec_cl_h

#include <CL/opencl.hpp>
#include "image_codec.h"

class image_codec_cl : public image_codec {
private:
    unsigned width;
    unsigned height;
    cl::Context context;
    cl::Device device;
    cl::Program program;
    cl::CommandQueue queue;
    bool initialized;

    bool initializeOpenCL();

public:
    image_codec_cl();
    ~image_codec_cl();

    ImageInfo read_info(std::vector<unsigned char>* img_buffer) override;
    void encode(std::vector<unsigned char>* img_buffer, matrix* img_matrix, ImageColorScheme colorScheme, unsigned bit_depth) override;
    void decode(std::vector<unsigned char>* img_source, matrix* img_matrix, ImageColorScheme colorScheme, unsigned bit_depth) override;
    void load_image_file(std::vector<unsigned char>* png_buffer, std::string image_filepath) override;
    void save_image_file(std::vector<unsigned char>* png_buffer, std::string image_filepath) override;
};

#endif 