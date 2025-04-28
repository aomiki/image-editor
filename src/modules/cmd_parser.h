#pragma once

#include <boost/program_options.hpp>
#include <string>
#include <memory>

class image_codec;

namespace po = boost::program_options;

enum class CommandType {
    NONE,
    HELP,
    DRAW_BORDER,
    CROP,
    ROTATE
};

class CommandData {
public:
    virtual ~CommandData() = default;
};

class HelpCommandData : public CommandData {};

class DrawBorderCommandData : public CommandData {
public:
    std::string imagePath;
};

class CropCommandData : public CommandData {
public:
    std::string imagePath;
    unsigned crop_left;
    unsigned crop_top;
    unsigned crop_right;
    unsigned crop_bottom;
};

class RotateCommandData : public CommandData {
public:
    std::string imagePath;
    int angle;
};

class CmdParser {
public:
    CmdParser();
    ~CmdParser();


    po::variables_map parse_arguments(int ac, char* av[]);

    CommandType get_command_type() const;

    std::unique_ptr<CommandData> get_command_data() const;

    // Specific command data acquisition functions
    std::unique_ptr<HelpCommandData> get_help_command_data() const;
    std::unique_ptr<DrawBorderCommandData> get_draw_border_command_data() const;
    std::unique_ptr<CropCommandData> get_crop_command_data() const;
    std::unique_ptr<RotateCommandData> get_rotate_command_data() const;


    void decode_encode_img(std::string filepath, image_codec *codec);

private:
    po::options_description desc;
    po::variables_map vm;
};
