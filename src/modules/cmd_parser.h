#pragma once

#include <boost/program_options.hpp>
#include <string>
#include <memory>

class image_codec;

namespace po = boost::program_options;

enum class CommandType {
    NONE,
    HELP,
    DRAW_BORDER
};

struct CommandData {
    virtual ~CommandData() = default;
};

struct HelpCommandData : public CommandData {
};

struct DrawBorderCommandData : public CommandData {
    std::string imagePath;
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


    void decode_encode_img(std::string filepath, image_codec *codec);

private:
    po::options_description desc;
    po::variables_map vm;
};
