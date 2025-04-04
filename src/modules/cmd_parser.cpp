#include "cmd_parser.h"
#include <iostream>
#include "image_codec.h"
#include <filesystem>
#include <vector>

namespace fs = std::filesystem;


extern void decode_encode_img(std::string filepath, image_codec *codec);

CmdParser::CmdParser() : desc("Allowed options") {
    desc.add_options()
        ("help", "produce help message")
        ("draw_border", po::value<std::string>(), "image in input directory to draw border on");
}

CmdParser::~CmdParser() {}

void CmdParser::decode_encode_img(std::string filepath, image_codec *codec) {
    ::decode_encode_img(filepath, codec);
}

po::variables_map CmdParser::parse_arguments(int ac, char* av[]) {
    po::store(po::parse_command_line(ac, av, desc), vm);
    po::notify(vm);

    if (vm.count("help")) {
        std::cout << desc << "\n";
        exit(0);
    }

    if (vm.count("draw_border")) {
        image_codec codec;
        std::string inp_img = vm["draw_border"].as<std::string>();
        decode_encode_img(inp_img, &codec);
        std::cout << inp_img << " drawed successfully\n";
    }

    return vm;
}

CommandType CmdParser::get_command_type() const {
    if (vm.count("help")) {
        return CommandType::HELP;
    }
    else if (vm.count("draw_border")) {
        return CommandType::DRAW_BORDER;
    }
    
    return CommandType::NONE;
}

std::unique_ptr<CommandData> CmdParser::get_command_data() const {
    CommandType type = get_command_type();
    
    switch (type) {
        case CommandType::HELP:
            return get_help_command_data();
        case CommandType::DRAW_BORDER:
            return get_draw_border_command_data();
        case CommandType::NONE:
        default:
            return nullptr;
    }
}

std::unique_ptr<HelpCommandData> CmdParser::get_help_command_data() const {
    if (!vm.count("help")) {
        return nullptr;
    }
    
    return std::make_unique<HelpCommandData>();
}

std::unique_ptr<DrawBorderCommandData> CmdParser::get_draw_border_command_data() const {
    if (!vm.count("draw_border")) {
        return nullptr;
    }
    
    auto data = std::make_unique<DrawBorderCommandData>();
    data->imagePath = vm["draw_border"].as<std::string>();
    return data;
} 