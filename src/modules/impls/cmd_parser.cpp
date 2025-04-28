#include "cmd_parser.h"
#include <iostream>
#include "image_codec.h"

extern void decode_encode_img(std::string filepath, image_codec *codec);

CmdParser::CmdParser() : desc("Allowed options") {
    desc.add_options()
        ("help", "produce help message")
        ("draw_border", po::value<std::string>(), "image in input directory to draw border on")
        ("crop", po::value<std::string>(), "image in input directory to crop")
        ("crop_left", po::value<unsigned>()->default_value(200), "pixels to crop from left")
        ("crop_top", po::value<unsigned>()->default_value(200), "pixels to crop from top")
        ("crop_right", po::value<unsigned>()->default_value(200), "pixels to crop from right")
        ("crop_bottom", po::value<unsigned>()->default_value(200), "pixels to crop from bottom")
        ("rotate", po::value<std::string>(), "image in input directory to rotate")
        ("rotate_angle", po::value<int>()->default_value(90), "rotation angle (90, 180, or 270 degrees)");
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
    else if (vm.count("crop")) {
        return CommandType::CROP;
    }
    else if (vm.count("rotate")) {
        return CommandType::ROTATE;
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
        case CommandType::CROP:
            return get_crop_command_data();
        case CommandType::ROTATE:
            return get_rotate_command_data();
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

std::unique_ptr<CropCommandData> CmdParser::get_crop_command_data() const {
    if (!vm.count("crop")) {
        return nullptr;
    }

    auto data = std::make_unique<CropCommandData>();
    data->imagePath = vm["crop"].as<std::string>();
    data->crop_left = vm["crop_left"].as<unsigned>();
    data->crop_top = vm["crop_top"].as<unsigned>();
    data->crop_right = vm["crop_right"].as<unsigned>();
    data->crop_bottom = vm["crop_bottom"].as<unsigned>();
    return data;
}

std::unique_ptr<RotateCommandData> CmdParser::get_rotate_command_data() const {
    if (!vm.count("rotate")) {
        return nullptr;
    }

    auto data = std::make_unique<RotateCommandData>();
    data->imagePath = vm["rotate"].as<std::string>();
    data->angle = vm["rotate_angle"].as<int>();
    return data;
}
