#include "cmd_parser.h"
#include <iostream>
#include "image_codec.h"

extern void decode_encode_img(std::string filepath, image_codec *codec);

CmdParser::CmdParser() : desc("Allowed options") {
    desc.add_options()
        ("help", "produce help message")
        ("gui", "open GUI window")
        ("verbose,v", "enable verbose debug output")
        ("force-gpu", "force GPU usage and prevent CPU fallback")
        ("draw_border", po::value<std::string>(), "image in input directory to draw border on")
        ("crop", po::value<std::string>(), "image in input directory to crop")
        ("rotate", po::value<std::string>(), "image in input directory to rotate")
        ("rotate_angle", po::value<int>()->default_value(90), "rotation angle")
        ("reflect", po::value<std::string>(), "image in input directory to reflect")
        ("horizontal", po::bool_switch(), "reflect horizontally")
        ("vertical", po::bool_switch(), "reflect vertically")
        ("shear", po::value<std::string>(), "image in input directory to shear")
        ("shear_x", po::value<float>()->default_value(0.0f), "horizontal shear factor")
        ("shear_y", po::value<float>()->default_value(0.0f), "vertical shear factor");
}

CmdParser::~CmdParser() {}

void CmdParser::decode_encode_img(std::string filepath, image_codec *codec) {
    ::decode_encode_img(filepath, codec);
}

po::variables_map CmdParser::parse_arguments(int ac, char* av[]) {
    try {
        po::store(po::command_line_parser(ac, av).options(desc).run(), vm);
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
    }
    catch (const po::error& e) {
        std::cerr << "Error: " << e.what() << "\n\n";
        std::cerr << "For help, use --help option\n";
        exit(1);
    }

    return vm;
}

CommandType CmdParser::get_command_type() const {
    if (vm.count("help")) {
        return CommandType::HELP;
    }
    else if (vm.count("gui")) {
        return CommandType::GUI;
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
    else if (vm.count("reflect")) {
        return CommandType::REFLECT;
    }
    else if (vm.count("shear")) {
        return CommandType::SHEAR;
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
        case CommandType::REFLECT:
            return get_reflect_command_data();
        case CommandType::SHEAR:
            return get_shear_command_data();
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

bool CmdParser::is_verbose() const {
    return vm.count("verbose") > 0;
}

bool CmdParser::is_force_gpu() const {
    return vm.count("force-gpu") > 0;
}

std::unique_ptr<ReflectCommandData> CmdParser::get_reflect_command_data() const {
    if (!vm.count("reflect")) {
        return nullptr;
    }

    auto data = std::make_unique<ReflectCommandData>();
    data->imagePath = vm["reflect"].as<std::string>();
    data->horizontal = vm["horizontal"].as<bool>();
    data->vertical = vm["vertical"].as<bool>();
    return data;
}

std::unique_ptr<ShearCommandData> CmdParser::get_shear_command_data() const {
    if (!vm.count("shear")) {
        return nullptr;
    }

    auto data = std::make_unique<ShearCommandData>();
    data->imagePath = vm["shear"].as<std::string>();
    data->shearX = vm["shear_x"].as<float>();
    data->shearY = vm["shear_y"].as<float>();
    return data;
}
