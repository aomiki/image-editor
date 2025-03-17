#include <boost/program_options.hpp>
#include <iostream>
namespace po = boost::program_options;

void decode_encode_img(std::string filepath, image_codec *codec);
po::variables_map parse_arguments(int ac, char* av[]) {
    po::options_description desc("Allowed options");
    desc.add_options()
        ("help", "produce help message")
        ("draw_border", po::value<std::string>(), "image in input directory to draw border on");

    po::variables_map vm;
    po::store(po::parse_command_line(ac, av, desc), vm);
    po::notify(vm);

    if (vm.count("help")) {
        std::cout << desc << "\n";
        exit(0);      }

    if (vm.count("draw_border")) {
        image_codec codec;
        std::string inp_img = vm["draw_border"].as<std::string>();
        decode_encode_img(inp_img, &codec);
        std::cout << inp_img << " drawed successfully\n";
    }

    return vm;
}
