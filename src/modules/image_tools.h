#include <vector>

#ifndef image_tools_h
#define image_tools_h

/// @brief Element of RGB matrix
struct color_rgb {
    color_rgb(unsigned char r, unsigned char g, unsigned char b)
    {
        red = r;
        green = g;
        blue = b;
    }

    unsigned char red;
    unsigned char green;
    unsigned char blue;
};

struct matrix_coord {

    matrix_coord(unsigned x, unsigned y)
    {
        this->x = x;
        this->y = y;
    }

    unsigned x;
    unsigned y;
};

/// @brief Abstract matrix
class matrix {
    public:
        std::vector<unsigned char> array;
        unsigned width;
        unsigned height;

        matrix(unsigned width, unsigned height);
        matrix();
};

/// @brief Abstract image matrix
/// @tparam E Type of matrix elements
template<typename E>
class matrix_color : public matrix {
    public:
    matrix_color() : matrix() {}
    matrix_color(unsigned width, unsigned height) : matrix(width, height) {}

    /// @brief Assign value to matrix cell
    /// @param[in] x x coordinate
    /// @param[in] y y coordinate
    /// @param[in] color element value
    void virtual set(unsigned x, unsigned y, E color) = 0;

    /// @brief Assign value to matrix cell
    /// @param[in] coord coordinates
    /// @param[in] color element value
    void virtual set(matrix_coord coord, E color) = 0;

    /// @brief Get matrix cell value
    /// @param[in] x x coordinate
    /// @param[in] y y coordinate
    /// @return cell value
    E virtual get(unsigned x, unsigned y) = 0;

    /// @brief Assign \p value to each matrix cell
    /// @param[in] value 
    void virtual fill(E value);
};

/// @brief Matrix for storing RGB images
class matrix_rgb : public matrix_color<color_rgb>
{
    public:
        matrix_rgb(): matrix_color<color_rgb>() {}
        matrix_rgb(unsigned width, unsigned height): matrix_color<color_rgb>(width, height) {}
        void virtual set(unsigned x, unsigned y, color_rgb color);
        void virtual set(matrix_coord coord, color_rgb color);
        color_rgb virtual get(unsigned x, unsigned y);
        void virtual fill(color_rgb value);
};

/// @brief Matrix for storing grayscale images
class matrix_gray : public matrix_color<unsigned char>
{
    public:
        matrix_gray(unsigned width, unsigned height): matrix_color<unsigned char>(width, height) {}
        void virtual set(unsigned x, unsigned y, unsigned char color);
        void virtual set(matrix_coord coord, unsigned char color);
        unsigned char virtual get(unsigned x, unsigned y);
        void virtual fill(unsigned char value);
};

#include "impls/image_tools.ipp"

#endif
