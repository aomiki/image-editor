#include <vector>

#ifndef image_tools_h
#define image_tools_h

/// @brief Element of RGB matrix
struct Colorrgb {
    Colorrgb(unsigned char r, unsigned char g, unsigned char b)
    {
        red = r;
        green = g;
        blue = b;
    }

    unsigned char red;
    unsigned char green;
    unsigned char blue;
};

struct Matrixcoord {

    Matrixcoord(unsigned x, unsigned y)
    {
        this->x = x;
        this->y = y;
    }

    unsigned x;
    unsigned y;
};

/// @brief Abstract matrix
class Matrix {
    public:
        std::vector<unsigned char> array;
        unsigned width;
        unsigned height;

        Matrix(unsigned width, unsigned height);
        Matrix();
};

/// @brief Abstract image matrix
/// @tparam E Type of matrix elements
template<typename E>
class MatrixColor : public Matrix {
    public:
    MatrixColor() : Matrix() {}
    MatrixColor(unsigned width, unsigned height) : Matrix(width, height) {}

    /// @brief Assign value to matrix cell
    /// @param[in] x x coordinate
    /// @param[in] y y coordinate
    /// @param[in] color element value
    void virtual set(unsigned x, unsigned y, E color) = 0;

    /// @brief Assign value to matrix cell
    /// @param[in] coord coordinates
    /// @param[in] color element value
    void virtual Set(Matrixcoord coord, E color) = 0;

    /// @brief Get matrix cell value
    /// @param[in] x x coordinate
    /// @param[in] y y coordinate
    /// @return cell value
    E virtual Get(unsigned x, unsigned y) = 0;

    /// @brief Assign \p value to each matrix cell
    /// @param[in] value
    void virtual Fill(E value) = 0;
};

/// @brief Matrix for storing RGB images
class Matrix_rgb : public Matrix_color<color_rgb>
{
    public:
        Matrix_rgb(): Matrix_color<color_rgb>() {}
        Matrix_rgb(unsigned width, unsigned height): Matrix_color<color_rgb>(width, height) {}
        void virtual set(unsigned x, unsigned y, color_rgb color);
        void virtual Set(Matrixcoord coord, color_rgb color);
        color_rgb virtual Get(unsigned x, unsigned y);
        void virtual Fill(color_rgb value);
};

/// @brief Matrix for storing grayscale images
class Matrix_gray : public Matrix_color<unsigned char>
{
    public:
        Matrix_gray(unsigned width, unsigned height): Matrix_color<unsigned char>(width, height) {}
        void virtual Set(unsigned x, unsigned y, unsigned char color);
        void virtual Set(Matrixcoord coord, unsigned char color);
        unsigned char virtual Get(unsigned x, unsigned y);
        void virtual Fill(unsigned char value);
};

#include "impls/image_tools.ipp"

#endif
