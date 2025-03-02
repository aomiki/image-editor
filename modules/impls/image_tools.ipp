
template<typename E>
void matrix_color<E>::fill(E value)
{
    for (size_t x = 0; x < width; x++)
    {
        for (size_t y = 0; y < height; y++)
        {
            set(x, y, value);
        }
    }
}

inline matrix::matrix(unsigned width, unsigned height)
{
    this->height = height;
    this->width = width;
}

inline matrix::matrix()
{
    this->height = 0;
    this->width = 0;
}