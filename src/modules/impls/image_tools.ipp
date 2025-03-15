template<typename E>
void matrix_color<E>::fill(E value)
{
    unsigned char* c_value = new unsigned char[components_num];
    element_to_c_arr(c_value, value);
    matrix::fill(c_value);
}

template<typename E>
E matrix_color<E>::get(unsigned x, unsigned y)
{
    unsigned char* cell = matrix::get(x, y);
    E value = c_arr_to_element(cell);

    return value;
};

template<typename E>
void matrix_color<E>::set(matrix_coord coord, E value)
{
    set(coord.x, coord.y , value);
}

template<typename E>
void matrix_color<E>::set(unsigned x, unsigned y, E value)
{
    unsigned char* cell = matrix::get(x, y);
    element_to_c_arr(cell, value);
}
