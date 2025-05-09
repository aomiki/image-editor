#include "image_tools.h"

/**
 * @brief Обрезает изображение по заданным границам
 * @param img Матрица изображения
 * @param crop_left Количество пикселей для обрезки слева
 * @param crop_top Количество пикселей для обрезки сверху
 * @param crop_right Количество пикселей для обрезки справа
 * @param crop_bottom Количество пикселей для обрезки снизу
 */
void crop(matrix& img, unsigned crop_left, unsigned crop_top, unsigned crop_right, unsigned crop_bottom);

/**
 * @brief Отражает изображение по горизонтали и/или вертикали
 * @param img Матрица изображения
 * @param horizontal Если `true`, отражает по горизонтали
 * @param vertical Если `true`, отражает по вертикали
 * @attention Если оба параметра `false`, изображение не изменится
 */
void reflect(matrix& img, bool horizontal, bool vertical);

/**
 * @brief Применяет аффинный сдвиг к изображению
 * @param img Матрица изображения
 * @param shx Коэффициент сдвига по оси X
 * @param shy Коэффициент сдвига по оси Y
 * @note как работает: https://www.geeksforgeeks.org/shearing-in-2d-graphics/
 */
void shear(matrix& img, float shx, float shy);

void bilinear_interpolate(matrix& img, float x, float y, unsigned char* result);

/**
 * @brief Поворачивает изображение на заданный (любой) угол
 * @param img Матрица изображения
 * @param angle Угол поворота в градусах
 */
void rotate(matrix& img, float angle);
