"""
Лабораторная работа №2: Цифровая обработка изображений.
Реализация эквализации для предоставленного изображения.
"""

import sys

import cv2 as cv
import matplotlib
import matplotlib.pyplot as plt
import numpy as np

try:
    matplotlib.use('Agg')

    ###############################################################################################

    # Загружаем изображение и переводим в оттенки серого
    image = cv.imread('task/not_equalized.png', cv.IMREAD_GRAYSCALE)
    # Массив яркостей пикселей в картинке от 0 до 255
    # Гистограмма изображения (показывает, сколько раз встречается каждое значение яркости)
    brightness_arr = cv.calcHist([image], [0], None, [256], [0, 256])
    # Массив вероятностей (вероятность встретить каждую яркость)
    # (кол-во пикселей с конкретным уровнем яркости) / (общее кол-во пикселей)
    probabilities_arr = brightness_arr.ravel() / brightness_arr.sum()
    # Кумулятивная функция распределения
    # Преобладает темнота (г. смещена влево) → произойдет общее осветление
    # Преобладают засветы (г. смещена вправо) → произойдет общее затемнение
    # Распределение по центру → произойдет усиление контраста (растяжение по обеим сторонам)
    cdf = probabilities_arr.cumsum()
    # Получаем новый процент яркости перемножением
    cdf_scaled = np.uint8(255 * cdf)
    # Преобразование по яркости
    # (каждый пиксель image[y][x] заменяется на cdf_scaled[image[y][x]])
    equalized_image = cdf_scaled[image]

    ###############################################################################################

    # Визуализация результатов
    gs = plt.GridSpec(2, 2)
    plt.figure(figsize=(10, 8))
    plt.subplot(gs[0])
    plt.title("Оригинал")
    plt.imshow(image, cmap='gray', vmin=0, vmax=255)
    plt.subplot(gs[1])
    plt.title("Эквализированное")
    plt.imshow(equalized_image, cmap='gray', vmin=0, vmax=255)
    plt.subplot(gs[2])
    plt.title("Гистограмма оригинала")
    plt.hist(image.flatten(), 256, range=(0, 256))
    plt.subplot(gs[3])
    plt.title("Гистограмма после эквализации")
    plt.hist(equalized_image.flatten(), 256, range=(0, 256))
    plt.tight_layout()

    # Сохраняем результат в файл
    plt.savefig('task/equalized.png')
    sys.exit()
except Exception as e:
    print(f"Ошибка: {e}")
