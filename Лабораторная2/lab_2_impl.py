"""
Лабораторная работа №2: Цифровая обработка изображений.
Реализация эквализации изображения.
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
    image = cv.imread('imgs/winter_cat.png', cv.IMREAD_GRAYSCALE)
    # Строим гистограмму исходного изображения = массив с распределением яркостей
    brightness_arr = cv.calcHist([image], [0], None, [256], [0, 256])
    # Массив вероятностей сущ-я величин яркости
    probabilities_arr = brightness_arr.ravel() / brightness_arr.sum()
    # Вычисляем кумулятивную сумму вероятности(новый процент яркости) для каждой яркости
    cdf = probabilities_arr.cumsum()
    # Применяем новый процент яркости
    cdf_scaled = np.uint8(255 * cdf)
    # Применяем преобразование
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
    plt.savefig('./imgs/output_image.png')
    sys.exit()
except Exception as e:
    print(f"Ошибка: {e}")
