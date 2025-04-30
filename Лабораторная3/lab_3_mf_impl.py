"""
Лабораторная работа №3: Цифровая обработка изображений.
Реализация медианной фильтрации изображения для подходящего шума соль-перец.
"""

import sys

import cv2 as cv
import matplotlib
import matplotlib.pyplot as plt
import numpy as np

try:
    matplotlib.use('Qt5Agg')

    # Загрузка изображения как оттенки серого
    image = cv.imread('./imgs/noized_image.jpg', cv.IMREAD_GRAYSCALE)

    # Поворот изображения
    h, w = image.shape
    center = (w // 2, h // 2)
    ANGLE = 12
    matrix = cv.getRotationMatrix2D(center, ANGLE, 1.0)
    rotated = cv.warpAffine(image, matrix, (w, h), flags=cv.INTER_LINEAR, borderMode=cv.BORDER_REPLICATE)

    ###############################################################################################

    # Медианная фильтрация
    median_filtered = cv.medianBlur(rotated, ksize=5)

    # Усиление резкости
    sharp_kernel = np.array([[0, -1, 0],
                             [-1, 5, -1],
                             [0, -1, 0]])
    sharpened = cv.filter2D(median_filtered, -1, sharp_kernel)

    ###############################################################################################

    # Визуализация
    fig, axs = plt.subplots(1, 3, figsize=(12, 4))
    axs[0].imshow(rotated, cmap='gray')
    axs[0].set_title('Повернутое')
    axs[0].axis('off')
    axs[1].imshow(median_filtered, cmap='gray')
    axs[1].set_title('Медианный фильтр')
    axs[1].axis('off')
    axs[2].imshow(sharpened, cmap='gray')
    axs[2].set_title('После резкости')
    axs[2].axis('off')
    plt.tight_layout()

    # plt.imshow(rotated_image, cmap='gray')
    plt.show(block=True)
    sys.exit()
except Exception as e:
    print(f"Ошибка: {e}")
