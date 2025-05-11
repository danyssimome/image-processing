"""
Лабораторная работа №4: Цифровая обработка изображений.
Выделить границы предоставленного изображения.
"""

import sys

import cv2 as cv
import matplotlib
import matplotlib.pyplot as plt

try:
    matplotlib.use('Qt5Agg')

    # Загрузка изображения
    image = cv.imread('./task/oranges.png')
    image = cv.cvtColor(image, cv.COLOR_BGR2RGB)
    gray = cv.cvtColor(image, cv.COLOR_RGB2GRAY)

    ###############################################################################################

    # Сглаживание для подавления шума
    blurred = cv.GaussianBlur(gray, (5, 5), 0)

    # Градиенты Собеля
    grad_x = cv.Sobel(blurred, cv.CV_64F, 1, 0, ksize=3)
    grad_y = cv.Sobel(blurred, cv.CV_64F, 0, 1, ksize=3)

    # Расчёт модуля градиента
    magnitude = cv.magnitude(grad_x, grad_y)
    magnitude = cv.convertScaleAbs(magnitude)

    # Пороговая бинаризация
    _, binary = cv.threshold(magnitude, 90, 255, cv.THRESH_BINARY)

    ###############################################################################################

    fig, axs = plt.subplots(1, 2, figsize=(15, 5))
    axs[0].imshow(image)
    axs[0].set_title('Оригинал')
    axs[0].axis('off')
    axs[1].imshow(binary, cmap='gray')
    axs[1].set_title('Выделенные границы')
    axs[1].axis('off')
    plt.tight_layout()

    plt.show(block=True)
    sys.exit()
except Exception as e:
    print(f"Ошибка: {e}")
