"""
Лабораторная работа №3: Цифровая обработка изображений.
Реализация низко-частотной фильтрации изображения для неподходящего шума соль-перец.
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

    # Вращаем изображение
    (h, w) = image.shape[:2]
    center = (w // 2, h // 2)
    angle = 12  # -78
    matrix = cv.getRotationMatrix2D(center, angle, 1.0)
    rotated = cv.warpAffine(image, matrix, (w, h), flags=cv.INTER_LINEAR, borderMode=cv.BORDER_REPLICATE)

    ###############################################################################################

    # Частотная фильтрация (НЧФ)
    # Фурье
    dft = np.fft.fft2(rotated)
    dft_shift = np.complex64(np.fft.fftshift(dft))
    # Маска
    mask = np.zeros_like(rotated, dtype=np.uint8)
    crow, ccol = h // 2, w // 2
    RAD = 60  # радиус низкочастотной области
    cv.circle(mask, (ccol, crow), RAD, 1, -1)
    dft_shift_filtered = dft_shift * mask

    # Обратное преобразование
    img_back = np.abs(np.fft.ifft2(np.fft.ifftshift(dft_shift_filtered)))
    img_back = np.uint8(np.clip(img_back, 0, 255))

    # Применение CLAHE (адаптивное выравнивание контраста)
    enhanced = cv.createCLAHE(clipLimit=3.0, tileGridSize=(8, 8)).apply(img_back)

    ###############################################################################################

    # Визуализация
    fig, axs = plt.subplots(1, 2, figsize=(10, 4))
    axs[0].imshow(rotated, cmap='gray')
    axs[0].set_title('Поворот')
    axs[0].axis('off')
    axs[1].imshow(enhanced, cmap='gray')
    axs[1].set_title('НЧФ + CLAHE')
    axs[1].axis('off')
    plt.tight_layout()

    # plt.imshow(rotated_image, cmap='gray')
    plt.show(block=True)
    sys.exit()
except Exception as e:
    print(f"Ошибка: {e}")
