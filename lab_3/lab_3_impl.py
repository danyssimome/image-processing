"""
Лабораторная работа №3: Цифровая обработка изображений.
Убрать шумы предоставленного изображения.
"""

import sys

import cv2 as cv
import matplotlib
import matplotlib.pyplot as plt
import numpy as np

try:
    matplotlib.use('Qt5Agg')

    # Загрузка изображения
    image = cv.imread('task/noisy_image.png')
    image = cv.cvtColor(image, cv.COLOR_BGR2RGB)

    ###############################################################################################

    # Поворот изображения
    h, w = image.shape[:2]
    center = (w // 2, h // 2)
    ANGLE = -78
    matrix = cv.getRotationMatrix2D(center, ANGLE, 1.0)
    rotated = cv.warpAffine(image, matrix, (w, h), flags=cv.INTER_LINEAR, borderMode=cv.BORDER_REPLICATE)

    # Медианная фильтрация
    # Берется окно(пр: 3*3) с яркостями пикселей, знач. сортируются, выбирается медиана
    # Если существуют шумы(контрастные единичные пиксели), они заменяются на медианное значение
    filtered = cv.medianBlur(rotated, ksize=3)

    # Коррекция/балансировка контрастности через CLAHE без искажения цветов
    # Операция происходит в lab пространстве, где яркость L отделена от цветов
    lab = cv.cvtColor(filtered, cv.COLOR_RGB2LAB)
    l, a, b = cv.split(lab)
    clahe = cv.createCLAHE(clipLimit=1.0, tileGridSize=(10, 10))
    l_clahe = clahe.apply(l)
    lab_clahe = cv.merge((l_clahe, a, b))
    enhanced = cv.cvtColor(lab_clahe, cv.COLOR_LAB2RGB)

    # Усиление резкости через свертку
    # Сохраняем основную часть яркости текущего центрального пикселя
    # Вычитаем немного от соседей, чтобы подчеркнуть разницу границ
    sharp_kernel = np.asarray([[0, -0.25, 0], [-0.25, 2, -0.25], [0, -0.25, 0]])
    sharpened = cv.filter2D(enhanced, -1, sharp_kernel)

    ###############################################################################################

    # Визуализация
    fig, axs = plt.subplots(1, 4, figsize=(15, 5))
    axs[0].imshow(rotated)
    axs[0].set_title('Повернутое')
    axs[0].axis('off')
    axs[1].imshow(filtered)
    axs[1].set_title('Медианный фильтр')
    axs[1].axis('off')
    axs[2].imshow(enhanced)
    axs[2].set_title('CLAHE')
    axs[2].axis('off')
    axs[3].imshow(sharpened)
    axs[3].set_title('После резкости')
    axs[3].axis('off')
    plt.tight_layout()

    plt.show(block=True)
    sys.exit()
except Exception as e:
    print(f"Ошибка: {e}")
