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

    # Фильтрация изображения, т.к. алгоритм чувствителен к шумам
    # Похоже на медианный фильтр, но здесь значения шумов пикселей усредняются
    # (хорошо работает для плавных шумов, т.к. картинку чуть размыливает)
    blurred = cv.GaussianBlur(gray, (5, 5), 0)

    # Градиенты Собеля
    # Получаем массивы с границами на изображении — места, где яркость резко меняется
    # Перемножаем яркости на текущем участке изображения с ядром(пр. 3*3) = получаем число
    # Однородный фон~0–50; Слабая граница~100–200; Резкая граница~300–800;
    grad_x = cv.Sobel(blurred, cv.CV_64F, 1, 0, ksize=3)
    grad_y = cv.Sobel(blurred, cv.CV_64F, 0, 1, ksize=3)

    # Расчёт модуля градиента
    # Усредняем результаты по разным осям, тем самым добавляя точность
    magnitude = cv.magnitude(grad_x, grad_y)
    magnitude = cv.convertScaleAbs(magnitude)

    # Пороговая бинаризация
    # Превращаем массив с градиентами в массив с яркостями относительно порогового значения
    # Пр.: если число градиента для пикселя > 300, присваиваем яркость 255, остальные помечаем 0
    _, binary = cv.threshold(magnitude, 150, 255, cv.THRESH_BINARY)

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
