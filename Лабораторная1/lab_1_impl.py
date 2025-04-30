"""
Лабораторная работа №1: Использованием NumPy.
Средствами numpy создать массив со случайными значениями и максимальный элемент заменить на 0.
"""

import numpy as np

try:
    size = int(input("Введите размер массива: "))

    if size <= 0:
        raise ValueError("Размер должен быть положительным числом")

    arr = np.random.rand(size)
    print("Исходный массив:", arr)

    max_index = np.argmax(arr)
    arr[max_index] = 0
    print("Массив после замены максимального элемента на 0:", arr)

except Exception as e:
    print(f"Ошибка: {e}")
