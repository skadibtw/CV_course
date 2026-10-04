import cv2
import numpy as np


def erode_opencv(image):
    kernel = np.ones((3, 3), dtype=np.uint8)
    return cv2.erode(image, kernel, borderType=cv2.BORDER_CONSTANT, borderValue=0)


def erode_native(image):
    height, width = image.shape
    pixels = image.tolist()
    result = [[0] * width for _ in range(height)]

    # При нулевом дополнении рамка всегда чёрная, её можно не обходить.
    for y in range(1, height - 1):
        for x in range(1, width - 1):
            minimum = 255
            for dy in range(-1, 2):
                for dx in range(-1, 2):
                    value = pixels[y + dy][x + dx]
                    if value < minimum:
                        minimum = value
            result[y][x] = minimum

    return np.array(result, dtype=np.uint8)
