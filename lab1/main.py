import argparse
from pathlib import Path
from time import perf_counter

import cv2
import numpy as np

from erosion import erode_native, erode_opencv


def read_image(path):
    data = np.fromfile(path, dtype=np.uint8)
    image = cv2.imdecode(data, cv2.IMREAD_GRAYSCALE)
    if image is None:
        raise ValueError(f"Не удалось прочитать изображение: {path}")
    return image


def save_image(path, image):
    success, data = cv2.imencode(".png", image)
    if not success:
        raise ValueError(f"Не удалось сохранить изображение: {path}")
    data.tofile(path)


def main():
    parser = argparse.ArgumentParser(description="Эрозия с ядром 3x3")
    parser.add_argument("image", type=Path, nargs="?", default=Path("images/shapes.png"))
    parser.add_argument("--threshold", type=int, default=127)
    parser.add_argument("--output", type=Path, default=Path("results"))
    args = parser.parse_args()
    if not 0 <= args.threshold <= 255:
        parser.error("Порог должен быть от 0 до 255")
    if not args.image.is_file():
        parser.error(f"Файл не найден: {args.image}")

    try:
        image = read_image(args.image)
    except ValueError as error:
        parser.error(str(error))
    _, binary = cv2.threshold(image, args.threshold, 255, cv2.THRESH_BINARY)
    start = perf_counter()
    native = erode_native(binary)
    native_time = (perf_counter() - start) * 1000

    start = perf_counter()
    library = erode_opencv(binary)
    opencv_time = (perf_counter() - start) * 1000

    args.output.mkdir(parents=True, exist_ok=True)
    save_image(args.output / "binary.png", binary)
    save_image(args.output / "native.png", native)
    save_image(args.output / "opencv.png", library)
    print(f"Python: {native_time:.3f} мс")
    print(f"OpenCV: {opencv_time:.3f} мс")
    print(f"Результаты совпадают: {np.array_equal(native, library)}")
    print(f"Изображения сохранены в {args.output}")


if __name__ == "__main__":
    main()
