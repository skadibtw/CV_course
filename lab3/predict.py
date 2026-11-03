import argparse
from pathlib import Path

from PIL import Image, UnidentifiedImageError
import torch

from model import CLASS_NAMES, image_transform, load_model


def main():
    torch.set_num_threads(2)
    root = Path(__file__).resolve().parent
    parser = argparse.ArgumentParser(description="Классификация фотографии листа фасоли")
    parser.add_argument("image", type=Path)
    parser.add_argument("--model", type=Path, default=root / "results/model.pt")
    args = parser.parse_args()
    try:
        model, classes = load_model(args.model)
        with Image.open(args.image) as source:
            image = image_transform()(source.convert("RGB")).unsqueeze(0)
    except (FileNotFoundError, UnidentifiedImageError) as error:
        parser.error(str(error))
    with torch.no_grad():
        probabilities = model(image).softmax(dim=1)[0]
    predicted = probabilities.argmax().item()
    print(f"Ответ: {CLASS_NAMES[classes[predicted]]}")
    for label, probability in zip(classes, probabilities):
        print(f"{CLASS_NAMES[label]}: {probability.item():.1%}")


if __name__ == "__main__":
    main()
