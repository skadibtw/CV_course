"""Проверка сохранённой модели и картинки для отчёта. Здесь нет обучения."""

import argparse
import csv
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader
from torchvision.datasets import ImageFolder

from model import CLASS_NAMES, image_transform, load_model
from train import evaluate


def main():
    root = Path(__file__).resolve().parent
    parser = argparse.ArgumentParser(description="Проверка модели и построение графиков")
    parser.add_argument("--data", type=Path, default=root / "data")
    parser.add_argument("--output", type=Path, default=root / "results")
    args = parser.parse_args()
    torch.set_num_threads(2)
    model, classes = load_model(args.output / "model.pt")
    names = [CLASS_NAMES[name] for name in classes]

    # Проверяем одну сохранённую модель на всех трёх выборках.
    results = {}
    for split in ["train", "val", "test"]:
        dataset = ImageFolder(args.data / split, transform=image_transform())
        if dataset.classes != classes:
            parser.error(f"Классы в {split} не совпадают с классами модели")
        if split == "test" and len(dataset) < 50:
            parser.error("В тесте должно быть хотя бы 50 изображений")
        loss, accuracy = evaluate(model, DataLoader(dataset, batch_size=64), nn.CrossEntropyLoss())
        correct = round(accuracy * len(dataset))
        results[split] = dict(images=len(dataset), correct=correct, loss=loss, accuracy=accuracy)
        print(f"{split}: {correct} из {len(dataset)} ({accuracy:.2%})")

    test_data = dataset
    answers = []
    matrix = np.zeros((3, 3), dtype=int)
    with torch.no_grad():
        for index, (image, label) in enumerate(test_data):
            probabilities = model(image.unsqueeze(0)).softmax(dim=1)[0]
            guess = probabilities.argmax().item()
            answers.append((index, label, guess, probabilities[guess].item()))
            matrix[label, guess] += 1
    with (args.output / "test_predictions.csv").open("w", newline="", encoding="utf-8") as file:
        writer = csv.writer(file)
        writer.writerow(["path", "label", "prediction"])
        for index, label, guess, _ in answers:
            path = Path(test_data.samples[index][0]).relative_to(args.data)
            writer.writerow([path.as_posix(), label, guess])

    with (args.output / "history.csv").open(encoding="utf-8") as file:
        history = list(csv.DictReader(file))
    checkpoint = torch.load(args.output / "model.pt", map_location="cpu", weights_only=True)
    metrics = dict(classes=classes, epochs=len(history), best_epoch=checkpoint["epoch"],
                   parameters=sum(p.numel() for p in model.parameters()), device="cpu",
                   metrics=results, test_confusion=matrix.tolist())
    (args.output / "metrics.json").write_text(json.dumps(metrics, indent=2) + "\n", encoding="utf-8")

    # Графики обучения и валидации.
    epochs = [int(row["epoch"]) for row in history]
    figure, axes = plt.subplots(1, 2, figsize=(10, 4))
    for split, name in [("train", "Обучение"), ("val", "Валидация")]:
        axes[0].plot(epochs, [float(row[f"{split}_loss"]) for row in history], label=name)
        axes[1].plot(epochs, [100 * float(row[f"{split}_accuracy"]) for row in history], label=name)
    axes[0].set_title("Ошибка сети")
    axes[1].set_title("Доля правильных ответов, %")
    for axis in axes:
        axis.set_xlabel("Эпоха")
        axis.grid(alpha=0.3)
        axis.legend()
    figure.tight_layout()
    figure.savefig(args.output / "training.png", dpi=150)
    plt.close(figure)

    # Таблица правильных и ошибочных ответов на тесте.
    figure, axis = plt.subplots(figsize=(7, 6))
    axis.imshow(matrix, cmap="Blues", vmin=0)
    axis.set_xticks(range(3), names)
    axis.set_yticks(range(3), names)
    axis.set_xlabel("Ответ сети")
    axis.set_ylabel("Правильный класс")
    axis.set_title("Результат на тестовых изображениях")
    for i in range(3):
        for j in range(3):
            color = "white" if matrix[i, j] > matrix.max() / 2 else "black"
            axis.text(j, i, str(matrix[i, j]), ha="center", va="center", color=color)
    figure.tight_layout()
    figure.savefig(args.output / "confusion.png", dpi=150)
    plt.close(figure)

    # Для каждого класса показываем правильный ответ и ошибку, если она есть.
    figure, axes = plt.subplots(2, 3, figsize=(10, 7))
    for label in range(3):
        class_answers = [answer for answer in answers if answer[1] == label]
        good = [answer for answer in class_answers if answer[1] == answer[2]]
        bad = [answer for answer in class_answers if answer[1] != answer[2]]
        examples = [good[0] if good else class_answers[0], bad[0] if bad else class_answers[-1]]
        for row, (index, actual, guess, probability) in enumerate(examples):
            image, _ = test_data[index]
            axis = axes[row, label]
            axis.imshow(image.permute(1, 2, 0))
            axis.set_title(f"{names[actual]} → {names[guess]}\n{probability:.1%}",
                           color="green" if actual == guess else "red", fontsize=10)
            axis.axis("off")
    figure.tight_layout()
    figure.savefig(args.output / "predictions.png", dpi=150)
    plt.close(figure)


if __name__ == "__main__":
    main()
