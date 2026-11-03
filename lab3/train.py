import argparse
import csv
from pathlib import Path

import torch
from torch import nn
from torch.utils.data import DataLoader
from torchvision.datasets import ImageFolder

from model import SimpleCNN, image_transform


def evaluate(model, loader, criterion):
    model.eval()
    total_loss = 0
    correct = 0
    with torch.no_grad():
        for images, labels in loader:
            outputs = model(images)
            total_loss += criterion(outputs, labels).item() * len(labels)
            correct += (outputs.argmax(dim=1) == labels).sum().item()
    return total_loss / len(loader.dataset), correct / len(loader.dataset)


def main():
    root = Path(__file__).resolve().parent
    parser = argparse.ArgumentParser(description="Обучение сети на фотографиях листьев")
    parser.add_argument("--data", type=Path, default=root / "data")
    parser.add_argument("--output", type=Path, default=root / "results")
    parser.add_argument("--epochs", type=int, default=20)
    args = parser.parse_args()
    if args.epochs < 1:
        parser.error("Число эпох должно быть положительным")

    torch.manual_seed(42)
    torch.set_num_threads(2)

    # 1. Загружаем картинки и объединяем их в небольшие группы.
    train_data = ImageFolder(args.data / "train", transform=image_transform(training=True))
    val_data = ImageFolder(args.data / "val", transform=image_transform())
    if train_data.classes != val_data.classes or len(train_data.classes) != 3:
        parser.error("В обучении и валидации должны быть одинаковые три класса")
    train_loader = DataLoader(train_data, batch_size=32, shuffle=True)
    val_loader = DataLoader(val_data, batch_size=64)

    # 2. Создаём сеть, функцию ошибки и способ обновления весов.
    model = SimpleCNN()
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
    args.output.mkdir(parents=True, exist_ok=True)
    best_loss = float("inf")
    history = []

    # 3. Повторяем обучение несколько эпох.
    for epoch in range(1, args.epochs + 1):
        model.train()
        total_loss = 0
        correct = 0
        for images, labels in train_loader:
            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            total_loss += loss.item() * len(labels)
            correct += (outputs.argmax(dim=1) == labels).sum().item()

        # 4. Проверяем сеть на картинках, на которых она не обучается.
        val_loss, val_accuracy = evaluate(model, val_loader, criterion)
        history.append({"epoch": epoch,
                        "train_loss": total_loss / len(train_data),
                        "train_accuracy": correct / len(train_data),
                        "val_loss": val_loss, "val_accuracy": val_accuracy})
        print(f"Эпоха {epoch}: train={correct / len(train_data):.1%}, "
              f"val={val_accuracy:.1%}, ошибка={val_loss:.4f}", flush=True)

        # 5. Сохраняем модель, если ошибка на валидации стала меньше.
        if val_loss < best_loss:
            best_loss = val_loss
            torch.save({"state_dict": model.state_dict(),
                        "classes": train_data.classes, "epoch": epoch},
                       args.output / "model.pt")

    with (args.output / "history.csv").open("w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=list(history[0]))
        writer.writeheader()
        writer.writerows(history)
    print(f"Модель и история обучения сохранены в {args.output}")


if __name__ == "__main__":
    main()
