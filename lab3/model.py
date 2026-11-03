import torch
from torch import nn
from torchvision import transforms


IMAGE_SIZE = 96
CLASSES = ["angular_leaf_spot", "bean_rust", "healthy"]
CLASS_NAMES = {
    "angular_leaf_spot": "пятнистость",
    "bean_rust": "ржавчина",
    "healthy": "здоровый лист",
}


def image_transform(training=False):
    steps = [transforms.Resize((IMAGE_SIZE, IMAGE_SIZE))]
    if training:
        steps += [transforms.RandomHorizontalFlip(), transforms.RandomRotation(15)]
    return transforms.Compose(steps + [transforms.ToTensor()])


class SimpleCNN(nn.Module):
    def __init__(self):
        super().__init__()
        self.layers = nn.Sequential(
            nn.Conv2d(3, 8, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Conv2d(8, 16, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Conv2d(16, 32, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Flatten(),
            nn.Linear(32 * 12 * 12, 32),
            nn.ReLU(),
            nn.Linear(32, len(CLASSES)),
        )

    def forward(self, images):
        return self.layers(images)


def load_model(path):
    checkpoint = torch.load(path, map_location="cpu", weights_only=True)
    model = SimpleCNN()
    model.load_state_dict(checkpoint["state_dict"])
    model.eval()
    return model, checkpoint["classes"]
