from __future__ import annotations

from typing import Callable

from torch import nn
import torch.nn.functional as F
from torchvision import models

MODEL_CHOICES = ("cifar_cnn", "resnet50", "resnet101", "vit_b_16")


class CifarCnn(nn.Module):
    """Small CNN for CIFAR-10 distributed training demos."""

    def __init__(self, num_classes: int = 10) -> None:
        super().__init__()
        self.conv1 = nn.Conv2d(3, 32, 3, padding=1)
        self.conv2 = nn.Conv2d(32, 64, 3, padding=1)
        self.conv3 = nn.Conv2d(64, 128, 3, padding=1)
        self.pool = nn.MaxPool2d(2, 2)
        self.feature_pool = nn.AdaptiveAvgPool2d((4, 4))
        self.fc1 = nn.Linear(128 * 4 * 4, 256)
        self.fc2 = nn.Linear(256, num_classes)
        self.dropout = nn.Dropout(0.4)

    def forward(self, inputs):
        outputs = self.pool(F.relu(self.conv1(inputs)))
        outputs = self.pool(F.relu(self.conv2(outputs)))
        outputs = self.pool(F.relu(self.conv3(outputs)))
        outputs = self.feature_pool(outputs)
        outputs = outputs.view(outputs.size(0), -1)
        outputs = self.dropout(F.relu(self.fc1(outputs)))
        return self.fc2(outputs)


def build_model(name: str, classes: int, image_size: int | None = None) -> nn.Module:
    model_name = name.lower()
    if model_name == "cifar_cnn":
        return CifarCnn(num_classes=classes)
    if model_name == "resnet50":
        model = models.resnet50(weights=None)
        model.fc = nn.Linear(model.fc.in_features, classes)
        return model
    if model_name == "resnet101":
        model = models.resnet101(weights=None)
        model.fc = nn.Linear(model.fc.in_features, classes)
        return model
    if model_name == "vit_b_16":
        kwargs = {"num_classes": classes}
        if image_size is not None:
            kwargs["image_size"] = image_size
        return models.vit_b_16(weights=None, **kwargs)
    raise ValueError(f"unsupported model: {name}")


def model_factory(name: str, classes: int, image_size: int | None = None) -> Callable[[], nn.Module]:
    return lambda: build_model(name, classes, image_size)
