from __future__ import annotations

from torch import nn
import torch.nn.functional as F


class CifarCnn(nn.Module):
    """Small CNN for CIFAR-10 distributed training demos."""

    def __init__(self, num_classes: int = 10) -> None:
        super().__init__()
        self.conv1 = nn.Conv2d(3, 32, 3, padding=1)
        self.conv2 = nn.Conv2d(32, 64, 3, padding=1)
        self.conv3 = nn.Conv2d(64, 128, 3, padding=1)
        self.pool = nn.MaxPool2d(2, 2)
        self.fc1 = nn.Linear(128 * 4 * 4, 256)
        self.fc2 = nn.Linear(256, num_classes)
        self.dropout = nn.Dropout(0.4)

    def forward(self, inputs):
        outputs = self.pool(F.relu(self.conv1(inputs)))
        outputs = self.pool(F.relu(self.conv2(outputs)))
        outputs = self.pool(F.relu(self.conv3(outputs)))
        outputs = outputs.view(outputs.size(0), -1)
        outputs = self.dropout(F.relu(self.fc1(outputs)))
        return self.fc2(outputs)
