"""The convolutional model introduced and shape-traced in session 22."""

from torch import nn


def cnn(dropout=0.0):
    return nn.Sequential(
        nn.Conv2d(1, 8, kernel_size=3, padding=1), nn.ReLU(), nn.MaxPool2d(2),
        nn.Conv2d(8, 16, kernel_size=3, padding=1), nn.ReLU(), nn.MaxPool2d(2),
        nn.Flatten(start_dim=1), nn.Dropout(dropout), nn.Linear(16 * 7 * 7, 10),
    )
