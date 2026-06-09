from __future__ import annotations

from torch import Tensor, nn


class Baseline3DCNN(nn.Module):
    """A small RGB-only 3D CNN baseline for boxing video classification."""

    def __init__(self, num_classes: int, in_channels: int = 3) -> None:
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv3d(in_channels, 32, kernel_size=3, padding=1),
            nn.BatchNorm3d(32),
            nn.ReLU(inplace=True),
            nn.MaxPool3d(kernel_size=(1, 2, 2)),
            nn.Conv3d(32, 64, kernel_size=3, padding=1),
            nn.BatchNorm3d(64),
            nn.ReLU(inplace=True),
            nn.MaxPool3d(kernel_size=2),
            nn.Conv3d(64, 128, kernel_size=3, padding=1),
            nn.BatchNorm3d(128),
            nn.ReLU(inplace=True),
            nn.AdaptiveAvgPool3d((1, 1, 1)),
        )
        self.classifier = nn.Linear(128, num_classes)

    def forward(self, inputs: Tensor) -> Tensor:
        features = self.features(inputs)
        flattened = features.flatten(start_dim=1)
        return self.classifier(flattened)
