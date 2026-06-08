from __future__ import annotations

import torch
from torch import Tensor, nn


class Baseline3DCNN(nn.Module):
    """Placeholder baseline for a future RGB-only video classifier."""

    def __init__(self, num_classes: int) -> None:
        super().__init__()
        self.num_classes = num_classes

    def forward(self, inputs: Tensor) -> Tensor:
        _ = inputs
        raise NotImplementedError(
            "TODO: Implement the baseline 3D CNN once the dataset pipeline is ready."
        )
