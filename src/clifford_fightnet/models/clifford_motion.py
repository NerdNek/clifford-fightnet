from __future__ import annotations

from torch import Tensor, nn


class CliffordMotionModel(nn.Module):
    """Placeholder for the future Clifford-inspired motion classifier."""

    def __init__(self, num_classes: int) -> None:
        super().__init__()
        self.num_classes = num_classes

    def forward(self, inputs: Tensor) -> Tensor:
        _ = inputs
        raise NotImplementedError(
            "TODO: Implement RGB, gradient, temporal, and bivector-style motion features."
        )
