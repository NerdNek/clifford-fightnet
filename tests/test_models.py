from __future__ import annotations

import sys
from pathlib import Path

import torch


PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_DIR = PROJECT_ROOT / "src"

if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))


def test_baseline_3dcnn_output_shape() -> None:
    from clifford_fightnet.models import Baseline3DCNN

    model = Baseline3DCNN(num_classes=4, in_channels=3)
    inputs = torch.randn(2, 3, 16, 112, 112)

    outputs = model(inputs)

    assert outputs.shape == (2, 4)
