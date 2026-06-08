"""Training and evaluation entry points."""

from clifford_fightnet.training.evaluate import evaluate
from clifford_fightnet.training.train import train

__all__ = ["train", "evaluate"]
