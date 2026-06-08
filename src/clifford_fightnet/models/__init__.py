"""Model definitions for clifford-fightnet."""

from clifford_fightnet.models.baseline_3dcnn import Baseline3DCNN
from clifford_fightnet.models.clifford_motion import CliffordMotionModel

__all__ = ["Baseline3DCNN", "CliffordMotionModel"]
