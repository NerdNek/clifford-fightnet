from __future__ import annotations

import sys
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_DIR = PROJECT_ROOT / "src"

if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))


def test_package_imports() -> None:
    import clifford_fightnet
    from clifford_fightnet.config import load_config
    from clifford_fightnet.data import BoxingClipsDataset
    from clifford_fightnet.models import Baseline3DCNN, CliffordMotionModel
    from clifford_fightnet.training import evaluate, train
    from clifford_fightnet.utils import get_repo_root, set_deterministic_seed

    assert clifford_fightnet.__version__ == "0.1.0"
    assert callable(load_config)
    assert BoxingClipsDataset is not None
    assert Baseline3DCNN is not None
    assert CliffordMotionModel is not None
    assert callable(train)
    assert callable(evaluate)
    assert callable(get_repo_root)
    assert callable(set_deterministic_seed)


def test_default_config_loads() -> None:
    from clifford_fightnet.config import load_config

    config = load_config()

    assert config.project.name == "clifford-fightnet"
    assert config.project.seed == 42
    assert config.paths.raw_dir.name == "raw"
    assert config.model.name == "clifford_motion"
