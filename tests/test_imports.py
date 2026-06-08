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
    from clifford_fightnet.constants import BOXING_LABELS
    from clifford_fightnet.utils import ensure_dir, get_project_root, set_deterministic_seed

    assert clifford_fightnet.__version__ == "0.1.0"
    assert callable(load_config)
    assert isinstance(BOXING_LABELS, list)
    assert callable(get_project_root)
    assert callable(ensure_dir)
    assert callable(set_deterministic_seed)


def test_default_config_loads() -> None:
    from clifford_fightnet.config import load_config

    config = load_config(PROJECT_ROOT / "configs" / "default.yaml")

    assert config["project_name"] == "clifford-fightnet"
    assert config["seed"] == 42
    assert config["model"]["num_frames"] == 16
    assert config["training"]["batch_size"] == 8


def test_project_root_resolves() -> None:
    from clifford_fightnet.utils.paths import get_project_root

    assert get_project_root() == PROJECT_ROOT


def test_ensure_dir_creates_directory(tmp_path: Path) -> None:
    from clifford_fightnet.utils.paths import ensure_dir

    target_dir = tmp_path / "artifacts" / "nested"
    created_dir = ensure_dir(target_dir)

    assert created_dir == target_dir
    assert created_dir.exists()
    assert created_dir.is_dir()
