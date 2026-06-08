from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import yaml

from clifford_fightnet.utils.paths import get_repo_root


@dataclass(slots=True)
class ProjectConfig:
    name: str
    seed: int


@dataclass(slots=True)
class PathsConfig:
    data_dir: Path
    raw_dir: Path
    processed_dir: Path
    outputs_dir: Path


@dataclass(slots=True)
class DataConfig:
    train_split: str
    val_split: str
    test_split: str
    clip_length: int
    frame_stride: int
    image_size: int
    num_workers: int


@dataclass(slots=True)
class ModelConfig:
    name: str
    num_classes: int
    input_channels: int
    feature_channels: dict[str, bool]


@dataclass(slots=True)
class TrainingConfig:
    batch_size: int
    epochs: int
    learning_rate: float
    weight_decay: float


@dataclass(slots=True)
class LoggingConfig:
    level: str


@dataclass(slots=True)
class AppConfig:
    project: ProjectConfig
    paths: PathsConfig
    data: DataConfig
    model: ModelConfig
    training: TrainingConfig
    logging: LoggingConfig


def _resolve_path(repo_root: Path, value: str) -> Path:
    path = Path(value)
    return path if path.is_absolute() else repo_root / path


def load_config(config_path: Path | None = None) -> AppConfig:
    repo_root = get_repo_root()
    selected_path = config_path or repo_root / "configs" / "default.yaml"

    with selected_path.open("r", encoding="utf-8") as config_file:
        raw_config: dict[str, Any] = yaml.safe_load(config_file)

    return AppConfig(
        project=ProjectConfig(**raw_config["project"]),
        paths=PathsConfig(
            data_dir=_resolve_path(repo_root, raw_config["paths"]["data_dir"]),
            raw_dir=_resolve_path(repo_root, raw_config["paths"]["raw_dir"]),
            processed_dir=_resolve_path(repo_root, raw_config["paths"]["processed_dir"]),
            outputs_dir=_resolve_path(repo_root, raw_config["paths"]["outputs_dir"]),
        ),
        data=DataConfig(**raw_config["data"]),
        model=ModelConfig(**raw_config["model"]),
        training=TrainingConfig(**raw_config["training"]),
        logging=LoggingConfig(**raw_config["logging"]),
    )
