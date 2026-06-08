from __future__ import annotations

import logging
from pathlib import Path

from clifford_fightnet.config import load_config
from clifford_fightnet.data.dataset import BoxingClipsDataset
from clifford_fightnet.utils.paths import ensure_dir, get_project_root
from clifford_fightnet.utils.reproducibility import set_deterministic_seed


LOGGER = logging.getLogger(__name__)


def train(config_path: str | Path | None = None) -> None:
    project_root = get_project_root()
    selected_config_path = config_path or project_root / "configs" / "default.yaml"
    config = load_config(selected_config_path)
    processed_dir = project_root / config["paths"]["processed_data_dir"]

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
    )

    ensure_dir(project_root / config["paths"]["outputs_dir"])
    set_deterministic_seed(config["seed"])

    dataset = BoxingClipsDataset(
        root_dir=processed_dir,
        split="train",
    )

    if len(dataset) == 0:
        LOGGER.warning(
            "Training dataset is empty at %s. Populate processed data before training.",
            processed_dir,
        )
        return

    LOGGER.info("Dataset contains %d training records.", len(dataset))
    LOGGER.info("TODO: Implement dataloaders, model creation, optimizer setup, and training loop.")


if __name__ == "__main__":
    train()
