from __future__ import annotations

import logging

from clifford_fightnet.config import AppConfig, load_config
from clifford_fightnet.data.dataset import BoxingClipsDataset
from clifford_fightnet.utils.paths import ensure_directory
from clifford_fightnet.utils.reproducibility import set_deterministic_seed


LOGGER = logging.getLogger(__name__)


def train(config: AppConfig | None = None) -> None:
    active_config = config or load_config()

    logging.basicConfig(
        level=getattr(logging, active_config.logging.level.upper(), logging.INFO),
        format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
    )

    ensure_directory(active_config.paths.outputs_dir)
    set_deterministic_seed(active_config.project.seed)

    dataset = BoxingClipsDataset(
        root_dir=active_config.paths.processed_dir,
        split=active_config.data.train_split,
    )

    if len(dataset) == 0:
        LOGGER.warning(
            "Training dataset is empty at %s. Populate processed data before training.",
            active_config.paths.processed_dir,
        )
        return

    LOGGER.info("Dataset contains %d training records.", len(dataset))
    LOGGER.info("TODO: Implement dataloaders, model creation, optimizer setup, and training loop.")


if __name__ == "__main__":
    train()
