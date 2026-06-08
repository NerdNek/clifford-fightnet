from __future__ import annotations

import logging

from clifford_fightnet.config import AppConfig, load_config
from clifford_fightnet.data.dataset import BoxingClipsDataset


LOGGER = logging.getLogger(__name__)


def evaluate(config: AppConfig | None = None) -> None:
    active_config = config or load_config()

    logging.basicConfig(
        level=getattr(logging, active_config.logging.level.upper(), logging.INFO),
        format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
    )

    dataset = BoxingClipsDataset(
        root_dir=active_config.paths.processed_dir,
        split=active_config.data.test_split,
    )

    if len(dataset) == 0:
        LOGGER.warning(
            "Evaluation dataset is empty at %s. Populate processed data before evaluation.",
            active_config.paths.processed_dir,
        )
        return

    LOGGER.info("Dataset contains %d evaluation records.", len(dataset))
    LOGGER.info("TODO: Implement checkpoint loading, metrics, and evaluation logic.")


if __name__ == "__main__":
    evaluate()
