from __future__ import annotations

import logging
from pathlib import Path

from clifford_fightnet.config import load_config
from clifford_fightnet.data.dataset import BoxingClipsDataset
from clifford_fightnet.utils.paths import get_project_root


LOGGER = logging.getLogger(__name__)


def evaluate(config_path: str | Path | None = None) -> None:
    project_root = get_project_root()
    selected_config_path = config_path or project_root / "configs" / "default.yaml"
    config = load_config(selected_config_path)
    processed_dir = project_root / config["paths"]["processed_data_dir"]

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
    )

    dataset = BoxingClipsDataset(
        root_dir=processed_dir,
        split="test",
    )

    if len(dataset) == 0:
        LOGGER.warning(
            "Evaluation dataset is empty at %s. Populate processed data before evaluation.",
            processed_dir,
        )
        return

    LOGGER.info("Dataset contains %d evaluation records.", len(dataset))
    LOGGER.info("TODO: Implement checkpoint loading, metrics, and evaluation logic.")


if __name__ == "__main__":
    evaluate()
