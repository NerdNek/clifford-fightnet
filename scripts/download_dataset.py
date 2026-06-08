from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_DIR = PROJECT_ROOT / "src"

if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from clifford_fightnet.utils.paths import ensure_dir, get_project_root


LOGGER = logging.getLogger(__name__)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Prepare local dataset directories for clifford-fightnet."
    )
    parser.add_argument(
        "--destination",
        type=Path,
        default=None,
        help="Optional override for the raw data directory.",
    )
    return parser.parse_args()


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
    args = parse_args()

    repo_root = get_project_root()
    raw_dir = args.destination or repo_root / "data" / "raw"
    processed_dir = repo_root / "data" / "processed"

    ensure_dir(raw_dir)
    ensure_dir(processed_dir)

    LOGGER.info("Prepared dataset directories.")
    LOGGER.info("Raw data directory: %s", raw_dir)
    LOGGER.info("Processed data directory: %s", processed_dir)
    LOGGER.info("No dataset download is implemented yet.")
    LOGGER.info("Place source clips or metadata under the raw directory manually.")
    LOGGER.info("Do not store Kaggle credentials in this repository.")


if __name__ == "__main__":
    main()
