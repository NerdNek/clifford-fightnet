from __future__ import annotations

import argparse
import logging
import sys
from collections import Counter
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_DIR = PROJECT_ROOT / "src"

if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from clifford_fightnet.utils.paths import get_project_root


LOGGER = logging.getLogger(__name__)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Inspect dataset directory contents without assuming a fixed format."
    )
    parser.add_argument(
        "--data-dir",
        type=Path,
        default=None,
        help="Optional directory to inspect. Defaults to data/raw.",
    )
    return parser.parse_args()


def summarize_files(data_dir: Path) -> Counter[str]:
    suffix_counts: Counter[str] = Counter()
    for path in data_dir.rglob("*"):
        if path.is_file() and not path.name.startswith("."):
            suffix = path.suffix.lower() or "<no_extension>"
            suffix_counts[suffix] += 1
    return suffix_counts


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
    args = parse_args()

    repo_root = get_project_root()
    data_dir = args.data_dir or repo_root / "data" / "raw"

    if not data_dir.exists():
        LOGGER.warning("Dataset directory does not exist yet: %s", data_dir)
        return

    counts = summarize_files(data_dir)
    total_files = sum(counts.values())

    LOGGER.info("Inspecting dataset directory: %s", data_dir)
    LOGGER.info("Total files found: %d", total_files)

    if not counts:
        LOGGER.info("No files found. Add raw dataset files and run this script again.")
        return

    for suffix, count in sorted(counts.items()):
        LOGGER.info("%s: %d", suffix, count)


if __name__ == "__main__":
    main()
