from __future__ import annotations

import argparse
import logging
import shutil
import sys
from pathlib import Path

import kagglehub
from kagglehub import exceptions as kaggle_exceptions

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_DIR = PROJECT_ROOT / "src"

if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from clifford_fightnet.utils.paths import ensure_dir, get_project_root


LOGGER = logging.getLogger(__name__)
DATASET_SLUG = "piotrstefaskiue/olympic-boxing-punch-classification-video-dataset"
PROJECT_DATASET_DIR = Path("data/raw/olympic_boxing")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Download the Olympic Boxing dataset via kagglehub."
    )
    parser.add_argument(
        "--copy-to-project",
        action="store_true",
        help="Copy the downloaded dataset from the Kaggle cache into data/raw/olympic_boxing.",
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="Force a fresh Kaggle download and allow overwriting files in the project copy.",
    )
    return parser.parse_args()


def download_dataset(force: bool) -> Path:
    try:
        cache_path = kagglehub.dataset_download(DATASET_SLUG, force_download=force)
    except (
        kaggle_exceptions.CredentialError,
        kaggle_exceptions.KaggleEnvironmentError,
        kaggle_exceptions.UnauthenticatedError,
    ) as exc:
        LOGGER.error(
            "Kaggle credentials are not available. Configure Kaggle access locally and try again."
        )
        raise SystemExit(1) from exc
    except Exception as exc:
        LOGGER.error("Dataset download failed: %s", exc)
        raise SystemExit(1) from exc

    cache_dir = Path(cache_path)
    if not cache_dir.exists():
        LOGGER.error("kagglehub returned a path that does not exist: %s", cache_dir)
        raise SystemExit(1)

    return cache_dir


def is_non_empty_directory(path: Path) -> bool:
    return path.exists() and any(path.iterdir())


def copy_dataset(source_dir: Path, target_dir: Path, force: bool) -> None:
    if is_non_empty_directory(target_dir) and not force:
        raise FileExistsError(
            f"Target directory already exists and is not empty: {target_dir}. "
            "Use --force to overwrite."
        )

    ensure_dir(target_dir)

    for source_path in source_dir.rglob("*"):
        relative_path = source_path.relative_to(source_dir)
        destination_path = target_dir / relative_path

        if source_path.is_dir():
            destination_path.mkdir(parents=True, exist_ok=True)
            continue

        destination_path.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(source_path, destination_path)


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
    args = parse_args()

    cache_dir = download_dataset(force=args.force)
    print(f"Kaggle cache path: {cache_dir}")

    if not args.copy_to_project:
        LOGGER.info("Dataset downloaded to the Kaggle cache only.")
        return

    project_root = get_project_root()
    target_dir = project_root / PROJECT_DATASET_DIR

    try:
        copy_dataset(cache_dir, target_dir, force=args.force)
    except FileExistsError as exc:
        LOGGER.error("%s", exc)
        raise SystemExit(1) from exc
    except OSError as exc:
        LOGGER.error("Failed to copy dataset into the project directory: %s", exc)
        raise SystemExit(1) from exc

    LOGGER.info("Copied dataset into project directory: %s", target_dir)


if __name__ == "__main__":
    main()
