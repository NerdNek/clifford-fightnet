from __future__ import annotations

import argparse
import sys
from pathlib import Path

import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_DIR = PROJECT_ROOT / "src"

if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from clifford_fightnet.utils.paths import get_project_root


VIDEO_EXTENSIONS = {".mp4", ".avi", ".mov", ".mkv"}
IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".webp", ".tif", ".tiff"}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Inspect the Olympic Boxing dataset structure before dataset implementation."
    )
    parser.add_argument(
        "--data-dir",
        type=Path,
        default=None,
        help="Optional dataset directory. Defaults to data/raw/olympic_boxing.",
    )
    return parser.parse_args()


def collect_files_by_extension(data_dir: Path, extensions: set[str]) -> list[Path]:
    return sorted(
        path for path in data_dir.rglob("*") if path.is_file() and path.suffix.lower() in extensions
    )


def print_top_level_entries(data_dir: Path) -> None:
    entries = sorted(data_dir.iterdir(), key=lambda entry: (entry.is_file(), entry.name.lower()))
    print("Top-level files/folders:")

    if not entries:
        print("  (empty directory)")
        return

    for entry in entries:
        entry_type = "dir" if entry.is_dir() else "file"
        print(f"  [{entry_type}] {entry.name}")


def print_video_preview(video_files: list[Path], data_dir: Path) -> None:
    print(f"Number of video files: {len(video_files)}")
    print("First 20 video file paths:")

    if not video_files:
        print("  (no video files found)")
        return

    for video_path in video_files[:20]:
        print(f"  {video_path.relative_to(data_dir)}")


def print_folder_video_counts(data_dir: Path, video_files: list[Path]) -> None:
    class_folders = sorted(path for path in data_dir.iterdir() if path.is_dir())
    print("Folder-wise video counts:")

    if not class_folders:
        print("  (no top-level folders found)")
        return

    folder_counts: list[tuple[str, int]] = []
    for folder in class_folders:
        count = sum(1 for video_path in video_files if folder in video_path.parents)
        folder_counts.append((folder.name, count))

    if not any(count > 0 for _, count in folder_counts):
        print("  (no class-like folders with video files found)")
        return

    for folder_name, count in folder_counts:
        if count > 0:
            print(f"  {folder_name}: {count}")


def print_csv_summary(csv_files: list[Path], data_dir: Path) -> None:
    print(f"Number of CSV files: {len(csv_files)}")

    if not csv_files:
        print("CSV files: none found")
        return

    print("CSV files:")
    for csv_path in csv_files:
        relative_path = csv_path.relative_to(data_dir)
        try:
            dataframe = pd.read_csv(csv_path)
            print(f"  {relative_path} -> shape={dataframe.shape}")
        except Exception as exc:
            print(f"  {relative_path} -> failed to read ({exc})")


def main() -> None:
    args = parse_args()
    project_root = get_project_root()
    data_dir = args.data_dir or project_root / "data" / "raw" / "olympic_boxing"

    print(f"Dataset path: {data_dir}")
    print(f"Path exists: {data_dir.exists()}")

    if not data_dir.exists():
        print("Dataset directory was not found.")
        print("Run `python scripts/download_dataset.py --copy-to-project` first.")
        return

    folder_count = sum(1 for path in data_dir.rglob("*") if path.is_dir())
    video_files = collect_files_by_extension(data_dir, VIDEO_EXTENSIONS)
    csv_files = collect_files_by_extension(data_dir, {".csv"})
    image_files = collect_files_by_extension(data_dir, IMAGE_EXTENSIONS)

    print_top_level_entries(data_dir)
    print(f"Number of folders: {folder_count}")
    print(f"Number of image files: {len(image_files)}")
    print_video_preview(video_files, data_dir)
    print_folder_video_counts(data_dir, video_files)
    print_csv_summary(csv_files, data_dir)


if __name__ == "__main__":
    main()
