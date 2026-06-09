from __future__ import annotations

import sys
from pathlib import Path

import cv2
import numpy as np
import torch


PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_DIR = PROJECT_ROOT / "src"

if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))


def create_test_video(video_path: Path, frame_count: int, frame_size: tuple[int, int] = (24, 24)) -> None:
    video_path.parent.mkdir(parents=True, exist_ok=True)
    writer = cv2.VideoWriter(
        str(video_path),
        cv2.VideoWriter_fourcc(*"MJPG"),
        5.0,
        frame_size,
    )

    if not writer.isOpened():
        raise RuntimeError(f"Could not create test video at {video_path}")

    try:
        for frame_index in range(frame_count):
            frame = np.full(
                (frame_size[1], frame_size[0], 3),
                fill_value=min(frame_index * 40, 255),
                dtype=np.uint8,
            )
            writer.write(frame)
    finally:
        writer.release()


def test_boxing_video_dataset_scans_videos_and_loads_item(tmp_path: Path) -> None:
    from clifford_fightnet.data.dataset import BoxingVideoDataset

    create_test_video(tmp_path / "jab" / "jab_clip.avi", frame_count=8)
    create_test_video(tmp_path / "cross" / "cross_clip.avi", frame_count=10)

    dataset = BoxingVideoDataset(root_dir=tmp_path, num_frames=6, image_size=32)

    assert len(dataset) == 2
    assert set(dataset.class_to_idx) == {"jab", "cross"}

    sample = dataset[0]

    assert isinstance(sample["video"], torch.Tensor)
    assert sample["video"].shape == (3, 6, 32, 32)
    assert sample["video"].dtype == torch.float32
    assert 0.0 <= float(sample["video"].min()) <= 1.0
    assert 0.0 <= float(sample["video"].max()) <= 1.0
    assert sample["label_name"] in {"jab", "cross"}


def test_boxing_video_dataset_pads_short_videos(tmp_path: Path) -> None:
    from clifford_fightnet.data.dataset import BoxingVideoDataset

    create_test_video(tmp_path / "hook" / "short_clip.avi", frame_count=3)
    dataset = BoxingVideoDataset(root_dir=tmp_path, num_frames=5, image_size=16)

    video = dataset[0]["video"]

    assert torch.allclose(video[:, 2], video[:, 3])
    assert torch.allclose(video[:, 3], video[:, 4])


def test_boxing_video_dataset_validates_root_directory(tmp_path: Path) -> None:
    from clifford_fightnet.data.dataset import BoxingVideoDataset

    missing_dir = tmp_path / "missing_dataset"

    try:
        BoxingVideoDataset(root_dir=missing_dir)
    except FileNotFoundError:
        pass
    else:
        raise AssertionError("Expected FileNotFoundError for a missing dataset directory.")
