from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Callable

import cv2
import numpy as np
import torch
from torch import Tensor
from torch.utils.data import Dataset


VIDEO_EXTENSIONS = {".mp4", ".avi", ".mov", ".mkv"}


@dataclass(slots=True)
class VideoSample:
    video_path: Path
    label_name: str
    label_idx: int


class BoxingVideoDataset(Dataset[dict[str, Tensor | str]]):
    """Lightweight video dataset for boxing clips with flexible path scanning."""

    def __init__(
        self,
        root_dir: str | Path,
        num_frames: int = 16,
        image_size: int = 112,
        transform: Callable[[Tensor], Tensor] | None = None,
        class_to_idx: dict[str, int] | None = None,
    ) -> None:
        self.root_dir = Path(root_dir)
        self.num_frames = num_frames
        self.image_size = image_size
        self.transform = transform

        if not self.root_dir.exists():
            raise FileNotFoundError(f"Dataset root directory does not exist: {self.root_dir}")
        if not self.root_dir.is_dir():
            raise NotADirectoryError(f"Dataset root path is not a directory: {self.root_dir}")

        self.video_paths = self._scan_video_files()
        self.class_to_idx = class_to_idx or self._build_class_index(self.video_paths)
        self.samples = self._build_samples()

    def _scan_video_files(self) -> list[Path]:
        return sorted(
            path
            for path in self.root_dir.rglob("*")
            if path.is_file() and path.suffix.lower() in VIDEO_EXTENSIONS
        )

    def _build_class_index(self, video_paths: list[Path]) -> dict[str, int]:
        label_names = sorted({self._infer_label_from_path(video_path) for video_path in video_paths})
        return {label_name: index for index, label_name in enumerate(label_names)}

    def _build_samples(self) -> list[VideoSample]:
        samples: list[VideoSample] = []
        for video_path in self.video_paths:
            label_name = self._infer_label_from_path(video_path)
            if label_name not in self.class_to_idx:
                continue

            samples.append(
                VideoSample(
                    video_path=video_path,
                    label_name=label_name,
                    label_idx=self.class_to_idx[label_name],
                )
            )
        return samples

    def _infer_label_from_path(self, video_path: Path) -> str:
        return video_path.parent.name

    def _load_video(self, video_path: Path) -> Tensor:
        capture = cv2.VideoCapture(str(video_path))
        if not capture.isOpened():
            raise RuntimeError(f"Could not open video file: {video_path}")

        frames: list[np.ndarray] = []
        try:
            while True:
                success, frame = capture.read()
                if not success:
                    break

                frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                frame = cv2.resize(frame, (self.image_size, self.image_size))
                frames.append(frame)
        finally:
            capture.release()

        if not frames:
            raise RuntimeError(f"No frames could be read from video file: {video_path}")

        # We sample evenly across longer clips so the model sees the whole motion,
        # and we repeat the last frame for short clips so every sample has T frames.
        if len(frames) >= self.num_frames:
            frame_indices = np.linspace(0, len(frames) - 1, num=self.num_frames, dtype=int)
            selected_frames = [frames[index] for index in frame_indices]
        else:
            selected_frames = list(frames)
            last_frame = frames[-1]
            selected_frames.extend([last_frame.copy() for _ in range(self.num_frames - len(frames))])

        video_array = np.stack(selected_frames, axis=0)
        video_tensor = torch.from_numpy(video_array).permute(3, 0, 1, 2).float() / 255.0
        return video_tensor

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, index: int) -> dict[str, Tensor | str]:
        sample = self.samples[index]
        video = self._load_video(sample.video_path)

        if self.transform is not None:
            video = self.transform(video)

        return {
            "video": video,
            "label": torch.tensor(sample.label_idx, dtype=torch.long),
            "label_name": sample.label_name,
            "video_path": str(sample.video_path),
        }


# Backward-compatible alias while the training placeholders are still evolving.
BoxingClipsDataset = BoxingVideoDataset
