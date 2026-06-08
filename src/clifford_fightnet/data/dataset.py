from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import torch
from torch import Tensor
from torch.utils.data import Dataset


@dataclass(slots=True)
class BoxingClipRecord:
    clip_path: Path
    label: int


class BoxingClipsDataset(Dataset[dict[str, Tensor]]):
    """Placeholder dataset for future boxing clip loading logic."""

    def __init__(self, root_dir: Path, split: str) -> None:
        self.root_dir = root_dir
        self.split = split
        self.records = self._index_records()

    def _index_records(self) -> list[BoxingClipRecord]:
        split_dir = self.root_dir / self.split
        if not split_dir.exists():
            return []

        # TODO: Replace with dataset-specific indexing logic once the format is defined.
        return []

    def __len__(self) -> int:
        return len(self.records)

    def __getitem__(self, index: int) -> dict[str, Tensor]:
        if index >= len(self.records):
            raise IndexError(f"Index {index} is out of range for split '{self.split}'.")

        record = self.records[index]

        # TODO: Load actual clip tensors and labels from disk.
        return {
            "video": torch.empty(0),
            "label": torch.tensor(record.label, dtype=torch.long),
        }
