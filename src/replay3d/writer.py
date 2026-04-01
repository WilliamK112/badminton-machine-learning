from __future__ import annotations

import json
from pathlib import Path
from typing import Iterable

from .schema import Replay3DFrame


class Replay3DJsonlWriter:
    """Write 3D replay frames to newline-delimited JSON (JSONL)."""

    def __init__(self, output_path: str | Path) -> None:
        self.output_path = Path(output_path)
        self.output_path.parent.mkdir(parents=True, exist_ok=True)

    def write_frames(self, frames: Iterable[Replay3DFrame]) -> int:
        count = 0
        with self.output_path.open("w", encoding="utf-8") as f:
            for frame in frames:
                f.write(json.dumps(frame.to_dict(), ensure_ascii=False) + "\n")
                count += 1
        return count
