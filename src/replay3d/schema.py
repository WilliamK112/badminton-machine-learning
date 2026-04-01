from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any, Iterable


@dataclass(slots=True)
class CourtSpec:
    length_m: float = 13.4
    width_m: float = 6.1


@dataclass(slots=True)
class Entity3D:
    x: float
    y: float
    z: float

    def as_list(self) -> list[float]:
        return [self.x, self.y, self.z]


@dataclass(slots=True)
class PlayerFrame:
    id: str
    xyz: Entity3D
    bbox_xyxy: list[float] | None = None
    pose2d: list[list[float]] | None = None
    confidence: float | None = None

    def to_dict(self) -> dict[str, Any]:
        out: dict[str, Any] = {
            "id": self.id,
            "xyz": self.xyz.as_list(),
        }
        if self.bbox_xyxy is not None:
            out["bbox_xyxy"] = self.bbox_xyxy
        if self.pose2d is not None:
            out["pose2d"] = self.pose2d
        if self.confidence is not None:
            out["confidence"] = self.confidence
        return out


@dataclass(slots=True)
class ShuttleFrame:
    xyz: Entity3D
    xy_image: list[float] | None = None
    visible: bool = True
    confidence: float | None = None
    velocity_xyz: list[float] | None = None

    def to_dict(self) -> dict[str, Any]:
        out: dict[str, Any] = {
            "xyz": self.xyz.as_list(),
            "visible": self.visible,
        }
        if self.xy_image is not None:
            out["xy_image"] = self.xy_image
        if self.confidence is not None:
            out["confidence"] = self.confidence
        if self.velocity_xyz is not None:
            out["velocity_xyz"] = self.velocity_xyz
        return out


@dataclass(slots=True)
class Replay3DFrame:
    frame: int
    t_sec: float
    fps: float
    court: CourtSpec
    player1: PlayerFrame
    player2: PlayerFrame
    shuttle: ShuttleFrame

    def to_dict(self) -> dict[str, Any]:
        return {
            "frame": self.frame,
            "t_sec": self.t_sec,
            "fps": self.fps,
            "court": asdict(self.court),
            "player1": self.player1.to_dict(),
            "player2": self.player2.to_dict(),
            "shuttle": self.shuttle.to_dict(),
        }


def frames_to_dicts(frames: Iterable[Replay3DFrame]) -> list[dict[str, Any]]:
    return [f.to_dict() for f in frames]
