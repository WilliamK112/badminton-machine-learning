"""3D replay schema and JSONL writer utilities."""

from .schema import (
    CourtSpec,
    Entity3D,
    PlayerFrame,
    ShuttleFrame,
    Replay3DFrame,
)
from .writer import Replay3DJsonlWriter
from .xy_mapper import build_homography_from_corners, map_image_point_to_court

# Lazy imports to avoid hard numpy dependency during test collection
def __getattr__(name):
    if name in ("Replay3DMappingConfig", "convert_tracking_frames_to_replay3d"):
        from . import from_tracking
        return getattr(from_tracking, name)
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")

__all__ = [
    "CourtSpec",
    "Entity3D",
    "PlayerFrame",
    "ShuttleFrame",
    "Replay3DFrame",
    "Replay3DJsonlWriter",
    "Replay3DMappingConfig",
    "build_homography_from_corners",
    "map_image_point_to_court",
    "convert_tracking_frames_to_replay3d",
]
