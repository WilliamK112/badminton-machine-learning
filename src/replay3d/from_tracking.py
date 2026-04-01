from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Iterable, Iterator

import numpy as np

from .schema import CourtSpec, Entity3D, PlayerFrame, Replay3DFrame, ShuttleFrame
from .xy_mapper import build_homography_from_corners, map_image_point_to_court


@dataclass(slots=True)
class Replay3DMappingConfig:
    fps: float = 30.0
    image_width: int = 1920
    image_height: int = 1080
    court: CourtSpec = field(default_factory=lambda: CourtSpec(length_m=13.4, width_m=6.1))
    player_outside_margin_m: float = 0.0
    enforce_role_lock: bool = True
    player_max_speed_mps: float = 5.5
    player_ema_alpha: float = 0.6


def _clip_to_court(x: float, y: float, court: CourtSpec) -> tuple[float, float]:
    return (
        float(np.clip(x, 0.0, court.width_m)),
        float(np.clip(y, 0.0, court.length_m)),
    )


def _bbox_bottom_center(bbox_xyxy: list[float]) -> tuple[float, float]:
    x1, y1, x2, y2 = bbox_xyxy
    return (float((x1 + x2) * 0.5), float(y2))


def _player_anchor_xy(
    bbox_xyxy: list[float] | None,
    keypoints: list[list[float]] | None,
) -> tuple[float, float] | None:
    """
    Prefer COCO lower-body joints for foot anchor:
    - ankles (15, 16) first
    - knees (13, 14) fallback
    Then fallback to bbox bottom-center.
    """
    if keypoints:
        def _pick(indices: list[int], min_conf: float = 0.2) -> list[tuple[float, float]]:
            out: list[tuple[float, float]] = []
            for idx in indices:
                if idx >= len(keypoints):
                    continue
                kp = keypoints[idx]
                if not isinstance(kp, (list, tuple)) or len(kp) < 2:
                    continue
                x = float(kp[0])
                y = float(kp[1])
                conf = float(kp[2]) if len(kp) >= 3 and kp[2] is not None else 1.0
                if conf >= min_conf:
                    out.append((x, y))
            return out

        ankles = _pick([15, 16], min_conf=0.15)
        if ankles:
            ax = float(sum(p[0] for p in ankles) / len(ankles))
            ay = float(sum(p[1] for p in ankles) / len(ankles))
            return (ax, ay)

        knees = _pick([13, 14], min_conf=0.2)
        if knees:
            kx = float(sum(p[0] for p in knees) / len(knees))
            ky = float(sum(p[1] for p in knees) / len(knees))
            return (kx, ky + 0.08 * (bbox_xyxy[3] - bbox_xyxy[1]) if bbox_xyxy else ky)

    if bbox_xyxy:
        return _bbox_bottom_center(bbox_xyxy)

    return None


def _limit_step(prev_xy: tuple[float, float], cur_xy: tuple[float, float], max_step_m: float) -> tuple[float, float]:
    dx = cur_xy[0] - prev_xy[0]
    dy = cur_xy[1] - prev_xy[1]
    dist = float((dx * dx + dy * dy) ** 0.5)
    if dist <= max_step_m or dist <= 1e-8:
        return cur_xy
    s = max_step_m / dist
    return (float(prev_xy[0] + dx * s), float(prev_xy[1] + dy * s))


def _shuttle_z_from_speed(prev_xy: tuple[float, float] | None, cur_xy: tuple[float, float]) -> float:
    """MVP fallback Z policy: derive a plausible height from 2D speed."""
    if prev_xy is None:
        return 1.2
    dx = cur_xy[0] - prev_xy[0]
    dy = cur_xy[1] - prev_xy[1]
    speed = float((dx * dx + dy * dy) ** 0.5)
    return float(np.clip(0.6 + 0.015 * speed, 0.2, 6.0))


def _as_tracking_dict(frame_obj: Any) -> dict[str, Any]:
    if hasattr(frame_obj, "to_dict"):
        return frame_obj.to_dict()
    if isinstance(frame_obj, dict):
        return frame_obj
    raise TypeError("tracking frame must be dict-like or provide to_dict()")


def convert_tracking_frames_to_replay3d(
    tracking_frames: Iterable[Any],
    homography: np.ndarray,
    config: Replay3DMappingConfig | None = None,
) -> list[Replay3DFrame]:
    cfg = config or Replay3DMappingConfig()

    replay_frames: list[Replay3DFrame] = []
    prev_shuttle_xy_img: tuple[float, float] | None = None

    # Warm-start prev_player_xy from the first frame's raw mapped positions.
    # Use frame_idx=0 (first sample) as anchor; no EMA on first frame.
    _warm_started = False

    prev_player_xy: dict[int, tuple[float, float]] = {
        1: (cfg.court.width_m * 0.5, cfg.court.length_m * 0.5),
        2: (cfg.court.width_m * 0.5, cfg.court.length_m * 0.5),
    }

    for item in tracking_frames:
        src = _as_tracking_dict(item)
        frame_idx = int(src.get("frame_idx", src.get("frame", 0)))
        players: dict[str, Any] = src.get("players", {}) or {}
        shuttle_xy = src.get("shuttle")

        # Warm-start from first frame's raw positions (no smoothing).
        if not _warm_started and frame_idx == 0:
            for slot_id in [1, 2]:
                info = players.get(str(slot_id)) or players.get(slot_id) or {}
                bbox = info.get("bbox")
                kp = info.get("keypoints")
                anchor = _player_anchor_xy(bbox, kp)
                if anchor is not None:
                    rx, ry = map_image_point_to_court(anchor, homography)
                    rx = float(np.clip(rx, 0.0, cfg.court.width_m))
                    ry = float(np.clip(ry, 0.0, cfg.court.length_m))
                    prev_player_xy[slot_id] = (rx, ry)
            _warm_started = True

        def build_player(slot_id: int) -> PlayerFrame:
            info = players.get(slot_id) or players.get(str(slot_id)) or {}
            bbox = info.get("bbox")
            keypoints = info.get("keypoints")
            confidence = info.get("conf")

            anchor_xy = _player_anchor_xy(bbox, keypoints)
            if anchor_xy is not None:
                raw_x, raw_y = map_image_point_to_court(anchor_xy, homography)
                margin = max(0.0, float(cfg.player_outside_margin_m))
                outside_too_far = (
                    raw_x < -margin
                    or raw_x > (cfg.court.width_m + margin)
                    or raw_y < -margin
                    or raw_y > (cfg.court.length_m + margin)
                )
                if outside_too_far:
                    px, py = prev_player_xy[slot_id]
                else:
                    px_raw, py_raw = _clip_to_court(raw_x, raw_y, cfg.court)
                    prev_xy = prev_player_xy[slot_id]
                    max_step = float(max(cfg.player_max_speed_mps, 0.1) / max(cfg.fps, 1e-6))
                    px_lim, py_lim = _limit_step(prev_xy, (px_raw, py_raw), max_step)
                    a = float(np.clip(cfg.player_ema_alpha, 0.0, 1.0))
                    px = float(a * px_lim + (1.0 - a) * prev_xy[0])
                    py = float(a * py_lim + (1.0 - a) * prev_xy[1])
                    prev_player_xy[slot_id] = (px, py)
            else:
                px, py = prev_player_xy[slot_id]

            return PlayerFrame(
                id=f"P{slot_id}",
                xyz=Entity3D(px, py, 0.0),
                bbox_xyxy=[float(v) for v in bbox] if bbox else None,
                pose2d=keypoints if keypoints else None,
                confidence=float(confidence) if confidence is not None else None,
            )

        p1 = build_player(1)
        p2 = build_player(2)

        if cfg.enforce_role_lock and p1.xyz.y < p2.xyz.y:
            p1, p2 = p2, p1
            p1.id = "P1"
            p2.id = "P2"
            prev_player_xy[1] = (p1.xyz.x, p1.xyz.y)
            prev_player_xy[2] = (p2.xyz.x, p2.xyz.y)

        shuttle_frame: ShuttleFrame
        if shuttle_xy is None or shuttle_xy[0] is None or shuttle_xy[1] is None:
            shuttle_frame = ShuttleFrame(
                xyz=Entity3D(0.0, 0.0, 0.0),
                xy_image=None,
                visible=False,
                confidence=None,
                velocity_xyz=None,
            )
        else:
            sx_img = float(shuttle_xy[0])
            sy_img = float(shuttle_xy[1])
            sx, sy = map_image_point_to_court((sx_img, sy_img), homography)
            sx, sy = _clip_to_court(sx, sy, cfg.court)
            sz = _shuttle_z_from_speed(prev_shuttle_xy_img, (sx_img, sy_img))

            velocity = None
            if prev_shuttle_xy_img is not None and cfg.fps > 0:
                dt = 1.0 / cfg.fps
                vx = (sx_img - prev_shuttle_xy_img[0]) / dt
                vy = (sy_img - prev_shuttle_xy_img[1]) / dt
                velocity = [float(vx), float(vy), 0.0]

            shuttle_frame = ShuttleFrame(
                xyz=Entity3D(sx, sy, sz),
                xy_image=[sx_img, sy_img],
                visible=True,
                confidence=1.0,
                velocity_xyz=velocity,
            )
            prev_shuttle_xy_img = (sx_img, sy_img)

        replay_frames.append(
            Replay3DFrame(
                frame=frame_idx,
                t_sec=float(frame_idx / cfg.fps if cfg.fps > 0 else 0.0),
                fps=cfg.fps,
                court=cfg.court,
                player1=p1,
                player2=p2,
                shuttle=shuttle_frame,
            )
        )

    return replay_frames