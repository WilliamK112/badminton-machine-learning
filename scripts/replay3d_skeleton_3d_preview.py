#!/usr/bin/env python3
"""
Render 3D skeleton preview (PNG + GIF) from replay_3d.jsonl.

Each COCO-17 keypoint is lifted from 2D image space to 3D court space:
  - (X, Y) via homography warp using per-video corners
  - Z via fixed anatomical height model per keypoint index

Skeleton limbs follow COCO-17 connectivity.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import numpy as np

# COCO-17 keypoint names and anatomical Z-heights (meters above court surface)
COCO_KEYPOINT_NAMES = [
    "nose", "l_eye", "r_eye", "l_ear", "r_ear",
    "l_shoulder", "r_shoulder", "l_elbow", "r_elbow",
    "l_wrist", "r_wrist", "l_hip", "r_hip",
    "l_knee", "r_knee", "l_ankle", "r_ankle",
]

# Fixed Z-height model: [min_z, max_z] per keypoint (used as-is, ankles=ground)
COCO_Z_HEIGHTS = [
    [1.60, 1.75],  # 0: nose
    [1.65, 1.78],  # 1: l_eye
    [1.65, 1.78],  # 2: r_eye
    [1.62, 1.76],  # 3: l_ear
    [1.62, 1.76],  # 4: r_ear
    [1.30, 1.45],  # 5: l_shoulder
    [1.30, 1.45],  # 6: r_shoulder
    [0.95, 1.10],  # 7: l_elbow
    [0.95, 1.10],  # 8: r_elbow
    [0.80, 0.95],  # 9: l_wrist
    [0.80, 0.95],  # 10: r_wrist
    [0.85, 1.00],  # 11: l_hip
    [0.85, 1.00],  # 12: r_hip
    [0.45, 0.60],  # 13: l_knee
    [0.45, 0.60],  # 14: r_knee
    [0.00, 0.12],  # 15: l_ankle
    [0.00, 0.12],  # 16: r_ankle
]

# COCO-17 skeleton limb pairs: (src_idx, dst_idx)
COCO_LIMBS = [
    (0, 1), (0, 2), (1, 3), (2, 4),  # face
    (5, 6),  # shoulders
    (5, 7), (7, 9),   # left arm
    (6, 8), (8, 10),  # right arm
    (5, 11), (6, 12),  # torso
    (11, 12),  # hip line
    (11, 13), (13, 15),  # left leg
    (12, 14), (14, 16),  # right leg
]


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Render 3D skeleton preview (PNG/GIF) from replay_3d.jsonl with COCO limb connections"
    )
    p.add_argument("--input", required=True, help="Path to replay_3d.jsonl")
    p.add_argument("--out-dir", required=True, help="Output directory")
    p.add_argument("--corners-json", default="runs/replay3d_corners_template.json",
                   help="Path to JSON containing image_corners for homography [default: runs/replay3d_corners_template.json]")
    p.add_argument("--gif-step", type=int, default=3, help="Sample every N frame(s) for GIF [default: 3]")
    p.add_argument("--elev", type=float, default=22.0, help="3D camera elevation [default: 22.0]")
    p.add_argument("--azim", type=float, default=-50.0, help="3D camera azimuth [default: -50.0]")
    p.add_argument("--z-max", type=float, default=7.0, help="Z-axis upper bound [default: 7.0]")
    p.add_argument("--no-gif", action="store_true", help="Only export PNG summary")
    p.add_argument("--visible-threshold", type=float, default=0.3,
                   help="Min confidence for a keypoint to be drawn [default: 0.3]")
    p.add_argument("--trail", type=int, default=20,
                   help="Trail length for center-position dots [default: 20]")
    return p.parse_args()


def load_corners(path: str | Path) -> list[list[float]]:
    p = Path(path)
    if not p.exists():
        raise FileNotFoundError(f"Corners file not found: {p}")
    with p.open() as f:
        data = json.load(f)
    corners = data.get("image_corners")
    if not corners or len(corners) != 4:
        raise ValueError("corners-json must contain image_corners: [[bl_x,bl_y],[br_x,br_y],[tr_x,tr_y],[tl_x,tl_y]]")
    return [[float(x), float(y)] for x, y in corners]


def build_homography(corners_image_xy: list[list[float]], court_w: float = 6.1, court_l: float = 13.4) -> np.ndarray:
    """Build DLT homography: image px -> court meters."""
    from src.replay3d.xy_mapper import build_homography_from_corners

    class FakeCourt:
        width_m = court_w
        length_m = court_l
    return build_homography_from_corners(corners_image_xy, FakeCourt())


def warp_point(px: float, py: float, H: np.ndarray) -> tuple[float, float]:
    p = np.asarray([px, py, 1.0], dtype=float)
    q = H @ p
    if abs(q[2]) < 1e-10:
        return (px, py)
    return (float(q[0] / q[2]), float(q[1] / q[2]))


def estimate_keypoint_z(kp_idx: int, bbox_top: float, bbox_bottom: float, frame_height: float = 1080.0) -> float:
    """Estimate Z height for a COCO keypoint using anatomical model + perspective scaling."""
    z_min, z_max = COCO_Z_HEIGHTS[kp_idx]
    # Person pixel height
    person_h_px = max(1.0, bbox_bottom - bbox_top)
    # Keypoint's relative vertical position within the person's bbox (0=top/head, 1=bottom/feet)
    # In image coords: top of bbox = highest Y (smallest value), bottom = lowest Y (largest value)
    # Lower in image = closer to camera = person is farther = smaller apparent scale
    # We normalise so that 0 (top) = tallest (z_max), 1 (bottom) = shortest (z_min)
    # Use ankle average as z_min for all but the lowest keypoints
    avg_z = (z_min + z_max) * 0.5
    # For ankles specifically use z_min
    if kp_idx in (15, 16):
        return z_min
    if kp_idx in (13, 14):  # knees
        return z_min + 0.45
    if kp_idx in (11, 12):  # hips
        return z_min + 0.88
    if kp_idx in (5, 6):  # shoulders
        return z_min + 1.35
    if kp_idx in (7, 8):  # elbows
        return z_min + 1.05
    if kp_idx in (9, 10):  # wrists
        return z_min + 0.90
    if kp_idx <= 4:  # face
        return z_min + 1.65
    return avg_z


def extract_player_keypoints_3d(player_data: dict, H: np.ndarray) -> tuple[list, list, list]:
    """Extract 3D keypoints for a player from homography-warped COCO-17 keypoints.
    
    Returns:
        keypoints_3d: list of (x, y, z) or None for each of 17 keypoints
        center_xyz: (x, y, z) of person center in court space
        visible_mask: list of bool for each keypoint
    """
    pose2d = player_data.get("pose2d", [])
    bbox = player_data.get("bbox_xyxy", None)
    xyz = player_data.get("xyz", None)

    center_xyz = xyz if xyz else [3.05, 6.7, 0.0]

    keypoints_3d = []
    visible_mask = []

    # Use ankle midpoint for scale calibration if available
    ankle_l_idx, ankle_r_idx = 15, 16
    has_ankles = (
        len(pose2d) > max(ankle_l_idx, ankle_r_idx)
    )

    for i in range(17):
        if i < len(pose2d):
            kp = pose2d[i]
            px, py, conf = float(kp[0]), float(kp[1]), float(kp[2]) if len(kp) > 2 else 1.0
            visible = conf > 0.25

            if visible:
                wx, wy = warp_point(px, py, H)
                # Estimate Z
                if bbox and len(bbox) == 4:
                    z = estimate_keypoint_z(i, bbox[1], bbox[3])
                else:
                    z = COCO_Z_HEIGHTS[i][0]
                keypoints_3d.append([wx, wy, z])
                visible_mask.append(True)
            else:
                keypoints_3d.append(None)
                visible_mask.append(False)
        else:
            keypoints_3d.append(None)
            visible_mask.append(False)

    return keypoints_3d, center_xyz, visible_mask


def load_replay_jsonl(path: Path) -> list[dict]:
    frames = []
    with path.open(encoding="utf-8") as f:
        for idx, line in enumerate(f, 1):
            line = line.strip()
            if not line:
                continue
            try:
                frames.append(json.loads(line))
            except json.JSONDecodeError as e:
                raise ValueError(f"Invalid JSONL at line {idx}: {e}") from e
    if not frames:
        raise ValueError("No frames found")
    return frames


def draw_court_3d(ax, width_m: float, length_m: float, net_h: float) -> None:
    corners = [
        (0.0, 0.0, 0.0), (width_m, 0.0, 0.0),
        (width_m, length_m, 0.0), (0.0, length_m, 0.0), (0.0, 0.0, 0.0),
    ]
    xs, ys, zs = zip(*corners)
    ax.plot(xs, ys, zs, color="#444444", linewidth=1.8)

    net_y = length_m * 0.5
    ax.plot([0.0, width_m], [net_y, net_y], [net_h, net_h], linestyle="--", linewidth=1.3, color="#666666")
    ax.plot([0.0, 0.0], [net_y, net_y], [0.0, net_h], color="#777777", linewidth=1.1)
    ax.plot([width_m, width_m], [net_y, net_y], [0.0, net_h], color="#777777", linewidth=1.1)


def setup_axes(ax, width_m: float, length_m: float, z_max: float, elev: float, azim: float) -> None:
    ax.set_xlim(-0.5, width_m + 0.5)
    ax.set_ylim(-0.5, length_m + 0.5)
    ax.set_zlim(0.0, z_max)
    ax.set_xlabel("X (m)")
    ax.set_ylabel("Y (m)")
    ax.set_zlabel("Z (m)")
    ax.view_init(elev=elev, azim=azim)


def plot_limbs(ax, keypoints_3d: list, visible_mask: list, color: str, alpha: float = 0.9, linewidth: float = 2.0) -> None:
    """Draw skeleton limbs for one player."""
    pts = keypoints_3d
    for (i, j) in COCO_LIMBS:
        if (i < len(pts) and j < len(pts) and
                pts[i] is not None and pts[j] is not None and
                visible_mask[i] and visible_mask[j]):
            x = [pts[i][0], pts[j][0]]
            y = [pts[i][1], pts[j][1]]
            z = [pts[i][2], pts[j][2]]
            ax.plot(x, y, z, color=color, alpha=alpha, linewidth=linewidth)


def plot_keypoints(ax, keypoints_3d: list, visible_mask: list, color: str, markersize: float = 8.0) -> None:
    """Plot keypoint dots for one player."""
    xs, ys, zs = [], [], []
    for i, (pt, vis) in enumerate(zip(keypoints_3d, visible_mask)):
        if pt is not None and vis:
            xs.append(pt[0])
            ys.append(pt[1])
            zs.append(pt[2])
    if xs:
        ax.scatter(xs, ys, zs, s=markersize, color=color, alpha=0.8)


def render_png_summary(frames: list[dict], H: np.ndarray, out_path: Path,
                       elev: float, azim: float, z_max: float) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    court = frames[0].get("court", {})
    court_w = float(court.get("width_m", 6.1))
    court_l = float(court.get("length_m", 13.4))
    net_h = float(court.get("net_height_center_m", 1.524))

    # Take last N frames with valid skeleton for summary
    valid_frames = [f for f in frames if f.get("player1", {}).get("pose2d") or f.get("player2", {}).get("pose2d")]
    if not valid_frames:
        valid_frames = frames[-10:]

    last = valid_frames[-1]

    fig = plt.figure(figsize=(12, 8), dpi=150)
    ax = fig.add_subplot(111, projection="3d")
    setup_axes(ax, court_w, court_l, z_max, elev, azim)
    draw_court_3d(ax, court_w, court_l, net_h)

    for pid, pkey in [("P1", "player1"), ("P2", "player2")]:
        pdata = last.get(pkey, {})
        if pdata.get("pose2d"):
            color = "#ff7f0e" if pid == "P1" else "#2ca02c"
            kp3d, _, vis = extract_player_keypoints_3d(pdata, H)
            plot_limbs(ax, kp3d, vis, color=color)
            plot_keypoints(ax, kp3d, vis, color=color)

    # Plot trajectory trail from all valid frames
    for pid, pkey, color in [("P1", "player1", "#ff7f0e"), ("P2", "player2", "#2ca02c")]:
        xs, ys, zs = [], [], []
        for f in valid_frames[-30:]:
            xyz = f.get(pkey, {}).get("xyz", [])
            if xyz and len(xyz) >= 3:
                xs.append(xyz[0])
                ys.append(xyz[1])
                zs.append(xyz[2])
        if xs:
            ax.plot(xs, ys, zs, color=color, alpha=0.35, linewidth=1.2)

    ax.set_title(f"3D Skeleton Replay | frame={last.get('frame','?')} | {len(valid_frames)} frames")
    ax.legend(loc="upper left")
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    fig.savefig(out_path)
    plt.close(fig)


def render_gif(frames: list[dict], H: np.ndarray, out_path: Path,
                gif_step: int, elev: float, azim: float, z_max: float, trail: int) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.animation import FuncAnimation, PillowWriter

    court = frames[0].get("court", {})
    court_w = float(court.get("width_m", 6.1))
    court_l = float(court.get("length_m", 13.4))
    net_h = float(court.get("net_height_center_m", 1.524))

    sampled = frames[::max(1, gif_step)]

    fig = plt.figure(figsize=(12, 8), dpi=110)
    ax = fig.add_subplot(111, projection="3d")
    setup_axes(ax, court_w, court_l, z_max, elev, azim)
    draw_court_3d(ax, court_w, court_l, net_h)

    # Trail lines for center positions
    p1_trail, = ax.plot([], [], [], linewidth=1.5, color="#ff7f0e", alpha=0.5, label="P1 trail")
    p2_trail, = ax.plot([], [], [], linewidth=1.5, color="#2ca02c", alpha=0.5, label="P2 trail")

    # Skeleton lines (will be redrawn each frame)
    p1_limb, = ax.plot([], [], [], color="#ff7f0e", linewidth=2.2, alpha=0.95)
    p2_limb, = ax.plot([], [], [], color="#2ca02c", linewidth=2.2, alpha=0.95)

    # Current keypoint dots
    p1_kp = ax.scatter([], [], [], s=14, color="#ff7f0e", alpha=0.9)
    p2_kp = ax.scatter([], [], [], s=14, color="#2ca02c", alpha=0.9)

    # Current center dots
    p1_dot = ax.scatter([], [], [], s=40, color="#ff7f0e", marker="o", label="P1 center")
    p2_dot = ax.scatter([], [], [], s=40, color="#2ca02c", marker="o", label="P2 center")

    ax.legend(loc="upper left", fontsize=8)

    def set_scatter3d(scatter, coords: list | None) -> None:
        if coords is None:
            scatter._offsets3d = (np.array([]), np.array([]), np.array([]))
        else:
            scatter._offsets3d = (np.array([coords[0]]), np.array([coords[1]]), np.array([coords[2]]))

    def keypoints_to_segments(kp3d: list, vis: list) -> tuple:
        xall, yall, zall = [], [], []
        for (i, j) in COCO_LIMBS:
            if (i < len(kp3d) and j < len(kp3d) and
                    kp3d[i] is not None and kp3d[j] is not None and
                    vis[i] and vis[j]):
                xall.extend([kp3d[i][0], kp3d[j][0], np.nan])
                yall.extend([kp3d[i][1], kp3d[j][1], np.nan])
                zall.extend([kp3d[i][2], kp3d[j][2], np.nan])
        return xall, yall, zall

    def update(frame_idx: int) -> list:
        i = frame_idx
        fr = sampled[i]

        # Update trail
        start = max(0, i - trail)
        trail_frames = sampled[start:i+1]

        p1_trail_x = [f.get("player1", {}).get("xyz", [None]*3)[0] for f in trail_frames]
        p1_trail_y = [f.get("player1", {}).get("xyz", [None]*3)[1] for f in trail_frames]
        p1_trail_z = [f.get("player1", {}).get("xyz", [None]*3)[2] for f in trail_frames]
        p2_trail_x = [f.get("player2", {}).get("xyz", [None]*3)[0] for f in trail_frames]
        p2_trail_y = [f.get("player2", {}).get("xyz", [None]*3)[1] for f in trail_frames]
        p2_trail_z = [f.get("player2", {}).get("xyz", [None]*3)[2] for f in trail_frames]

        p1_trail.set_data_3d(
            [x for x in p1_trail_x if x is not None],
            [y for y in p1_trail_y if y is not None],
            [z for z in p1_trail_z if z is not None],
        )
        p2_trail.set_data_3d(
            [x for x in p2_trail_x if x is not None],
            [y for y in p2_trail_y if y is not None],
            [z for z in p2_trail_z if z is not None],
        )

        # Skeleton for P1 and P2
        for pkey, limb_line, kp_scatter, dot, color in [
            ("player1", p1_limb, p1_kp, p1_dot, "#ff7f0e"),
            ("player2", p2_limb, p2_kp, p2_dot, "#2ca02c"),
        ]:
            pdata = fr.get(pkey, {})
            if pdata.get("pose2d"):
                kp3d, center_xyz, vis = extract_player_keypoints_3d(pdata, H)
                xall, yall, zall = keypoints_to_segments(kp3d, vis)
                if xall:
                    limb_line.set_data_3d(xall, yall, zall)
                    limb_line.set_color(color)
                kpxs = [p[0] for p in kp3d if p is not None]
                kpys = [p[1] for p in kp3d if p is not None]
                kpzs = [p[2] for p in kp3d if p is not None]
                if kpxs:
                    kp_scatter._offsets3d = (np.array(kpxs), np.array(kpys), np.array(kpzs))
                    kp_scatter.set_color(color)
                if center_xyz:
                    set_scatter3d(dot, center_xyz)
            else:
                limb_line.set_data_3d([], [], [])
                kp_scatter._offsets3d = (np.array([]), np.array([]), np.array([]))
                set_scatter3d(dot, None)

        title = f"3D Skeleton Replay | frame={fr.get('frame','?')} t={fr.get('t_sec',0):.2f}s"
        ax.set_title(title, fontsize=9)

        return [p1_trail, p2_trail, p1_limb, p2_limb, p1_kp, p2_kp, p1_dot, p2_dot]

    anim = FuncAnimation(fig, update, frames=len(sampled), interval=60, blit=False)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    anim.save(out_path, writer=PillowWriter(fps=18))
    plt.close(fig)


def main() -> None:
    args = parse_args()
    in_path = Path(args.input)
    out_dir = Path(args.out_dir)

    print(f"Loading corners from: {args.corners_json}")
    corners = load_corners(args.corners_json)
    H = build_homography(corners)
    print(f"Built homography: image px -> court meters")

    frames = load_replay_jsonl(in_path)
    print(f"Loaded {len(frames)} frames from {in_path}")

    png_path = out_dir / "skeleton_3d_summary.png"
    render_png_summary(frames, H, png_path, elev=args.elev, azim=args.azim, z_max=max(1.0, args.z_max))
    print(f"EXPORTED skeleton PNG -> {png_path}")

    if not args.no_gif:
        gif_path = out_dir / "skeleton_3d_replay.gif"
        render_gif(
            frames, H, gif_path,
            gif_step=max(1, args.gif_step),
            elev=args.elev,
            azim=args.azim,
            z_max=max(1.0, args.z_max),
            trail=max(1, args.trail),
        )
        print(f"EXPORTED skeleton GIF -> {gif_path}")


if __name__ == "__main__":
    main()
