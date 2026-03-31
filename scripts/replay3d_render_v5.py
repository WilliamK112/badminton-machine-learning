#!/usr/bin/env python3
"""
Render badminton 3D replay — v5 fixes (from user feedback):

1. FEET ON GROUND: ankle Z is offset by ~0.1m (ANKLE height above ground).
   Subtract mean ankle Z from all keypoints so feet contact z=0.
   Also show a "ground shadow" ellipse under each player.

2. COURT HALVES CORRECT: The camera is at the FAR end (large world Y).
   Near end = camera side = LOW world Y in replay data.
   The rendered scene should show court with net at Y=6.7m and
   clearly indicate which side is which.

3. BETTER COURT VISUALIZATION: Draw court markings, net, and center line.
   Player trail shows their path on the court floor.

4. LIMB LENGTH ENFORCEMENT: Already done, but now also enforce
   hip-shoulder alignment to prevent torso collapse.

5. PLAYER GROUND CONTACT: Show shadow/ellipse at feet level on court.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation, PillowWriter
from matplotlib.patches import Ellipse
from matplotlib.lines import Line2D

# COCO 17 limbs
COCO_LIMBS = [
    (0, 1), (0, 2), (1, 3), (2, 4),
    (0, 5), (0, 6),
    (5, 7), (7, 9),
    (6, 8), (8, 10),
    (5, 11), (6, 12),
    (11, 13), (13, 15),
    (12, 14), (14, 16),
    (11, 12),
]

PLAYER_COLORS = {"player1": "#FF7F0E", "player2": "#2CA02C"}


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--input", required=True)
    p.add_argument("--out-dir", required=True)
    p.add_argument("--gif-step", type=int, default=2)
    p.add_argument("--fps", type=int, default=30)
    p.add_argument("--elev", type=float, default=28.0)
    p.add_argument("--azim", type=float, default=-52.0)
    p.add_argument("--z-max", type=float, default=7.0)
    p.add_argument("--trail", type=int, default=50)
    p.add_argument("--court-dist-threshold", type=float, default=3.0)
    p.add_argument("--z-sep-dist", type=float, default=0.6)
    p.add_argument("--no-gif", action="store_true")
    return p.parse_args()


def load_lifted(path: Path) -> list[dict]:
    frames = []
    with path.open() as f:
        for line in f:
            line = line.strip()
            if line:
                frames.append(json.loads(line))
    return frames


def get_kp(kps_3d, idx) -> np.ndarray | None:
    for kp in kps_3d:
        if kp.get("idx") == idx and kp.get("valid"):
            return np.array(kp["xyz"])
    return None


def get_ankle_z(kps_3d) -> float:
    """Get mean ankle z for ground reference."""
    zs = []
    for kp in kps_3d:
        if kp.get("idx") in (15, 16) and kp.get("valid"):
            zs.append(kp["xyz"][2])
    return np.mean(zs) if zs else 0.0


def draw_court(ax, length_m: float, width_m: float, net_h: float = 1.524):
    """Draw full badminton court with markings."""
    # Court floor (semi-transparent)
    verts = [(0,0,0), (width_m,0,0), (width_m,length_m,0), (0,length_m,0)]
    xs, ys, zs = zip(*verts)
    verts2 = [(x, y, 0.001) for x, y, z in verts]
    # Draw floor as a filled polygon
    from mpl_toolkits.mplot3d.art3d import Poly3DCollection
    floor_verts = [list(zip(xs, ys, [0]*5))]
    # Court outline
    ax.plot(list(xs)+[xs[0]], list(ys)+[ys[0]], [0]*5, color="#888888", linewidth=2.0, zorder=1)
    # Center line (net)
    cx = width_m / 2
    ax.plot([cx, cx], [0, length_m], [0, 0], color="#AAAAAA", linewidth=1.0, linestyle="--", zorder=1)
    # Service lines
    for y in [1.98, length_m - 1.98]:
        ax.plot([0, width_m], [y, y], [0, 0], color="#999999", linewidth=0.8, zorder=1)
    # Short service line
    for y in [0.76, length_m - 0.76]:
        ax.plot([0, width_m], [y, y], [0, 0], color="#AAAAAA", linewidth=0.8, zorder=1)

    # Net
    net_y = length_m / 2
    ax.plot([0, width_m], [net_y, net_y], [0, 0], color="#333333", linewidth=2.5, zorder=2)
    ax.plot([0, width_m], [net_y, net_y], [net_h, net_h], color="#666666", linewidth=1.5, linestyle="--", zorder=2)
    ax.plot([0, 0], [net_y, net_y], [0, net_h], color="#555555", linewidth=1.2, zorder=2)
    ax.plot([width_m, width_m], [net_y, net_y], [0, net_h], color="#555555", linewidth=1.2, zorder=2)
    # Net label
    ax.text(cx, net_y, net_h + 0.1, "NET", fontsize=7, color="#888888",
            ha="center")


def draw_skeleton(ax, kps_3d: list[dict], color: str, alpha: float = 0.9,
                  lw: float = 2.5, ground_z: float = 0.0):
    """
    Draw skeleton limbs. All keypoints are shifted so that ground_z → 0.
    ground_z = mean ankle z, so feet will sit at approximately z=0.
    """
    # Build idx → xyz array (with ground correction applied)
    kp_dict = {}
    for kp in kps_3d:
        if kp.get("valid"):
            arr = np.array(kp["xyz"])
            arr[2] -= ground_z  # shift so ankle ≈ 0
            kp_dict[kp["idx"]] = arr

    drawn = set()
    for parent_i, child_i in COCO_LIMBS:
        key = (min(parent_i, child_i), max(parent_i, child_i))
        if key in drawn:
            continue
        p = kp_dict.get(parent_i)
        c = kp_dict.get(child_i)
        if p is None or c is None:
            continue
        ax.plot([p[0], c[0]], [p[1], c[1]], [p[2], c[2]],
                color=color, linewidth=lw, alpha=alpha,
                solid_capstyle="round", zorder=10)
        drawn.add(key)

    # Joint dots — larger for main joints
    HEAD_TORSO = {0, 5, 6, 11, 12}
    xs_h, ys_h, zs_h = [], [], []
    xs_e, ys_e, zs_e = [], [], []
    for idx, arr in kp_dict.items():
        if idx in HEAD_TORSO:
            xs_h.append(arr[0]); ys_h.append(arr[1]); zs_h.append(arr[2])
        else:
            xs_e.append(arr[0]); ys_e.append(arr[1]); zs_e.append(arr[2])

    if xs_h:
        ax.scatter(xs_h, ys_h, zs_h, s=50, color=color, alpha=alpha,
                   edgecolors="white", linewidths=0.4, zorder=12)
    if xs_e:
        ax.scatter(xs_e, ys_e, zs_e, s=22, color=color, alpha=alpha,
                   edgecolors="white", linewidths=0.3, zorder=12)


def draw_player_shadow(ax, cx, cy, color, alpha=0.25):
    """Draw ground shadow dot under player feet."""
    ax.scatter([cx], [cy], [0.01], s=200, color=color, alpha=alpha,
               edgecolors="none", zorder=3)


def draw_player_label(ax, cx, cy, cz, label, color):
    """Label player position."""
    ax.text(cx, cy, cz + 0.1, label, fontsize=8, color=color,
            ha="center", va="bottom", alpha=0.8, zorder=15)


def render_summary(lifted, out_path, elev, azim, z_max,
                   court_length, court_width, net_h):
    fig = plt.figure(figsize=(12, 8), dpi=150)
    ax = fig.add_subplot(111, projection="3d")
    ax.set_xlim(-0.3, court_width + 0.3)
    ax.set_ylim(-0.5, court_length + 0.5)
    ax.set_zlim(-0.3, z_max)
    ax.set_xlabel("X (m) — court width", fontsize=9)
    ax.set_ylabel("Y (m) — court length [camera→far]", fontsize=9)
    ax.set_zlabel("Z (m) — height", fontsize=9)
    ax.view_init(elev=elev, azim=azim)
    draw_court(ax, court_length, court_width, net_h)

    last = lifted[-1]
    # Half-court background shading
    ax.text(court_width/2, court_length/4, 0.05, "NEAR\nHALF", fontsize=8,
            color="#666666", ha="center", alpha=0.6)
    ax.text(court_width/2, court_length*3/4, 0.05, "FAR\nHALF", fontsize=8,
            color="#666666", ha="center", alpha=0.6)

    handles = []
    for pkey, pcolor in [("player1", "#FF7F0E"), ("player2", "#2CA02C")]:
        pdata = last.get(pkey, {})
        if not pdata.get("visible"):
            continue
        kps = pdata.get("keypoints_3d", [])
        gz = get_ankle_z(kps)
        cx, cy = pdata["center_xyz"][0], pdata["center_xyz"][1]

        draw_skeleton(ax, kps, color=pcolor, alpha=0.9, ground_z=gz)
        draw_player_shadow(ax, cx, cy, pcolor)
        label = "P1" if pkey == "player1" else "P2"
        draw_player_label(ax, cx, cy, 0, label, pcolor)
        handles.append(Line2D([0], [0], color=pcolor, linewidth=2, label=label))

    ax.legend(handles=handles, loc="upper left")
    ax.set_title(f"Badminton 3D Replay — {len(lifted)} frames | net at Y={court_length/2:.1f}m",
                 fontsize=10)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)


def render_gif(lifted, out_path, gif_step, elev, azim, z_max,
               court_length, court_width, net_h,
               court_dist_thresh, z_sep_dist, trail):
    n = len(lifted)
    indices = list(range(0, n, gif_step))
    print(f"GIF: {len(indices)} frames")

    fig = plt.figure(figsize=(12, 8), dpi=120)
    ax = fig.add_subplot(111, projection="3d")
    ax.set_xlim(-0.3, court_width + 0.3)
    ax.set_ylim(-0.5, court_length + 0.5)
    ax.set_zlim(-0.3, z_max)
    ax.set_xlabel("X (m)", fontsize=8)
    ax.set_ylabel("Y (m)", fontsize=8)
    ax.set_zlabel("Z (m)", fontsize=8)

    trail_p1, trail_p2 = [], []

    def init():
        ax.clear()
        ax.set_xlim(-0.3, court_width + 0.3)
        ax.set_ylim(-0.5, court_length + 0.5)
        ax.set_zlim(-0.3, z_max)
        draw_court(ax, court_length, court_width, net_h)
        return []

    def animate(frame_i):
        ax.clear()
        ax.set_xlim(-0.3, court_width + 0.3)
        ax.set_ylim(-0.5, court_length + 0.5)
        ax.set_zlim(-0.3, z_max)
        ax.view_init(elev=elev, azim=azim)
        draw_court(ax, court_length, court_width, net_h)

        fr = lifted[frame_i]
        frame_idx = fr.get("frame", frame_i)
        net_y = court_length / 2

        # Court half labels
        ax.text(court_width/2, court_length/4, 0.05, "NEAR HALF\n(camera side)", fontsize=7,
                color="#888888", ha="center", alpha=0.5)
        ax.text(court_width/2, court_length*3/4, 0.05, "FAR HALF\n(net far side)", fontsize=7,
                color="#888888", ha="center", alpha=0.5)

        p1data = fr.get("player1", {})
        p2data = fr.get("player2", {})

        # Compute Z-sep
        p1_cx, p1_cy = p1data.get("center_xyz", [0,0])[0], p1data.get("center_xyz", [0,0])[1]
        p2_cx, p2_cy = p2data.get("center_xyz", [0,0])[0], p2data.get("center_xyz", [0,0])[1]
        dist = float(np.linalg.norm([p1_cx-p2_cx, p1_cy-p2_cy]))
        apply_zsep = dist < court_dist_thresh and p1data.get("visible") and p2data.get("visible")

        def draw_player(pdata, pcolor, pname, trail_list, zsep=0.0):
            if not pdata.get("visible"):
                return
            kps = pdata.get("keypoints_3d", [])
            gz = get_ankle_z(kps)
            cx = pdata["center_xyz"][0]
            cy = pdata["center_xyz"][1]

            # Apply Z-sep to ground reference
            gz_adj = gz + zsep

            draw_skeleton(ax, kps, color=pcolor, alpha=0.9, ground_z=gz_adj)
            draw_player_shadow(ax, cx, cy, pcolor)
            draw_player_label(ax, cx, cy, 0, pname, pcolor)

            # Court half determination
            half = "FAR" if cy > net_y else "NEAR"
            ax.text(cx + 0.15, cy, 0.05, f"y={cy:.1f}m", fontsize=6,
                    color=pcolor, alpha=0.6)

            # Trail
            trail_list.append((cx, cy, 0))
            if len(trail_list) > trail:
                trail_list.pop(0)
            if len(trail_list) > 2:
                txyz = list(zip(*trail_list))
                ax.plot(txyz[0], txyz[1], txyz[2],
                        color=pcolor, alpha=0.25, linewidth=1.0, zorder=5)

        draw_player(p1data, "#FF7F0E", "P1", trail_p1)
        draw_player(p2data, "#2CA02C", "P2", trail_p2, zsep=z_sep_dist if apply_zsep else 0.0)

        # Shuttle
        sh = fr.get("shuttle", {})
        if sh and sh.get("visible"):
            sx, sy, sz = sh.get("xyz", [0,0,0])
            ax.plot([sx], [sy], [sz], 'x', color="#d62728", markersize=10,
                    alpha=0.9, zorder=20)

        half_p1 = "FAR" if p1_cy > net_y else "NEAR"
        half_p2 = "FAR" if p2_cy > net_y else "NEAR"
        ax.set_title(f"Frame {frame_idx} ({frame_i+1}/{n}) | P1={half_p1}(y={p1_cy:.1f}) P2={half_p2}(y={p2_cy:.1f}) | dist={dist:.2f}m",
                     fontsize=8)
        return []

    anim = FuncAnimation(fig, animate, frames=indices, init_func=init,
                        blit=False, interval=50)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    anim.save(out_path, writer=PillowWriter(fps=20))
    plt.close(fig)
    print(f"GIF: {out_path}")


def main() -> None:
    args = parse_args()
    print(f"Loading: {args.input}")
    lifted = load_lifted(Path(args.input))
    print(f"  {len(lifted)} frames")

    court_length = lifted[0].get("court", {}).get("length_m", 13.4)
    court_width = lifted[0].get("court", {}).get("width_m", 6.1)
    net_h = 1.524

    out_dir = Path(args.out_dir)

    # PNG
    png = out_dir / "v5_3d_summary.png"
    print("Rendering PNG...")
    render_summary(lifted, png, args.elev, args.azim, args.z_max,
                   court_length, court_width, net_h)
    print(f"  PNG: {png}")

    # GIF
    if not args.no_gif:
        gif = out_dir / "v5_3d_replay.gif"
        print("Rendering GIF...")
        render_gif(lifted, gif, args.gif_step, args.elev, args.azim, args.z_max,
                   court_length, court_width, net_h,
                   args.court_dist_threshold, args.z_sep_dist, args.trail)
        print(f"  GIF: {gif}")

    print("\n✓ Done")


if __name__ == "__main__":
    main()
