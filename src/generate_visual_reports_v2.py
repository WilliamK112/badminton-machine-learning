#!/usr/bin/env python3
from __future__ import annotations

import json
from datetime import datetime
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


def resolve_csv_path(path: Path) -> Path:
    if path.exists():
        return path
    gz = Path(f"{path}.gz")
    if gz.exists():
        return gz
    raise FileNotFoundError(f"Missing input file: {path} (or {gz.name})")


def main() -> None:
    root = Path(__file__).resolve().parents[1]
    data_path = resolve_csv_path(root / "data" / "quant_features_v4.csv")
    reports = root / "reports"
    reports.mkdir(parents=True, exist_ok=True)

    df = pd.read_csv(data_path)
    if df.empty:
        raise RuntimeError("quant_features_v4.csv is empty")

    # 1) Win-prob timeline (smoothed winner proxy as P(X wins next point))
    raw = df["winner_proxy"].astype(float).to_numpy()
    w = 121
    kernel = np.ones(w, dtype=float) / w
    smooth = np.convolve(raw, kernel, mode="same")

    plt.figure(figsize=(10, 4))
    t = df["t_sec"].to_numpy()
    plt.plot(t, smooth, color="#2F80ED", linewidth=2, label="P(X wins next)")
    plt.fill_between(t, smooth, 0.5, where=smooth >= 0.5, color="#2F80ED", alpha=0.15)
    plt.fill_between(t, smooth, 0.5, where=smooth < 0.5, color="#EB5757", alpha=0.12)
    plt.axhline(0.5, color="#666", linestyle="--", linewidth=1)
    plt.ylim(0, 1)
    plt.xlim(t.min(), t.max())
    plt.xlabel("Time (sec)")
    plt.ylabel("Win Probability")
    plt.title("Win-Probability Timeline (v4 features)")
    plt.legend(loc="upper right")
    plt.tight_layout()
    wp_path = reports / "win_prob_timeline_v2.png"
    plt.savefig(wp_path, dpi=160)
    plt.close()

    # 2) Landing heatmap proxy from low-shuttle-speed points
    sx = df["shuttle_x"].clip(0, 1).to_numpy()
    sy = df["shuttle_y"].clip(0, 1).to_numpy()
    speed = df["shuttle_speed"].to_numpy()

    # Use bottom 20% speed as potential landing vicinity
    thr = float(np.nanquantile(speed, 0.2))
    mask = speed <= thr
    lx = sx[mask]
    ly = sy[mask]
    if len(lx) < 20:
        lx, ly = sx, sy

    bins = 80
    h, xedges, yedges = np.histogram2d(lx, ly, bins=bins, range=[[0, 1], [0, 1]])

    plt.figure(figsize=(6, 10))
    plt.imshow(
        h.T,
        origin="lower",
        extent=[0, 1, 0, 1],
        aspect="auto",
        cmap="magma",
    )
    plt.colorbar(label="Landing Density (proxy)")
    plt.xlabel("Court X (normalized)")
    plt.ylabel("Court Y (normalized)")
    plt.title("Landing Heatmap (v4, low-speed proxy)")
    plt.tight_layout()
    hm_path = reports / "landing_heatmap_v2.png"
    plt.savefig(hm_path, dpi=160)
    plt.close()

    out = {
        "timestamp": datetime.now().isoformat(timespec="seconds"),
        "input": str(data_path.relative_to(root)),
        "outputs": [
            str(wp_path.relative_to(root)),
            str(hm_path.relative_to(root)),
        ],
        "frames": int(len(df)),
        "landing_proxy_speed_threshold": thr,
        "landing_proxy_samples": int(len(lx)),
    }
    meta = reports / f"visual_reports_v2_{datetime.now().strftime('%Y-%m-%dT%H-%M-%S')}.json"
    meta.write_text(json.dumps(out, indent=2), encoding="utf-8")
    print(json.dumps(out, indent=2))


if __name__ == "__main__":
    main()
