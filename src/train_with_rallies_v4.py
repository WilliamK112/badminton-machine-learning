#!/usr/bin/env python3
from __future__ import annotations

import csv
import gzip
import json
from datetime import datetime
from pathlib import Path

import numpy as np
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.metrics import accuracy_score, mean_squared_error
from sklearn.model_selection import train_test_split

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "data"
REPORTS = ROOT / "reports"


def open_text_auto(path: Path, mode: str = "rt"):
    if path.suffix == ".gz":
        return gzip.open(path, mode, encoding="utf-8", newline="")
    return path.open(mode, encoding="utf-8", newline="")


def resolve_input(base_name: str) -> Path:
    plain = DATA / base_name
    gz = DATA / f"{base_name}.gz"
    if plain.exists():
        return plain
    if gz.exists():
        return gz
    raise FileNotFoundError(f"missing input: {plain} or {gz}")


def main() -> None:
    REPORTS.mkdir(parents=True, exist_ok=True)

    feat_file = resolve_input("frame_features_v6.jsonl")
    rally_file = resolve_input("rally_labels_v5.csv")
    base_metrics_file = REPORTS / "quant_model_metrics_v4.json"
    out = REPORTS / "rally_metrics_v4_compare.json"

    with open_text_auto(feat_file, "rt") as f:
        rows = [json.loads(x) for x in f if x.strip()]
    by_frame = {int(r["frame"]): r for r in rows if "frame" in r}

    rallies = []
    with open_text_auto(rally_file, "rt") as f:
        rd = csv.DictReader(f)
        for r in rd:
            rallies.append(r)

    X, y_xy, y_w = [], [], []
    for rr in rallies:
        sf = int(rr["start_frame"])
        ef = int(rr["end_frame"])
        winner = int(rr["winner"])
        lx = float(rr["next_landing_x"])
        ly = float(rr["next_landing_y"])

        rec = None
        for fr in range(ef, sf - 1, -1):
            cand = by_frame.get(fr)
            if not cand:
                continue
            sh = cand.get("shuttle", {}) or {}
            x = sh.get("x")
            y = sh.get("y")
            if x is None or y is None:
                xy = sh.get("xy")
                if xy is not None:
                    x, y = xy[0], xy[1]
            if x is not None and y is not None:
                rec = cand
                break
        if rec is None:
            continue

        sh = rec.get("shuttle", {}) or {}
        sx = sh.get("x")
        sy = sh.get("y")
        if sx is None or sy is None:
            xy = sh.get("xy") or [0.5, 0.5]
            sx, sy = xy[0], xy[1]

        pX = (rec.get("players", {}).get("X") or {}).get("center") or [0.25, 0.25]
        pY = (rec.get("players", {}).get("Y") or {}).get("center") or [0.75, 0.75]
        sp = float(sh.get("speed", 0.0) or 0.0)

        feat = [sx, sy, sp, pX[0], pX[1], pY[0], pY[1], (ef - sf)]
        X.append(feat)
        y_xy.append([lx, ly])
        y_w.append(winner)

    X = np.array(X, dtype=float)
    y_xy = np.array(y_xy, dtype=float)
    y_w = np.array(y_w, dtype=int)

    Xtr, Xte, ytr_xy, yte_xy, ytr_w, yte_w = train_test_split(
        X, y_xy, y_w, test_size=0.25, random_state=42
    )

    reg = RandomForestRegressor(n_estimators=260, random_state=42)
    reg.fit(Xtr, ytr_xy)
    pxy = reg.predict(Xte)
    rmse = mean_squared_error(yte_xy, pxy) ** 0.5

    clf = RandomForestClassifier(n_estimators=260, random_state=42)
    clf.fit(Xtr, ytr_w)
    pw = clf.predict(Xte)
    acc = accuracy_score(yte_w, pw)

    base = {}
    if base_metrics_file.exists():
        base = json.loads(base_metrics_file.read_text())

    report = {
        "timestamp_local": datetime.now().astimezone().isoformat(timespec="seconds"),
        "step_executed": "Reduce storage/runtime (priority #6): rally-aware training now supports compressed .jsonl/.csv inputs end-to-end",
        "dataset": str(rally_file.relative_to(ROOT)),
        "feature_source": str(feat_file.relative_to(ROOT)),
        "samples": int(len(X)),
        "winner_acc": float(acc),
        "landing_rmse": float(rmse),
        "compare_vs_quant_v4": {
            "quant_v4_winner_acc": base.get("winner_acc"),
            "quant_v4_landing_rmse": base.get("landing_rmse"),
            "delta_winner_acc": (float(acc) - float(base["winner_acc"])) if base.get("winner_acc") is not None else None,
            "delta_landing_rmse": (float(rmse) - float(base["landing_rmse"])) if base.get("landing_rmse") is not None else None,
        },
        "next_step": "Switch remaining visual/training entrypoints to compressed-first outputs and remove stale uncompressed artifacts safely.",
    }

    out.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(str(out))
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
