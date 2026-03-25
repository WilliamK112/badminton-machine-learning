#!/usr/bin/env python3
"""
train_with_rallies_v6.py - Rally-aware training with auto-detect for .csv/.csv.gz labels
"""
import csv
import gzip
import json
from pathlib import Path
from datetime import datetime
from typing import Dict, List, Tuple, Optional

import numpy as np
from sklearn.base import clone
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, ExtraTreesClassifier, VotingClassifier
from sklearn.ensemble import ExtraTreesRegressor
from sklearn.multioutput import MultiOutputRegressor
from sklearn.model_selection import StratifiedShuffleSplit
from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    f1_score,
    matthews_corrcoef,
    mean_squared_error,
    mean_absolute_error,
    average_precision_score,
    roc_auc_score,
)

ROOT = Path(__file__).resolve().parents[1]
feat_file = ROOT / "data" / "frame_features_v6.jsonl"


def resolve_input(base_name: str) -> Path:
    """Auto-detect .csv or .csv.gz"""
    plain = ROOT / "data" / base_name
    gz = ROOT / "data" / f"{base_name}.gz"
    if plain.exists():
        return plain
    if gz.exists():
        return gz
    raise FileNotFoundError(f"missing input: {plain} or {gz}")


def open_csv_auto(path: Path):
    """Open CSV or gzipped CSV"""
    if path.suffix == ".gz":
        return gzip.open(path, "rt", encoding="utf-8", newline="")
    return path.open("r", encoding="utf-8", newline="")


# Auto-detect rally labels
rally_file = resolve_input("rally_labels_v5.csv")
if not rally_file.exists():
    rally_file = resolve_input("rally_labels_v4.csv")

out_file = ROOT / "reports" / f"rally_metrics_v6_{datetime.now().strftime('%Y-%m-%dT%H-%M-%S')}.json"


def safe_float(x, default=0.0):
    try:
        if x is None:
            return default
        return float(x)
    except Exception:
        return default


def shuttle_xy(rec: Dict) -> Tuple[Optional[float], Optional[float]]:
    sh = rec.get("shuttle", {}) or {}
    x = sh.get("x")
    y = sh.get("y")
    if x is None or y is None:
        xy = sh.get("xy")
        if xy is not None and len(xy) >= 2:
            x, y = xy[0], xy[1]
    if x is None or y is None:
        return None, None
    return float(x), float(y)


def agg_stats(prefix: str, arr: List[float]) -> Dict[str, float]:
    if not arr:
        return {
            f"{prefix}_mean": 0.0,
            f"{prefix}_std": 0.0,
            f"{prefix}_min": 0.0,
            f"{prefix}_max": 0.0,
            f"{prefix}_q25": 0.0,
            f"{prefix}_q75": 0.0,
        }
    a = np.array(arr, dtype=float)
    return {
        f"{prefix}_mean": float(np.mean(a)),
        f"{prefix}_std": float(np.std(a)),
        f"{prefix}_min": float(np.min(a)),
        f"{prefix}_max": float(np.max(a)),
        f"{prefix}_q25": float(np.quantile(a, 0.25)),
        f"{prefix}_q75": float(np.quantile(a, 0.75)),
    }


def build_models(seed: int = 42):
    cls_model = VotingClassifier(
        estimators=[
            (
                "lr",
                Pipeline(
                    [
                        ("scaler", StandardScaler()),
                        (
                            "lr",
                            LogisticRegression(
                                max_iter=3000,
                                class_weight="balanced",
                                random_state=seed,
                            ),
                        ),
                    ]
                ),
            ),
            (
                "rf",
                RandomForestClassifier(
                    n_estimators=500,
                    min_samples_leaf=2,
                    class_weight="balanced_subsample",
                    random_state=seed,
                    n_jobs=-1,
                ),
            ),
            (
                "et",
                ExtraTreesClassifier(
                    n_estimators=700,
                    min_samples_leaf=2,
                    class_weight="balanced",
                    random_state=seed,
                    n_jobs=-1,
                ),
            ),
        ],
        voting="soft",
        weights=[1.0, 1.2, 1.2],
        n_jobs=-1,
    )

    reg_model = MultiOutputRegressor(
        ExtraTreesRegressor(
            n_estimators=900,
            min_samples_leaf=2,
            random_state=seed,
            n_jobs=-1,
        )
    )
    return cls_model, reg_model


def evaluate_fold(
    cls_model,
    reg_model,
    Xtr,
    Xte,
    ytr_w,
    yte_w,
    ytr_xy,
    yte_xy,
):
    cls_model.fit(Xtr, ytr_w)
    pred_w = cls_model.predict(Xte)

    metrics = {
        "acc": float(accuracy_score(yte_w, pred_w)),
        "balanced_acc": float(balanced_accuracy_score(yte_w, pred_w)),
        "f1_macro": float(f1_score(yte_w, pred_w, average="macro", zero_division=0)),
        "mcc": float(matthews_corrcoef(yte_w, pred_w)) if len(np.unique(pred_w)) > 1 else 0.0,
    }

    if len(np.unique(yte_w)) > 1:
        try:
            proba = cls_model.predict_proba(Xte)[:, 1]
            metrics["pr_auc"] = float(average_precision_score(yte_w, proba))
            metrics["roc_auc"] = float(roc_auc_score(yte_w, proba))
        except Exception:
            metrics["pr_auc"] = None
            metrics["roc_auc"] = None
    else:
        metrics["pr_auc"] = None
        metrics["roc_auc"] = None

    majority = int(np.bincount(yte_w).argmax())
    metrics["majority_baseline_acc"] = float(np.mean(yte_w == majority))

    reg_model.fit(Xtr, ytr_xy)
    pred_xy = reg_model.predict(Xte)
    rmse = mean_squared_error(yte_xy, pred_xy) ** 0.5
    mae = mean_absolute_error(yte_xy, pred_xy)
    metrics["landing_rmse"] = float(rmse)
    metrics["landing_mae"] = float(mae)

    pred_side = (pred_xy[:, 1] > 0.5).astype(int)
    metrics["landing_side_acc_from_reg"] = float(accuracy_score(yte_w, pred_side))

    return metrics


def summarize(name: str, vals: List[Optional[float]]):
    clean = [v for v in vals if v is not None]
    if not clean:
        return {
            "name": name,
            "count": 0,
            "mean": None,
            "std": None,
            "p10": None,
            "p50": None,
            "p90": None,
        }
    arr = np.array(clean, dtype=float)
    return {
        "name": name,
        "count": int(len(arr)),
        "mean": float(np.mean(arr)),
        "std": float(np.std(arr)),
        "p10": float(np.quantile(arr, 0.10)),
        "p50": float(np.quantile(arr, 0.50)),
        "p90": float(np.quantile(arr, 0.90)),
    }


# Load frame features
rows = [json.loads(x) for x in feat_file.read_text().splitlines() if x.strip()]
by_frame = {int(r["frame"]): r for r in rows}

# Load rally labels (auto-detect .csv/.csv.gz)
rallies = []
with open_csv_auto(rally_file) as f:
    rd = csv.DictReader(f)
    for r in rd:
        rallies.append(r)

print(f"Loaded {len(rallies)} rallies from {rally_file.name}")

X_rows: List[Dict[str, float]] = []
y_w: List[int] = []
y_xy: List[List[float]] = []
meta = []

for rr in rallies:
    sf = int(rr["start_frame"])
    ef = int(rr["end_frame"])
    winner = int(rr["winner"])
    lx = safe_float(rr.get("next_landing_x") or rr.get("landing_x"))
    ly = safe_float(rr.get("next_landing_y") or rr.get("landing_y"))

    segment = [by_frame[f] for f in range(sf, ef + 1) if f in by_frame]
    if len(segment) < 4:
        continue

    cut = max(3, int(len(segment) * 0.4))
    early = segment[:cut]

    sx, sy, spd, vx, vy, vis, src_det, px_dist = [], [], [], [], [], [], [], []

    prev_xy = None
    for rec in early:
        sh = rec.get("shuttle", {}) or {}
        x, y = shuttle_xy(rec)
        if x is None or y is None:
            continue

        sx.append(x)
        sy.append(y)
        spd.append(safe_float(sh.get("speed"), 0.0))
        vis.append(1.0 if sh.get("visible") else 0.0)
        src_det.append(1.0 if sh.get("source") == "det" else 0.0)

        if prev_xy is None:
            vx.append(0.0)
            vy.append(0.0)
        else:
            vx.append(float(x - prev_xy[0]))
            vy.append(float(y - prev_xy[1]))
        prev_xy = (x, y)

        pX = (rec.get("players", {}).get("X") or {}).get("center") or [0.25, 0.25]
        pY = (rec.get("players", {}).get("Y") or {}).get("center") or [0.75, 0.75]
        dx = safe_float(pX[0]) - safe_float(pY[0])
        dy = safe_float(pX[1]) - safe_float(pY[1])
        px_dist.append(float((dx * dx + dy * dy) ** 0.5))

    if len(sx) < 3:
        continue

    feat = {
        "rally_span": float(ef - sf),
        "obs_frames": float(len(early)),
    }
    feat.update(agg_stats("sx", sx))
    feat.update(agg_stats("sy", sy))
    feat.update(agg_stats("spd", spd))
    feat.update(agg_stats("vx", vx))
    feat.update(agg_stats("vy", vy))
    feat.update(agg_stats("pxdist", px_dist))
    feat.update(agg_stats("vis", vis))
    feat.update(agg_stats("srcdet", src_det))

    X_rows.append(feat)
    y_w.append(winner)
    y_xy.append([lx, ly])
    meta.append({"rally_id": int(rr["rally_id"]), "start_frame": sf, "end_frame": ef})

if not X_rows:
    raise RuntimeError("No training samples built from rally labels and frame features.")

print(f"Built {len(X_rows)} training samples")

feature_names = sorted(X_rows[0].keys())
X = np.array([[r[k] for k in feature_names] for r in X_rows], dtype=float)
y_w_arr = np.array(y_w, dtype=int)
y_xy_arr = np.array(y_xy, dtype=float)

class_counts = {int(k): int(v) for k, v in zip(*np.unique(y_w_arr, return_counts=True))}
minority = min(class_counts.values()) if class_counts else 0

base_cls, base_reg = build_models(seed=42)

split_metrics = []
if minority >= 2:
    n_splits = 40
    sss = StratifiedShuffleSplit(n_splits=n_splits, test_size=0.25, random_state=42)
    for tr_idx, te_idx in sss.split(X, y_w_arr):
        m = evaluate_fold(
            clone(base_cls),
            clone(base_reg),
            X[tr_idx],
            X[te_idx],
            y_w_arr[tr_idx],
            y_w_arr[te_idx],
            y_xy_arr[tr_idx],
            y_xy_arr[te_idx],
        )
        split_metrics.append(m)

# Temporal holdout (no shuffle)
order = np.argsort([m["start_frame"] for m in meta])
Xo = X[order]
ywo = y_w_arr[order]
yxyo = y_xy_arr[order]
cut_idx = int(len(Xo) * 0.75)
Xtr, Xte = Xo[:cut_idx], Xo[cut_idx:]
ytr_w, yte_w = ywo[:cut_idx], ywo[cut_idx:]
ytr_xy, yte_xy = yxyo[:cut_idx], yxyo[cut_idx:]

temporal_metrics = evaluate_fold(
    clone(base_cls),
    clone(base_reg),
    Xtr,
    Xte,
    ytr_w,
    yte_w,
    ytr_xy,
    yte_xy,
)

keys = [
    "acc",
    "balanced_acc",
    "f1_macro",
    "mcc",
    "pr_auc",
    "roc_auc",
    "majority_baseline_acc",
    "landing_rmse",
    "landing_mae",
    "landing_side_acc_from_reg",
]

summary = {k: summarize(k, [m.get(k) for m in split_metrics]) for k in keys}

report = {
    "timestamp_local": datetime.now().astimezone().isoformat(timespec="seconds"),
    "labels_file": str(rally_file.relative_to(ROOT)),
    "samples": int(len(X)),
    "class_counts": class_counts,
    "imbalance_ratio_majority_to_minority": float(max(class_counts.values()) / max(1, minority)),
    "features": {
        "count": int(len(feature_names)),
        "feature_names": feature_names,
        "feature_window": "first 40% frames of each rally (to reduce endpoint leakage)",
    },
    "model": {
        "classifier": "Soft-voting ensemble [LogisticRegression(class_weight=balanced) + RandomForest + ExtraTrees]",
        "regressor": "MultiOutput ExtraTreesRegressor",
    },
    "evaluation": {
        "repeated_stratified_shuffle_split": {
            "enabled": bool(split_metrics),
            "splits": int(len(split_metrics)),
            "test_size": 0.25,
            "metric_summary": summary,
        },
        "temporal_holdout": {
            "train_samples": int(len(Xtr)),
            "test_samples": int(len(Xte)),
            "train_pos_rate": float(np.mean(ytr_w)) if len(ytr_w) else None,
            "test_pos_rate": float(np.mean(yte_w)) if len(yte_w) else None,
            "metrics": temporal_metrics,
        },
    },
    "trust_notes": [
        "Accuracy alone is not reliable under extreme class imbalance.",
        "Balanced accuracy, macro F1 and MCC are reported for credibility.",
        "Current labels are still proxy-generated; true manually labeled winners are needed for strong validity.",
    ],
    "comparison_hint": "Compare balanced_acc/f1_macro/mcc vs majority_baseline_acc; if close, winner model has weak real value.",
    "next_step": "Compare v5 vs v6 metrics; if improved, use v6 as new baseline.",
}

out_file.write_text(json.dumps(report, indent=2), encoding="utf-8")
print("saved", out_file)
print(json.dumps({
    "samples": report["samples"],
    "class_counts": report["class_counts"],
    "split_balanced_acc_mean": report["evaluation"]["repeated_stratified_shuffle_split"]["metric_summary"]["balanced_acc"]["mean"],
    "split_mcc_mean": report["evaluation"]["repeated_stratified_shuffle_split"]["metric_summary"]["mcc"]["mean"],
    "temporal_balanced_acc": report["evaluation"]["temporal_holdout"]["metrics"]["balanced_acc"],
    "temporal_acc": report["evaluation"]["temporal_holdout"]["metrics"]["acc"],
    "temporal_landing_rmse": report["evaluation"]["temporal_holdout"]["metrics"]["landing_rmse"],
}, indent=2))
