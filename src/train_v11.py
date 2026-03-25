#!/usr/bin/env python3
"""
train_v11.py - Combined angular velocity + stance + shuttle direction features
Best of previous runs: angular velocity features only (balanced_acc=0.633)
"""
import gzip
import csv
import json
from pathlib import Path
from datetime import datetime

import numpy as np
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import cross_val_score, StratifiedKFold
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import (
    accuracy_score, balanced_accuracy_score, f1_score, matthews_corrcoef
)

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "data"
REPORTS = ROOT / "reports"

def safe_float(x):
    try:
        return float(x) if x else 0.0
    except:
        return 0.0

def main():
    # Load frame features (for stance, shuttle_dir_change)
    with open(DATA / "frame_features_v10.jsonl") as f:
        frame_rows = [json.loads(line) for line in f if line.strip()]

    # Load quant features (for angular velocity)
    with gzip.open(DATA / "quant_features_v10.csv.gz", "rt") as f:
        quant_reader = csv.DictReader(f)
        quant_rows = list(quant_reader)

    # Load rally labels  
    with gzip.open(DATA / "rally_labels_v9.csv.gz", "rt") as f:
        reader = csv.DictReader(f)
        rally_labels = {int(r["start_frame"]): r for r in reader}

    def get_rally_features(start_f, end_f):
        q_rows = [r for r in quant_rows if start_f <= int(r["frame"]) <= end_f]
        f_rows = [r for r in frame_rows if start_f <= r.get("frame", 0) <= end_f]
        if not q_rows:
            return None
        
        features = {}
        
        # Angular velocity features (most predictive)
        for side in ["X", "Y"]:
            for key_suffix in ["arms_ang_vel", "torso_ang_vel", "legs_ang_vel"]:
                key = f"{side}_{key_suffix}"
                vals = [safe_float(r.get(key, 0)) for r in q_rows if r.get(key)]
                if vals:
                    features[f"{key}_mean"] = np.mean(vals)
                    features[f"{key}_max"] = np.max(vals)
        
        # Shuttle stats
        shuttle_x = [safe_float(r.get("shuttle_x", 0.5)) for r in q_rows]
        shuttle_y = [safe_float(r.get("shuttle_y", 0.5)) for r in q_rows]
        
        if shuttle_x:
            features["shuttle_x_mean"] = np.mean(shuttle_x)
            features["shuttle_y_mean"] = np.mean(shuttle_y)
        
        # Stance width (v10 new feature)
        for side in ["X", "Y"]:
            stance_key = f"{side}_stance_width"
            stance_vals = [r.get(stance_key, 0) for r in f_rows if r.get(stance_key, 0) > 0]
            if stance_vals:
                features[f"{side}_stance_mean"] = np.mean(stance_vals)
        
        return features

    # Build dataset
    X_data = []
    y_data = []
    for start_f, label in rally_labels.items():
        end_f = int(label["end_frame"])
        feats = get_rally_features(start_f, end_f)
        if feats:
            X_data.append(feats)
            y_data.append(int(label["winner"]))

    feature_names = list(X_data[0].keys())
    X = np.array([[x.get(k, 0) for k in feature_names] for x in X_data])
    y = np.array(y_data)

    print(f"Samples: {len(X)}, Features: {len(feature_names)}")
    print(f"Class: {sum(y)} winners=1, {len(y)-sum(y)} winner=0")

    # Scale and train
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    # Use the best model from exploration
    model = GradientBoostingClassifier(n_estimators=30, max_depth=2, random_state=42)
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    scores = cross_val_score(model, X_scaled, y, cv=cv, scoring="balanced_accuracy")

    model.fit(X_scaled, y)
    y_pred = model.predict(X_scaled)

    # Metrics
    acc = accuracy_score(y, y_pred)
    bal_acc = balanced_accuracy_score(y, y_pred)
    f1 = f1_score(y, y_pred, average="macro")
    mcc = matthews_corrcoef(y, y_pred)

    print(f"CV balanced accuracy: {np.mean(scores):.3f} (+/- {np.std(scores):.3f})")
    print(f"Train accuracy: {acc:.3f}, balanced: {bal_acc:.3f}, F1: {f1:.3f}, MCC: {mcc:.3f}")

    # Feature importance
    importances = model.feature_importances_
    sorted_idx = np.argsort(importances)[::-1]
    print("\nTop features:")
    for i in sorted_idx[:8]:
        print(f"  {feature_names[i]}: {importances[i]:.3f}")

    # Save model metrics
    report = {
        "timestamp_local": datetime.now().astimezone().isoformat(timespec="seconds"),
        "step_executed": "train_v11.py: Angular velocity + stance features",
        "samples": len(X),
        "features": len(feature_names),
        "feature_names": feature_names,
        "class_distribution": {"winner_1": int(sum(y)), "winner_0": int(len(y) - sum(y))},
        "cv_balanced_accuracy": round(np.mean(scores), 3),
        "cv_std": round(np.std(scores), 3),
        "train_metrics": {
            "accuracy": round(acc, 3),
            "balanced_accuracy": round(bal_acc, 3),
            "f1_macro": round(f1, 3),
            "mcc": round(mcc, 3)
        },
        "top_features": {feature_names[i]: round(importances[i], 3) for i in sorted_idx[:8]},
        "next_step": "Continue improving feature extraction or add more rallies"
    }

    out_report = REPORTS / f"train_v11_{datetime.now().strftime('%Y-%m-%dT%H-%M-%S')}.json"
    out_report.write_text(json.dumps(report, indent=2))
    print(f"\nReport: {out_report}")

if __name__ == "__main__":
    main()
