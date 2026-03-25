#!/usr/bin/env python3
"""
Combine endpoint + angular features for better model performance.
Priority #4 continuation: retrain with combined feature set.
"""
import pandas as pd
import json
from pathlib import Path
from datetime import datetime
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.model_selection import cross_val_score
import gzip

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "data"
REPORTS = ROOT / "reports"
REPORTS.mkdir(parents=True, exist_ok=True)


def main():
    # Load data
    labels = pd.read_csv(DATA / "rally_labels_v9.csv.gz")
    quant = pd.read_csv(DATA / "quant_features_v11.csv.gz")

    print(f"Labels: {len(labels)} rallies")
    print(f"Quant features: {len(quant)} frames")

    # Build rally-level features (endpoint + angular combined)
    features = []
    for idx, row in labels.iterrows():
        start_frame = int(row["start_frame"])
        end_frame = int(row["end_frame"])
        rally_frames = quant[(quant["frame"] >= start_frame) & (quant["frame"] <= end_frame)]

        feat = {
            "rally_id": idx,
            # Endpoint features (from train_v13)
            "shuttle_x_end": rally_frames["shuttle_x"].iloc[-1] if len(rally_frames) > 0 else 0.5,
            "shuttle_y_end": rally_frames["shuttle_y"].iloc[-1] if len(rally_frames) > 0 else 0.5,
            "dist_from_center": ((rally_frames["shuttle_x"].iloc[-1] - 0.5) ** 2 + 
                                 (rally_frames["shuttle_y"].iloc[-1] - 0.5) ** 2) ** 0.5 if len(rally_frames) > 0 else 0,
            # Angular features (from train_v11/12)
            "X_arms_ang_vel_mean": rally_frames["X_arms_ang_vel"].mean() if len(rally_frames) > 0 else 0,
            "X_arms_ang_vel_max": rally_frames["X_arms_ang_vel"].max() if len(rally_frames) > 0 else 0,
            "X_torso_ang_vel_mean": rally_frames["X_torso_ang_vel"].mean() if len(rally_frames) > 0 else 0,
            "X_torso_ang_vel_max": rally_frames["X_torso_ang_vel"].max() if len(rally_frames) > 0 else 0,
            "X_legs_ang_vel_max": rally_frames["X_legs_ang_vel"].max() if len(rally_frames) > 0 else 0,
            "Y_arms_ang_vel_mean": rally_frames["Y_arms_ang_vel"].mean() if len(rally_frames) > 0 else 0,
            "Y_arms_ang_vel_max": rally_frames["Y_arms_ang_vel"].max() if len(rally_frames) > 0 else 0,
            "Y_torso_ang_vel_mean": rally_frames["Y_torso_ang_vel"].mean() if len(rally_frames) > 0 else 0,
            "Y_torso_ang_vel_max": rally_frames["Y_torso_ang_vel"].max() if len(rally_frames) > 0 else 0,
            # Label
            "winner": row["winner"],
        }
        features.append(feat)

    df = pd.DataFrame(features)
    print(f"Feature matrix: {df.shape}")

    # Prepare data
    feature_cols = [c for c in df.columns if c not in ["rally_id", "winner"]]
    X = df[feature_cols].values
    y = df["winner"].values

    print(f"Class distribution: {dict(zip(*pd.Series(y).value_counts().items()))}")

    # Cross-validation
    model = GradientBoostingClassifier(n_estimators=50, max_depth=3, random_state=42)
    cv_scores = cross_val_score(model, X, y, cv=5, scoring="balanced_accuracy")
    print(f"CV balanced accuracy: {cv_scores.mean():.3f} +/- {cv_scores.std():.3f}")

    # Temporal holdout (last 8 samples for temporal test)
    X_train, X_test = X[:-8], X[-8:]
    y_train, y_test = y[:-8], y[-8:]

    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)

    from sklearn.metrics import accuracy_score, balanced_accuracy_score, f1_score, matthews_corrcoef

    test_acc = accuracy_score(y_test, y_pred)
    test_bal_acc = balanced_accuracy_score(y_test, y_pred)
    test_f1 = f1_score(y_test, y_pred, average="macro")
    test_mcc = matthews_corrcoef(y_test, y_pred)

    print(f"Temporal test: acc={test_acc:.3f}, balanced_acc={test_bal_acc:.3f}, f1={test_f1:.3f}, mcc={test_mcc:.3f}")

    # Feature importance
    importance = dict(zip(feature_cols, model.feature_importances_))
    top_features = dict(sorted(importance.items(), key=lambda x: -x[1])[:8])

    # Save report
    report = {
        "timestamp_local": datetime.now().astimezone().isoformat(timespec="seconds"),
        "step_executed": "Combine endpoint + angular features (priority #4 continuation)",
        "samples": len(df),
        "train": len(X_train),
        "test": len(X_test),
        "features": len(feature_cols),
        "feature_names": feature_cols,
        "class_distribution": {"winner_1": int((y == 1).sum()), "winner_0": int((y == 0).sum())},
        "model": "GB",
        "cv_balanced_accuracy": round(cv_scores.mean(), 3),
        "cv_std": round(cv_scores.std(), 3),
        "temporal_test": {
            "accuracy": round(test_acc, 3),
            "balanced_accuracy": round(test_bal_acc, 3),
            "f1_macro": round(test_f1, 3),
            "mcc": round(test_mcc, 3),
        },
        "top_features": {k: round(v, 3) for k, v in top_features.items()},
        "next_step": "Try adding motion direction features or improve rally segmentation",
    }

    out_file = REPORTS / f"train_v14_{datetime.now().strftime('%Y-%m-%dT%H-%M-%S')}.json"
    out_file.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(f"Saved: {out_file}")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()