#!/usr/bin/env python3
"""
Train v45 - Rally detection with v10 labels
Uses shuttle motion + player features to detect rally vs non-rally frames.
"""
import gzip
import json
import numpy as np
import pandas as pd
from pathlib import Path
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.model_selection import cross_val_score, StratifiedKFold
from sklearn.preprocessing import StandardScaler

ROOT = Path(__file__).resolve().parent
DATA = ROOT / "data"


def load_frame_features():
    """Load frame features from v13."""
    feats = {}
    with open(DATA / "frame_features_v13.jsonl") as f:
        for line in f:
            if not line.strip():
                continue
            r = json.loads(line)
            frame = int(r["frame"])
            
            # Extract features
            sh = r.get("shuttle", {})
            x = sh.get("xy", [None, None])[0] if sh.get("xy") else None
            y = sh.get("xy", [None, None])[1] if sh.get("xy") else None
            vx = sh.get("v", [0, 0])[0]
            vy = sh.get("v", [0, 0])[1]
            speed = sh.get("speed", 0)
            
            # Player features
            players = r.get("players", {}) or {}
            player_features = []
            for p in ["X", "Y"]:
                player = players.get(p) or {}
                center = player.get("center") if player else None
                if center and len(center) >= 2:
                    player_features.extend(center)
                else:
                    player_features.extend([0, 0])
            
            # Court zone
            zone = r.get("court_zone", "unknown")
            zone_map = {"service": 0, "mid": 1, "defense": 2, "attack": 3, "unknown": 4}
            zone_code = zone_map.get(zone, 4)
            
            # Stance
            X_stance = r.get("X_stance_width", 0)
            Y_stance = r.get("Y_stance_width", 0)
            
            # COG
            X_cog = r.get("X_cog", [0, 0])
            Y_cog = r.get("Y_cog", [0, 0])
            
            # Landing prediction
            landing_x = r.get("predicted_landing_x")
            landing_y = r.get("predicted_landing_y")
            
            # Momentum
            momentum = r.get("shuttle_momentum", 0)
            
            feats[frame] = {
                "shuttle_x": x or 0,
                "shuttle_y": y or 0,
                "shuttle_vx": vx,
                "shuttle_vy": vy,
                "shuttle_speed": speed,
                "player_X_x": player_features[0],
                "player_X_y": player_features[1],
                "player_Y_x": player_features[2],
                "player_Y_y": player_features[3],
                "court_zone": zone_code,
                "X_stance": X_stance,
                "Y_stance": Y_stance,
                "X_cog_x": X_cog[0] if isinstance(X_cog, list) else 0,
                "X_cog_y": X_cog[1] if isinstance(X_cog, list) else 0,
                "Y_cog_x": Y_cog[0] if isinstance(Y_cog, list) else 0,
                "Y_cog_y": Y_cog[1] if isinstance(Y_cog, list) else 0,
                "landing_x": landing_x or 0,
                "landing_y": landing_y or 0,
                "momentum": momentum,
            }
    
    return feats


def main():
    # Load features
    print("Loading features...")
    feats = load_frame_features()
    print(f"Loaded {len(feats)} frames")
    
    # Load labels
    print("Loading labels...")
    labels = pd.read_csv(DATA / "rally_labels_v10.csv.gz")
    # Binary: 1 if in rally, 0 otherwise
    labels["label"] = (labels["label"] == 1).astype(int)
    print(f"Labels: {labels.label.value_counts().to_dict()}")
    
    # Match features to labels
    X_rows = []
    y_rows = []
    for _, row in labels.iterrows():
        frame = row["frame"]
        if frame in feats:
            f = feats[frame]
            X_rows.append([
                f["shuttle_x"], f["shuttle_y"], 
                f["shuttle_vx"], f["shuttle_vy"], f["shuttle_speed"],
                f["player_X_x"], f["player_X_y"], f["player_Y_x"], f["player_Y_y"],
                f["court_zone"], f["X_stance"], f["Y_stance"],
                f["X_cog_x"], f["X_cog_y"], f["Y_cog_x"], f["Y_cog_y"],
                f["landing_x"], f["landing_y"], f["momentum"],
            ])
            y_rows.append(row["label"])
    
    X = np.array(X_rows)
    y = np.array(y_rows)
    
    print(f"Training data: {X.shape}, labels: {y.sum()}/{len(y)}")
    
    # Train model
    print("Training models...")
    
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    
    # Random Forest
    rf = RandomForestClassifier(n_estimators=100, max_depth=10, 
                                class_weight='balanced', random_state=42, n_jobs=-1)
    rf_scores = cross_val_score(rf, X, y, cv=cv, scoring='accuracy')
    print(f"RF CV: {rf_scores.mean():.3f} (+/- {rf_scores.std():.3f})")
    
    # Gradient Boosting
    gb = GradientBoostingClassifier(n_estimators=100, max_depth=3, random_state=42)
    gb_scores = cross_val_score(gb, X, y, cv=cv, scoring='accuracy')
    print(f"GB CV: {gb_scores.mean():.3f} (+/- {gb_scores.std():.3f})")
    
    # Train final model
    best_model = rf if rf_scores.mean() >= gb_scores.mean() else gb
    best_name = "RF" if rf_scores.mean() >= gb_scores.mean() else "GB"
    best_cv = max(rf_scores.mean(), gb_scores.mean())
    
    print(f"\nBest: {best_name} CV={best_cv:.3f}")
    
    # Fit on all data
    best_model.fit(X, y)
    
    # Save model
    import pickle
    with open(ROOT / "models" / "v45.pkl", "wb") as f:
        pickle.dump(best_model, f)
    
    # Save features
    feature_names = [
        "shuttle_x", "shuttle_y", "shuttle_vx", "shuttle_vy", "shuttle_speed",
        "player_X_x", "player_X_y", "player_Y_x", "player_Y_y",
        "court_zone", "X_stance", "Y_stance",
        "X_cog_x", "X_cog_y", "Y_cog_x", "Y_cog_y",
        "landing_x", "landing_y", "momentum",
    ]
    with open(ROOT / "models" / "v45_features.json", "w") as f:
        json.dump(feature_names, f)
    
    print(f"Saved v45.pkl (CV={best_cv:.3f})")


if __name__ == "__main__":
    main()
