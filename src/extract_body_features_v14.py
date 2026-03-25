#!/usr/bin/env python3
"""
Add detailed body part tracking features v14
Tracks: arms, torso, legs angles + joint distances
Priority #1: Improve feature extraction quality
"""
import json
import numpy as np
import pandas as pd
from pathlib import Path
from collections import defaultdict
from datetime import datetime

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "data"
REPORTS = ROOT / "reports"


def compute_angle(p1, p2, p3):
    """Compute angle at p2 between p1-p2-p3."""
    if p1 is None or p2 is None or p3 is None:
        return None
    v1 = np.array([p1[0] - p2[0], p1[1] - p2[1]])
    v2 = np.array([p3[0] - p2[0], p3[1] - p2[1]])
    if np.linalg.norm(v1) < 1e-6 or np.linalg.norm(v2) < 1e-6:
        return None
    cos_angle = np.dot(v1, v2) / (np.linalg.norm(v1) * np.linalg.norm(v2))
    cos_angle = np.clip(cos_angle, -1, 1)
    return np.arccos(cos_angle)


def extract_body_features(kpts, player_prefix):
    """Extract body part angles and positions from keypoints."""
    # YOLO keypoint format: [x, y, conf]
    # Standard 17-keypoint order
    if not kpts or len(kpts) < 11:
        return {}
    
    features = {}
    
    # Helper to get point
    def get_pt(idx):
        if idx >= len(kpts) or kpts[idx][2] < 0.3:
            return None
        return [kpts[idx][0], kpts[idx][1]]
    
    # Key body points (COCO format)
    # 0: nose, 5: left_shoulder, 6: right_shoulder, 7: left_elbow, 8: right_elbow
    # 9: left_wrist, 10: right_wrist, 11: left_hip, 12: right_hip
    # 13: left_knee, 14: right_knee, 15: left_ankle, 16: right_ankle
    
    # Shoulder angle (horizontal)
    l_shoulder = get_pt(5)
    r_shoulder = get_pt(6)
    if l_shoulder and r_shoulder:
        features[f"{player_prefix}_shoulder_angle"] = np.arctan2(
            r_shoulder[1] - l_shoulder[1], 
            r_shoulder[0] - l_shoulder[0]
        )
        features[f"{player_prefix}_shoulder_width"] = np.sqrt(
            (r_shoulder[0] - l_shoulder[0])**2 + (r_shoulder[1] - l_shoulder[1])**2
        )
    
    # Arm angles
    l_shoulder = get_pt(5)
    l_elbow = get_pt(7)
    l_wrist = get_pt(9)
    if l_shoulder and l_elbow and l_wrist:
        features[f"{player_prefix}_l_arm_angle"] = compute_angle(l_shoulder, l_elbow, l_wrist) or 0
    
    r_shoulder = get_pt(6)
    r_elbow = get_pt(8)
    r_wrist = get_pt(10)
    if r_shoulder and r_elbow and r_wrist:
        features[f"{player_prefix}_r_arm_angle"] = compute_angle(r_shoulder, r_elbow, r_wrist) or 0
    
    # Torso angle (shoulders to hips)
    l_hip = get_pt(11)
    r_hip = get_pt(12)
    if l_shoulder and r_shoulder and l_hip and r_hip:
        shoulder_mid = [(l_shoulder[0]+r_shoulder[0])/2, (l_shoulder[1]+r_shoulder[1])/2]
        hip_mid = [(l_hip[0]+r_hip[0])/2, (l_hip[1]+r_hip[1])/2]
        features[f"{player_prefix}_torso_angle"] = np.arctan2(
            shoulder_mid[1] - hip_mid[1],
            shoulder_mid[0] - hip_mid[0]
        )
        features[f"{player_prefix}_torso_height"] = np.sqrt(
            (shoulder_mid[0] - hip_mid[0])**2 + (shoulder_mid[1] - hip_mid[1])**2
        )
    
    # Leg angles
    l_hip = get_pt(11)
    l_knee = get_pt(13)
    l_ankle = get_pt(15)
    if l_hip and l_knee and l_ankle:
        features[f"{player_prefix}_l_leg_angle"] = compute_angle(l_hip, l_knee, l_ankle) or 0
    
    r_hip = get_pt(12)
    r_knee = get_pt(14)
    r_ankle = get_pt(16)
    if r_hip and r_knee and r_ankle:
        features[f"{player_prefix}_r_leg_angle"] = compute_angle(r_hip, r_knee, r_ankle) or 0
    
    # Wrist-to-shoulder distances (racket reach)
    if l_shoulder and l_wrist:
        features[f"{player_prefix}_l_reach"] = np.sqrt(
            (l_wrist[0] - l_shoulder[0])**2 + (l_wrist[1] - l_shoulder[1])**2
        )
    if r_shoulder and r_wrist:
        features[f"{player_prefix}_r_reach"] = np.sqrt(
            (r_wrist[0] - r_shoulder[0])**2 + (r_wrist[1] - r_shoulder[1])**2
        )
    
    return features


def main():
    print("Loading v6 features...")
    frames = {}
    with open(DATA / "frame_features_v6.jsonl") as f:
        for line in f:
            if not line.strip():
                continue
            r = json.loads(line)
            frames[r["frame"]] = r
    
    print(f"Loaded {len(frames)} frames")
    
    # Extract body part features
    print("Extracting body part features...")
    body_features = []
    
    for frame in sorted(frames.keys()):
        r = frames[frame]
        feat = {"frame": frame, "t_sec": r.get("t_sec", 0)}
        
        # Get players
        players = r.get("players", {}) or {}
        
        for p in ["X", "Y"]:
            player = players.get(p) or {}
            kpts = player.get("kpts")
            if kpts:
                body_feats = extract_body_features(kpts, p)
                feat.update(body_feats)
        
        body_features.append(feat)
    
    df = pd.DataFrame(body_features)
    print(f"Extracted features: {df.shape}")
    print(f"Feature columns: {list(df.columns)}")
    
    # Save
    out_path = DATA / "body_features_v14.csv"
    df.to_csv(out_path, index=False)
    print(f"Saved {out_path}")
    
    # Summary stats
    print("\nFeature statistics:")
    for col in df.columns:
        if col not in ["frame", "t_sec"]:
            valid = df[col].dropna()
            if len(valid) > 0:
                print(f"  {col}: mean={valid.mean():.3f}, std={valid.std():.3f}, n={len(valid)}")


if __name__ == "__main__":
    main()
