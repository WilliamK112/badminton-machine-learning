"""
Enhanced feature extraction v15: body + shuttle + racket proxy features
Combines: body_features_v14 + frame features (shuttle) + racket proxy
"""
import pandas as pd
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]

# Load body features (v14)
body = pd.read_csv(ROOT / 'data' / 'body_features_v14.csv')
print(f"Loaded body features: {len(body)} rows")

# Load shuttle features from frame_features_v6.jsonl
shuttle_rows = []
with open(ROOT / 'data' / 'frame_features_v6.jsonl') as f:
    for line in f:
        shuttle_rows.append(json.loads(line))
shuttle_df = pd.DataFrame(shuttle_rows)
print(f"Loaded shuttle features: {len(shuttle_df)} rows")
print(f"Shuttle columns: {list(shuttle_df.columns)}")

# Use shuttle column which is a list [x, y], and extract x, y components
if 'shuttle' in shuttle_df.columns:
    # shuttle is [x, y] coordinates
    shuttle_df['shuttle_x'] = shuttle_df['shuttle'].apply(lambda x: x[0] if isinstance(x, list) and len(x) > 0 else 0)
    shuttle_df['shuttle_y'] = shuttle_df['shuttle'].apply(lambda x: x[1] if isinstance(x, list) and len(x) > 1 else 0)

# Get shuttle columns
shuttle_cols = ['frame', 't_sec', 'shuttle_x', 'shuttle_y', 'shuttle_dir_change']

# Merge on frame
df = pd.merge(body, shuttle_df[shuttle_cols], on='frame', how='left')
print(f"Combined: {len(df)} rows")

# Add racket proxy features (from wrist/elbow surrogate)
# Racket is near right arm - use r_arm_angle changes as proxy
for side in ['X', 'Y']:
    r_arm = f'{side}_r_arm_angle'
    l_arm = f'{side}_l_arm_angle'
    df[f'{side}_racket_proxy_dir'] = (df[r_arm] - df[l_arm]).fillna(0)
    df[f'{side}_racket_proxy_speed'] = df[f'{side}_racket_proxy_dir'].diff().fillna(0).abs()

# Shuttle direction change (velocity direction)
if 'shuttle_x' in df.columns and 'shuttle_y' in df.columns:
    dx = df['shuttle_x'].diff().fillna(0)
    dy = df['shuttle_y'].diff().fillna(0)
    df['shuttle_dir'] = pd.Series([0]*len(dx))
    for i in range(1, len(dx)):
        if dx.iloc[i] != 0 or dy.iloc[i] != 0:
            df.iloc[i, df.columns.get_loc('shuttle_dir')] = 1 if dx.iloc[i] > 0 else -1
    df['shuttle_dir_change'] = (df['shuttle_dir'].diff() != 0).astype(int).fillna(0)

# Save combined features
out_path = ROOT / 'data' / 'combined_features_v15.csv'
df.to_csv(out_path, index=False)
print(f"Saved combined features to {out_path}")
print(f"Total features: {len(df.columns)}")
print(f"Columns: {list(df.columns)}")
