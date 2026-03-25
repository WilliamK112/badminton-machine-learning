#!/usr/bin/env python3
"""
Train new model on v15 features + generate visual reports
"""
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import pickle
import json
from pathlib import Path
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import cross_val_score
import warnings
warnings.filterwarnings('ignore')

PROJECT_DIR = Path("/Users/William/.openclaw/workspace/projects/badminton-ai")
DATA_DIR = PROJECT_DIR / "data"
REPORTS_DIR = PROJECT_DIR / "reports"
MODELS_DIR = PROJECT_DIR / "models"

print("=" * 50)
print("Training v50 model on combined_features_v15")
print("=" * 50)

# Load combined features v15
df = pd.read_csv(DATA_DIR / "combined_features_v15.csv")
print(f"Combined features v15: {df.shape}")

# Load labels
labels_df = pd.read_csv(DATA_DIR / "quant_features_v5.csv")[['frame', 'winner_proxy']]
labels_df = labels_df.fillna(0)

# Merge labels
df = df.merge(labels_df[['frame', 'winner_proxy']], on='frame', how='inner')

# Filter to labeled frames
df = df[df['winner_proxy'].isin([0, 1])].copy()
print(f"With labels: {df.shape}")

# Clean column names
if 't_sec_x' in df.columns:
    df = df.rename(columns={'t_sec_x': 't_sec'})
if 't_sec_y' in df.columns:
    df = df.drop(columns=['t_sec_y'])

# Get numeric features
numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
exclude_cols = ['frame', 'winner_proxy', 'shuttle_dir']
feature_cols = [c for c in numeric_cols if c not in exclude_cols]

print(f"Features: {len(feature_cols)}")

X = df[feature_cols].fillna(0).values
y = df['winner_proxy'].values

# Train model
model = RandomForestClassifier(n_estimators=100, random_state=42, n_jobs=-1)
cv_scores = cross_val_score(model, X, y, cv=5, scoring='accuracy')
cv_mean = cv_scores.mean()
print(f"CV accuracy: {cv_mean:.3f} (+/- {cv_scores.std():.3f})")

# Fit final model
model.fit(X, y)

# Save model
model_path = MODELS_DIR / "v50.pkl"
pickle.dump(model, open(model_path, "wb"))

# Save feature list
feature_info = {
    'version': 'v50',
    'features': feature_cols,
    'cv_score': float(cv_mean),
    'n_features': len(feature_cols)
}
with open(MODELS_DIR / "v50_features.json", "w") as f:
    json.dump(feature_info, f, indent=2)

print(f"✅ Model saved: {model_path}")

# Generate win-prob timeline
print("\nGenerating win-prob timeline...")
probs = model.predict_proba(X)[:, 1]

timeline_df = pd.DataFrame({
    'frame': df['frame'],
    't_sec': df['t_sec'],
    'win_prob': probs
})

fig, ax = plt.subplots(figsize=(14, 5))
ax.plot(timeline_df['t_sec'], timeline_df['win_prob'], 'b-', linewidth=1.5, alpha=0.8)
ax.fill_between(timeline_df['t_sec'], timeline_df['win_prob'], alpha=0.2)
ax.set_xlabel('Time (seconds)', fontsize=11)
ax.set_ylabel('Win Probability', fontsize=11)
ax.set_title(f'Win Probability Timeline (v50 model, CV={cv_mean:.3f})', fontsize=13)
ax.set_ylim(0, 1)
ax.grid(True, alpha=0.3)
plt.tight_layout()

output_path = REPORTS_DIR / "winprob_timeline_v50.png"
plt.savefig(output_path, dpi=120)
plt.close()

# Save stats
stats = {
    'model': 'v50',
    'cv_score': float(cv_mean),
    'features_used': len(feature_cols),
    'n_frames': len(df),
    'mean_prob': float(probs.mean()),
    'std_prob': float(probs.std())
}
with open(REPORTS_DIR / "winprob_stats_v50.json", "w") as f:
    json.dump(stats, f, indent=2)

timeline_df.to_csv(REPORTS_DIR / "winprob_timeline_v50.csv", index=False)
print(f"✅ Win-prob timeline: {output_path}")

# Generate landing heatmap
print("\nGenerating landing heatmap...")
x = df['shuttle_x'].dropna()
y = df['shuttle_y'].dropna()

mask = (x >= 0) & (x <= 1) & (y >= 0) & (y <= 1)
x = x[mask]
y = y[mask]

if len(x) > 10:
    fig, ax = plt.subplots(figsize=(8, 10))
    h = ax.hist2d(x, y, bins=20, cmap='YlOrRd', alpha=0.8)
    plt.colorbar(h[3], ax=ax, label='Frame Count')
    ax.axhline(y=0.5, color='white', linestyle='--', linewidth=1.5)
    ax.axvline(x=0.5, color='white', linestyle='--', linewidth=1.5)
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.set_xlabel('Court Width (normalized)')
    ax.set_ylabel('Court Length (normalized)')
    ax.set_title('Shuttle Landing Heatmap')
    ax.invert_yaxis()
    plt.tight_layout()
    
    hm_path = REPORTS_DIR / "landing_heatmap_v50.png"
    plt.savefig(hm_path, dpi=120)
    plt.close()
    
    hm_stats = {
        'n_points': len(x),
        'mean_x': float(x.mean()),
        'mean_y': float(y.mean())
    }
    with open(REPORTS_DIR / "landing_heatmap_v50.json", "w") as f:
        json.dump(hm_stats, f, indent=2)
    
    print(f"✅ Landing heatmap: {hm_path}")

print("\n✅ Visual reports v50 complete!")
