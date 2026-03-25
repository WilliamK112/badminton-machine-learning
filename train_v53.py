"""Train v53 model with motion quantification features"""
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import cross_val_score
import json
import warnings
warnings.filterwarnings('ignore')

# Load data
combined = pd.read_csv('data/combined_features_v16.csv')
motion = pd.read_csv('data/motion_quant_v1.csv')
labels = pd.read_csv('data/rally_labels_v13.csv.gz')

# Select only the velocity/acceleration columns from motion (exclude frame, t_sec)
motion_features = motion.columns.tolist()[2:]  # Skip frame, t_sec
print(f"Motion features: {len(motion_features)}")

# Create merged dataset - add motion features to combined
merged = combined.copy()
for col in motion_features:
    merged[col] = motion[col].values

# Merge labels
merged = merged.merge(labels[['frame', 'label']], on='frame', how='left')
merged['win'] = merged['label'].fillna(0).astype(int)

print(f"Total features: {merged.shape[1] - 4}")  # -frame, -t_sec, -label, -win

# Prepare features and target
feature_cols = [c for c in merged.columns if c not in ['frame', 't_sec_x', 't_sec_y', 'label', 'win', 'rally_id']]
X = merged[feature_cols].fillna(0).replace([np.inf, -np.inf], 0)
y = merged['win']

print(f"X shape: {X.shape}, y shape: {y.shape}")
print(f"Class distribution: {y.value_counts().to_dict()}")

# Train RandomForest with motion features
print("\n=== Training RandomForest with Motion Features (v53) ===")
rf = RandomForestClassifier(n_estimators=200, max_depth=15, min_samples_leaf=3, random_state=53, n_jobs=-1)
cv_scores = cross_val_score(rf, X, y, cv=5, scoring='accuracy')
print(f"CV Accuracy: {cv_scores.mean():.4f} (+/- {cv_scores.std():.4f})")

# Train on full data
rf.fit(X, y)
print(f"Training accuracy: {rf.score(X, y):.4f}")

# Feature importance - top 20
importances = pd.DataFrame({
    'feature': feature_cols,
    'importance': rf.feature_importances_
}).sort_values('importance', ascending=False)

print("\nTop 20 features:")
for i, row in importances.head(20).iterrows():
    motion_tag = " [MOTION]" if 'vel' in row['feature'] else ""
    print(f"  {row['feature']}: {row['importance']:.4f}{motion_tag}")

# Count motion features in top 20
motion_in_top20 = sum(1 for f in importances.head(20)['feature'] if 'vel' in f)
print(f"\nMotion features in top 20: {motion_in_top20}")

# Save model info
result = {
    'version': 'v53',
    'model': 'RandomForest',
    'cv_accuracy': float(cv_scores.mean()),
    'cv_std': float(cv_scores.std()),
    'train_accuracy': float(rf.score(X, y)),
    'n_features': len(feature_cols),
    'n_motion_features': len(motion_features),
    'motion_in_top20': motion_in_top20
}

with open('reports/model_v53.json', 'w') as f:
    json.dump(result, f, indent=2)

print(f"\n✅ v53 model trained with {len(motion_features)} motion features")
print(f"   CV Accuracy: {cv_scores.mean():.4f}")