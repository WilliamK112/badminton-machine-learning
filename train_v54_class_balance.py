"""Train v54 model with class weight balancing to address severe imbalance"""
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import cross_val_score, StratifiedKFold
from sklearn.utils.class_weight import compute_class_weight
import json
import warnings
warnings.filterwarnings('ignore')

# Load data
combined = pd.read_csv('data/combined_features_v16.csv')
motion = pd.read_csv('data/motion_quant_v1.csv')
labels = pd.read_csv('data/rally_labels_v13.csv.gz')

# Select only the velocity/acceleration columns from motion
motion_features = motion.columns.tolist()[2:]

# Create merged dataset
merged = combined.copy()
for col in motion_features:
    merged[col] = motion[col].values

# Merge labels
merged = merged.merge(labels[['frame', 'label']], on='frame', how='left')
merged['win'] = merged['label'].fillna(0).astype(int)

# Prepare features and target
feature_cols = [c for c in merged.columns if c not in ['frame', 't_sec_x', 't_sec_y', 'label', 'win', 'rally_id']]
X = merged[feature_cols].fillna(0).replace([np.inf, -np.inf], 0)
y = merged['win']

print(f"Original class distribution: {y.value_counts().to_dict()}")

# Compute class weights to balance the dataset
# For imbalanced data: weight = n_samples / (n_classes * n_samples_class)
class_weights = compute_class_weight('balanced', classes=np.unique(y), y=y)
class_weight_dict = {0: class_weights[0], 1: class_weights[1]}
print(f"Class weights: {class_weight_dict}")

# Train with class weights
print("\n=== Training v54 with Class Weight Balancing ===")
rf_balanced = RandomForestClassifier(
    n_estimators=200, 
    max_depth=15, 
    min_samples_leaf=3, 
    random_state=54, 
    n_jobs=-1,
    class_weight='balanced'  # This automatically adjusts weights
)

# Use stratified CV to ensure both classes are represented in each fold
cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
cv_scores = cross_val_score(rf_balanced, X, y, cv=cv, scoring='accuracy')
print(f"CV Accuracy (stratified): {cv_scores.mean():.4f} (+/- {cv_scores.std():.4f})")

# Train on full data
rf_balanced.fit(X, y)
train_acc = rf_balanced.score(X, y)
print(f"Training accuracy: {train_acc:.4f}")

# Check predictions on training data
y_pred = rf_balanced.predict(X)
print(f"\nPrediction distribution: {pd.Series(y_pred).value_counts().to_dict()}")

# Feature importance
importances = pd.DataFrame({
    'feature': feature_cols,
    'importance': rf_balanced.feature_importances_
}).sort_values('importance', ascending=False)

print("\nTop 15 features:")
for i, row in importances.head(15).iterrows():
    print(f"  {row['feature']}: {row['importance']:.4f}")

# Save model
import joblib
joblib.dump(rf_balanced, 'models/v54.pkl')

# Save feature list
with open('models/v54_features.json', 'w') as f:
    json.dump(feature_cols, f, indent=2)

# Save results
result = {
    'version': 'v54',
    'model': 'RandomForest (class_weight=balanced)',
    'cv_accuracy': float(cv_scores.mean()),
    'cv_std': float(cv_scores.std()),
    'train_accuracy': float(train_acc),
    'n_features': len(feature_cols),
    'original_class_dist': y.value_counts().to_dict(),
    'prediction_dist': {int(k): int(v) for k, v in pd.Series(y_pred).value_counts().to_dict().items()},
    'class_weight': 'balanced'
}

with open('reports/model_v54.json', 'w') as f:
    json.dump(result, f, indent=2)

print(f"\n✅ v54 model trained with class balancing")
print(f"   CV Accuracy: {cv_scores.mean():.4f}")
print(f"   Original: 0={15}, 1={3037} -> Predictions: {pd.Series(y_pred).value_counts().to_dict()}")