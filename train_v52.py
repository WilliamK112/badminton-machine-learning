#!/usr/bin/env python3
"""
Train v52 model using combined_features_v16 with rally_labels_v13.
Prioritize motion/velocity features + shuttle tracking.
"""
import pandas as pd
import numpy as np
import json
import pickle
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.model_selection import cross_val_score
from sklearn.preprocessing import StandardScaler
from sklearn.feature_selection import SelectKBest, f_classif
import warnings
warnings.filterwarnings('ignore')

# Load features
features_df = pd.read_csv('data/combined_features_v16.csv')
print(f"Loaded features: {features_df.shape}")

# Load rally labels
rally_df = pd.read_csv('data/rally_labels_v13.csv.gz')
print(f"Loaded rally labels: {rally_df.shape}")

# Merge
merged = features_df.merge(rally_df[['frame', 'label']], on='frame', how='inner')
print(f"Merged dataset: {merged.shape}")

# Drop rows with missing labels
merged = merged.dropna(subset=['label'])
merged['label'] = merged['label'].astype(int)
print(f"After dropping NaN: {merged.shape}, label distribution: {merged['label'].value_counts().to_dict()}")

# Separate features and label
feature_cols = [c for c in merged.columns if c not in ['frame', 'label', 'rally_id', 't_sec_x', 't_sec_y']]
X = merged[feature_cols].fillna(0).replace([np.inf, -np.inf], 0)
y = merged['label'].values

print(f"Using {len(feature_cols)} features: {feature_cols[:10]}...")

# Feature selection - pick top 25 features
selector = SelectKBest(f_classif, k=min(25, len(feature_cols)))
X_selected = selector.fit_transform(X, y)
selected_indices = selector.get_support(indices=True)
selected_features = [feature_cols[i] for i in selected_indices]
print(f"Selected {len(selected_features)} features: {selected_features}")

# Scale
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X_selected)

# Train RandomForest
rf = RandomForestClassifier(n_estimators=200, max_depth=10, min_samples_leaf=2, random_state=42, n_jobs=-1)
cv_scores = cross_val_score(rf, X_scaled, y, cv=5, scoring='accuracy')
print(f"RandomForest CV: {cv_scores.mean():.4f} (+/- {cv_scores.std()*2:.4f})")

# Train GradientBoosting
gb = GradientBoostingClassifier(n_estimators=100, max_depth=5, learning_rate=0.1, random_state=42)
gb_cv = cross_val_score(gb, X_scaled, y, cv=5, scoring='accuracy')
print(f"GradientBoosting CV: {gb_cv.mean():.4f} (+/- {gb_cv.std()*2:.4f})")

# Use better model
if cv_scores.mean() >= gb_cv.mean():
    best_model = rf
    best_cv = cv_scores.mean()
    model_name = "RandomForest"
else:
    best_model = gb
    best_cv = gb_cv.mean()
    model_name = "GradientBoosting"

# Fit on full data
best_model.fit(X_scaled, y)

# Get feature importance
if hasattr(best_model, 'feature_importances_'):
    importance = pd.DataFrame({
        'feature': selected_features,
        'importance': best_model.feature_importances_
    }).sort_values('importance', ascending=False)
    importance.to_csv('models/v52_importance.csv', index=False)
    print(f"\nTop 10 features:")
    print(importance.head(10).to_string(index=False))

# Save model
model_data = {
    'model': best_model,
    'scaler': scaler,
    'selector': selector,
    'features': selected_features,
    'cv_score': best_cv,
    'model_name': model_name
}
with open('models/v52.pkl', 'wb') as f:
    pickle.dump(model_data, f)

# Save metadata
with open('models/v52_features.json', 'w') as f:
    json.dump({
        'features': selected_features,
        'cv_score': float(best_cv),
        'model_name': model_name,
        'n_features': len(selected_features),
        'n_samples': len(y)
    }, f, indent=2)

print(f"\n✅ v52 model saved: CV={best_cv:.4f}, {model_name}, {len(selected_features)} features")
