"""
Feature Selection + Model Training v49
Uses combined_features_v15 (body + shuttle + racket proxy) with feature selection
Goal: Improve on v48 (CV=0.989) by adding shuttle/racket features while keeping best features
"""
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import cross_val_score
from sklearn.feature_selection import SelectKBest, f_classif
from sklearn.preprocessing import StandardScaler
import warnings
warnings.filterwarnings('ignore')

# Load combined features v15
df = pd.read_csv('data/combined_features_v15.csv')
print(f"Combined features v15: {df.shape}")

# Load labels - use winner_proxy from quant_features_v5
labels_df = pd.read_csv('data/quant_features_v5.csv')[['frame', 'winner_proxy']]
labels_df = labels_df.fillna(0)

# Merge labels
df = df.merge(labels_df[['frame', 'winner_proxy']], on='frame', how='inner')

# Filter to only labeled frames
df = df[df['winner_proxy'].isin([0, 1])].copy()
print(f"With labels: {df.shape}")

# Clean up column names (handle t_sec_x vs t_sec_y)
if 't_sec_x' in df.columns:
    df = df.rename(columns={'t_sec_x': 't_sec'})
    if 't_sec_y' in df.columns:
        df = df.drop(columns=['t_sec_y'])

# Only use numeric columns as features
numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
exclude_cols = ['frame', 'winner_proxy', 'shuttle_dir']
feature_cols = [c for c in numeric_cols if c not in exclude_cols]

print(f"Feature columns: {len(feature_cols)}")
print(f"Features: {feature_cols}")

X = df[feature_cols].values
y = df['winner_proxy'].values

# Handle any NaN values
X = np.nan_to_num(X, nan=0.0)

print(f"Total features: {len(feature_cols)}")
print(f"Samples: {len(X)}")

# Feature Selection using SelectKBest
print("\n=== Feature Selection ===")
selector = SelectKBest(f_classif, k='all')
selector.fit(X, y)
scores = selector.scores_

# Rank features by score
feature_scores = list(zip(feature_cols, scores))
feature_scores.sort(key=lambda x: x[1], reverse=True)

print("\nTop 20 features by ANOVA F-score:")
for i, (name, score) in enumerate(feature_scores[:20]):
    print(f"  {i+1}. {name}: {score:.4f}")

# Test different k values
print("\n=== Testing different k values ===")
best_cv = 0
best_k = 10

for k in [10, 15, 20, 25, 30]:
    if k > len(feature_cols):
        continue
    selector = SelectKBest(f_classif, k=k)
    X_selected = selector.fit_transform(X, y)
    
    clf = RandomForestClassifier(n_estimators=100, max_depth=10, 
                                  random_state=42, n_jobs=-1)
    cv_scores = cross_val_score(clf, X_selected, y, cv=5, scoring='accuracy')
    cv_mean = cv_scores.mean()
    
    print(f"k={k}: CV accuracy = {cv_mean:.3f} (+/- {cv_scores.std():.3f})")
    
    if cv_mean > best_cv:
        best_cv = cv_mean
        best_k = k

print(f"\nBest k={best_k} with CV={best_cv:.3f}")

# Final model with best k
selector = SelectKBest(f_classif, k=best_k)
X_selected = selector.fit_transform(X, y)

# Get selected feature names
selected_indices = selector.get_support(indices=True)
selected_features = [feature_cols[i] for i in selected_indices]
print(f"\nSelected {len(selected_features)} features: {selected_features}")

clf = RandomForestClassifier(n_estimators=200, max_depth=15, 
                              random_state=42, n_jobs=-1)
cv_scores = cross_val_score(clf, X_selected, y, cv=5, scoring='accuracy')

print(f"\n=== Final Model v49 ===")
print(f"CV Accuracy: {cv_scores.mean():.3f} (+/- {cv_scores.std():.3f})")
print(f"Features used: {len(selected_features)}")

# Train final model
clf.fit(X_selected, y)

# Save model and features
import joblib
joblib.dump(clf, 'models/v49.pkl')
joblib.dump(selected_features, 'models/v49_features.pkl')

# Save feature importance
importances = clf.feature_importances_
imp_df = pd.DataFrame({'feature': selected_features, 'importance': importances})
imp_df = imp_df.sort_values('importance', ascending=False)
imp_df.to_csv('models/v49_importance.csv', index=False)
print("\nTop 10 features by importance:")
print(imp_df.head(10).to_string(index=False))

print(f"\nModel saved to models/v49.pkl")
print(f"Comparison: v48 CV=0.989, v49 CV={cv_scores.mean():.3f}")
