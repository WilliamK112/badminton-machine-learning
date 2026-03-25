#!/usr/bin/env python3
"""Train v47: Combine v6 frame features + body features v14 + motion quantification"""

import json
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import cross_val_score
from sklearn.preprocessing import StandardScaler
import pickle
import warnings
warnings.filterwarnings('ignore')

def load_frame_features_v6(path='data/frame_features_v6.jsonl'):
    """Load frame features from jsonl"""
    records = []
    with open(path) as f:
        for line in f:
            records.append(json.loads(line))
    df = pd.DataFrame(records)
    
    # Flatten player features - handle None values
    for p in ['X', 'Y']:
        # Stance and center
        def get_stance(x):
            if x is None or not isinstance(x, dict):
                return 0
            return x.get('stance_width', 0)
        
        def get_cog(x, idx):
            if x is None or not isinstance(x, dict):
                return 0
            cog = x.get('cog', [0,0])
            return cog[idx] if len(cog) > idx else 0
        
        df[f'{p}_stance_width'] = df['players'].apply(lambda x: get_stance(x[p]) if isinstance(x, dict) else 0)
        df[f'{p}_cog_x'] = df['players'].apply(lambda x: get_cog(x, 0) if isinstance(x, dict) else 0)
        df[f'{p}_cog_y'] = df['players'].apply(lambda x: get_cog(x, 1) if isinstance(x, dict) else 0)
        
    # Keep frame, t_sec, and basic features
    keep_cols = ['frame', 't_sec', 'shuttle_dir_change', 'X_stance_width', 'X_cog_x', 'X_cog_y', 'Y_stance_width', 'Y_cog_x', 'Y_cog_y']
    df = df[[c for c in keep_cols if c in df.columns]]
    return df

def load_body_features_v14(path='data/body_features_v14.csv'):
    """Load body features from CSV"""
    df = pd.read_csv(path)
    # Drop t_sec duplicate
    if 't_sec' in df.columns:
        df = df.drop(columns=['t_sec'])
    return df

def load_motion_quant(path='data/motion_quant_v1.csv'):
    """Load motion quantification features"""
    df = pd.read_csv(path)
    if 't_sec' in df.columns:
        df = df.drop(columns=['t_sec'])
    return df

def main():
    print("Loading features...")
    
    # Load all three feature sets
    frame_df = load_frame_features_v6()
    print(f"Frame features: {len(frame_df)} rows, cols: {list(frame_df.columns)}")
    
    body_df = load_body_features_v14()
    print(f"Body features: {len(body_df)} rows, cols: {len(body_df.columns)-2}")
    
    motion_df = load_motion_quant()
    print(f"Motion features: {len(motion_df)} rows, cols: {len(motion_df.columns)-2}")
    
    # Merge on frame
    print("Merging features...")
    merged = frame_df.merge(body_df, on='frame', how='inner')
    merged = merged.merge(motion_df, on='frame', how='inner')
    print(f"Merged: {len(merged)} rows")
    
    # Load labels (rally detection: 1 = in rally, 0 = not in rally)
    print("Loading labels...")
    import gzip
    labels = pd.read_csv('data/rally_labels_v10.csv.gz')
    labels["label"] = (labels["label"] == 1).astype(int)
    print(f"Labels: {labels.label.value_counts().to_dict()}")
    
    # Merge with labels
    merged = merged.merge(labels[['frame', 'label']], on='frame', how='inner')
    print(f"With labels: {len(merged)} rows")
    
    # Prepare features
    feature_cols = [c for c in merged.columns if c not in ['frame', 't_sec', 'label']]
    X = merged[feature_cols].fillna(0).values
    y = merged['label'].values
    
    print(f"Features: {len(feature_cols)} columns")
    print(f"Class distribution: rally={sum(y)}, not_rally={len(y)-sum(y)}")
    
    # Scale features
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    
    # Train model
    print("Training RandomForest...")
    model = RandomForestClassifier(
        n_estimators=200,
        max_depth=15,
        min_samples_split=5,
        min_samples_leaf=2,
        class_weight='balanced',
        random_state=123,
        n_jobs=-1
    )
    
    # Cross-validation
    print("Running 5-fold CV...")
    cv_scores = cross_val_score(model, X_scaled, y, cv=5, scoring='accuracy')
    cv_mean = cv_scores.mean()
    cv_std = cv_scores.std()
    print(f"CV Accuracy: {cv_mean:.3f} (+/- {cv_std:.3f})")
    
    # Fit on full data
    model.fit(X_scaled, y)
    
    # Save model
    model_path = 'models/v47.pkl'
    with open(model_path, 'wb') as f:
        pickle.dump({'model': model, 'scaler': scaler, 'features': feature_cols}, f)
    print(f"Saved model to {model_path}")
    
    # Save feature list
    feature_info = {
        'version': 'v47',
        'n_features': len(feature_cols),
        'features': feature_cols,
        'cv_accuracy': cv_mean,
        'cv_std': cv_std,
        'sources': ['frame_features_v6', 'body_features_v14', 'motion_quant_v1']
    }
    with open('models/v47_features.json', 'w') as f:
        json.dump(feature_info, f, indent=2)
    
    print(f"\n=== v47 Training Complete ===")
    print(f"Features: {len(feature_cols)} (v6 frame + v14 body + motion)")
    print(f"CV Accuracy: {cv_mean:.3f}")
    print(f"Model saved: {model_path}")

if __name__ == '__main__':
    main()