#!/usr/bin/env python3
"""
predict_v1.py - Use trained v31 model for predictions
This loads the best model (class_weight='balanced' from v31) and can predict winner for new rallies
"""
import gzip
import json
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler
import joblib
import os

ROOT = "/Users/William/.openclaw/workspace/projects/badminton-ai"

def load_model():
    """Load the trained model and scaler"""
    # For now, retrain on the fly since we don't have a saved model file
    # This uses the same configuration as v31 (best performing)
    
    with gzip.open(f"{ROOT}/data/quant_features_v11.csv.gz", 'rt') as f:
        df = pd.read_csv(f)
    
    df_labels = pd.read_csv(f"{ROOT}/data/rally_labels_v4.csv")
    
    features = []
    for idx, row in df_labels.iterrows():
        sf, ef = int(row['start_frame']), int(row['end_frame'])
        if sf >= len(df) or ef >= len(df):
            continue
        rally_df = df[(df['frame'] >= sf) & (df['frame'] <= ef)]
        if len(rally_df) == 0:
            continue
        feat = {
            'winner': row['winner'],
            'shuttle_y_end': rally_df['shuttle_y'].iloc[-1] if 'shuttle_y' in rally_df else 0,
            'is_deep': 1 if rally_df['shuttle_y'].iloc[-1] > 0.5 else 0,
            'X_arms_ang_vel_max': rally_df['X_arms_ang_vel'].max() if 'X_arms_ang_vel' in rally_df else 0,
            'X_torso_ang_vel_max': rally_df['X_torso_ang_vel'].max() if 'X_torso_ang_vel' in rally_df else 0,
            'X_legs_ang_vel_max': rally_df['X_legs_ang_vel'].max() if 'X_legs_ang_vel' in rally_df else 0,
            'Y_arms_ang_vel_max': rally_df['Y_arms_ang_vel'].max() if 'Y_arms_ang_vel' in rally_df else 0,
            'avg_motion': rally_df['shuttle_speed'].mean() if 'shuttle_speed' in rally_df else 0,
            'max_motion': rally_df['shuttle_speed'].max() if 'shuttle_speed' in rally_df else 0,
            'X_stance_mean': rally_df['X_torso_rot'].mean() if 'X_torso_rot' in rally_df else 0,
            'Y_stance_mean': rally_df['Y_torso_rot'].mean() if 'Y_torso_rot' in rally_df else 0,
        }
        features.append(feat)
    
    df_feat = pd.DataFrame(features)
    
    feature_cols = [
        'shuttle_y_end', 'is_deep', 'X_arms_ang_vel_max', 'X_torso_ang_vel_max',
        'X_legs_ang_vel_max', 'Y_arms_ang_vel_max', 'avg_motion', 'max_motion',
        'X_stance_mean', 'Y_stance_mean'
    ]
    
    X = df_feat[feature_cols].values
    y = df_feat['winner'].values
    
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    
    # Best config from v31
    clf = RandomForestClassifier(
        n_estimators=100, 
        max_depth=5,
        min_samples_leaf=2, 
        class_weight='balanced', 
        random_state=42
    )
    clf.fit(X_scaled, y)
    
    return clf, scaler, feature_cols

def predict_rally(features_dict):
    """Predict winner for a rally given features"""
    clf, scaler, feature_cols = load_model()
    
    # Extract features in correct order
    feat_values = [features_dict.get(col, 0) for col in feature_cols]
    X = np.array([feat_values])
    X_scaled = scaler.transform(X)
    
    pred = clf.predict(X_scaled)[0]
    prob = clf.predict_proba(X_scaled)[0]
    
    return {
        'predicted_winner': int(pred),
        'probability_player_0': float(prob[0]),
        'probability_player_1': float(prob[1])
    }

def extract_rally_features(df, start_frame, end_frame):
    """Extract features from a rally in a video frame dataframe"""
    rally_df = df[(df['frame'] >= start_frame) & (df['frame'] <= end_frame)]
    
    if len(rally_df) == 0:
        return None
    
    features = {
        'shuttle_y_end': rally_df['shuttle_y'].iloc[-1] if 'shuttle_y' in rally_df else 0,
        'is_deep': 1 if rally_df['shuttle_y'].iloc[-1] > 0.5 else 0,
        'X_arms_ang_vel_max': rally_df['X_arms_ang_vel'].max() if 'X_arms_ang_vel' in rally_df else 0,
        'X_torso_ang_vel_max': rally_df['X_torso_ang_vel'].max() if 'X_torso_ang_vel' in rally_df else 0,
        'X_legs_ang_vel_max': rally_df['X_legs_ang_vel'].max() if 'X_legs_ang_vel' in rally_df else 0,
        'Y_arms_ang_vel_max': rally_df['Y_arms_ang_vel'].max() if 'Y_arms_ang_vel' in rally_df else 0,
        'avg_motion': rally_df['shuttle_speed'].mean() if 'shuttle_speed' in rally_df else 0,
        'max_motion': rally_df['shuttle_speed'].max() if 'shuttle_speed' in rally_df else 0,
        'X_stance_mean': rally_df['X_torso_rot'].mean() if 'X_torso_rot' in rally_df else 0,
        'Y_stance_mean': rally_df['Y_torso_rot'].mean() if 'Y_torso_rot' in rally_df else 0,
    }
    return features

if __name__ == "__main__":
    # Test prediction on existing data
    print("Loading model...")
    clf, scaler, feature_cols = load_model()
    
    # Test on training data
    with gzip.open(f"{ROOT}/data/quant_features_v11.csv.gz", 'rt') as f:
        df = pd.read_csv(f)
    
    df_labels = pd.read_csv(f"{ROOT}/data/rally_labels_v4.csv")
    
    # Test on first 5 rallies
    print("\nTesting on sample rallies:")
    for idx, row in df_labels.head(5).iterrows():
        sf, ef = int(row['start_frame']), int(row['end_frame'])
        if sf >= len(df) or ef >= len(df):
            continue
        features = extract_rally_features(df, sf, ef)
        if features:
            result = predict_rally(features)
            actual = row['winner']
            print(f"Rally {idx}: Predicted={result['predicted_winner']}, Actual={actual}, "
                  f"P(0)={result['probability_player_0']:.2f}, P(1)={result['probability_player_1']:.2f}")
    
    print("\nModel ready for predictions!")
    print("Use predict_rally(features_dict) to predict winners for new rallies.")