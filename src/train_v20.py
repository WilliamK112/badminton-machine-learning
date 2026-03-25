#!/usr/bin/env python3
"""Train v20: Try XGBoost model instead of RandomForest for potentially better performance"""
import json
import gzip
import pandas as pd
import numpy as np
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import accuracy_score, balanced_accuracy_score, f1_score, matthews_corrcoef
from sklearn.impute import SimpleImputer
from sklearn.feature_selection import SelectKBest, f_classif
from datetime import datetime
from pathlib import Path
import warnings
warnings.filterwarnings('ignore')

try:
    import xgboost as xgb
    HAS_XGB = True
except ImportError:
    HAS_XGB = False
    print("XGBoost not available, will use alternative approach")

ROOT = Path(__file__).resolve().parents[1]

# Load rally labels with motion features
with gzip.open(ROOT / 'data/rally_labels_v9.csv.gz', 'rt') as f:
    rally_v9 = pd.read_csv(f)
print(f"Loaded {len(rally_v9)} rallies with motion features")

# Load quant features
with gzip.open(ROOT / 'data/quant_features_v11.csv.gz', 'rt') as f:
    quant = pd.read_csv(f)
print(f"Loaded {len(quant)} frame features")

# Load frame features for stance
import json as json_lib
frame_features = []
with open(ROOT / 'data/frame_features_v10.jsonl', 'r') as f:
    for line in f:
        frame_features.append(json_lib.loads(line))
frame_df = pd.DataFrame(frame_features)
print(f"Loaded {len(frame_df)} frame features with stance data")

# Build rally-level features
features_list = []
for _, row in rally_v9.iterrows():
    start, end = int(row['start_frame']), int(row['end_frame'])
    rally_frames = quant[(quant['frame'] >= start) & (quant['frame'] <= end)]
    rally_frames_frame = frame_df[(frame_df['frame'] >= start) & (frame_df['frame'] <= end)]
    
    if len(rally_frames) < 3:
        continue
    
    end_row = rally_frames.iloc[-1]
    shuttle_x_end = end_row['shuttle_x']
    shuttle_y_end = end_row['shuttle_y']
    dist_from_center = abs(shuttle_x_end - 0.5)
    is_deep = 1 if shuttle_y_end > 0.7 else 0
    
    feat_dict = {
        'rally_idx': len(features_list),
        'winner': int(row['winner']),
        'shuttle_x_end': shuttle_x_end,
        'shuttle_y_end': shuttle_y_end,
        'dist_from_center': dist_from_center,
        'is_deep': is_deep,
    }
    
    for player in ['X', 'Y']:
        for joint in ['arms', 'torso', 'legs']:
            col = f"{player}_{joint}_ang_vel"
            if col in rally_frames.columns:
                vals = rally_frames[col].dropna()
                feat_dict[f'{player}_{joint}_ang_vel_max'] = vals.max() if len(vals) else 0
                feat_dict[f'{player}_{joint}_ang_vel_mean'] = vals.mean() if len(vals) else 0
                feat_dict[f'{player}_{joint}_ang_vel_std'] = vals.std() if len(vals) > 1 else 0
    
    feat_dict['avg_motion'] = row.get('avg_motion', 0)
    feat_dict['max_motion'] = row.get('max_motion', 0)
    
    for player in ['X', 'Y']:
        stance_col = f'{player}_stance_width'
        if stance_col in rally_frames_frame.columns:
            stance_vals = pd.to_numeric(rally_frames_frame[stance_col], errors='coerce').dropna()
            if len(stance_vals) > 0:
                feat_dict[f'{player}_stance_mean'] = stance_vals.mean()
                feat_dict[f'{player}_stance_std'] = stance_vals.std() if len(stance_vals) > 1 else 0
    
    features_list.append(feat_dict)

df = pd.DataFrame(features_list)
print(f"Built {len(df)} rally samples with features")

feature_cols = [c for c in df.columns if c not in ['rally_idx', 'winner']]
X = df[feature_cols].values
y = df['winner'].values

imputer = SimpleImputer(strategy='median')
X = imputer.fit_transform(X)

selector = SelectKBest(f_classif, k=min(12, len(feature_cols)))
X_selected = selector.fit_transform(X, y)
selected_features = [feature_cols[i] for i in selector.get_support(indices=True)]
print(f"Selected features ({len(selected_features)}): {selected_features}")

print(f"Class distribution: winner=1: {sum(y==1)}, winner=0: {sum(y==0)}")

# Test with XGBoost if available
if HAS_XGB:
    print("\n=== XGBoost Training ===")
    xgb_params = [
        {'n_estimators': 50, 'max_depth': 3, 'learning_rate': 0.1, 'subsample': 0.8},
        {'n_estimators': 100, 'max_depth': 4, 'learning_rate': 0.1, 'subsample': 0.8},
        {'n_estimators': 100, 'max_depth': 5, 'learning_rate': 0.05, 'subsample': 0.9},
        {'n_estimators': 150, 'max_depth': 3, 'learning_rate': 0.1, 'subsample': 0.8},
        {'n_estimators': 200, 'max_depth': 4, 'learning_rate': 0.05, 'subsample': 0.9},
    ]
    
    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    best_score = 0
    best_params = None
    xgb_results = []
    
    for params in xgb_params:
        cv_scores = []
        for fold, (train_idx, test_idx) in enumerate(skf.split(X_selected, y)):
            X_train, X_test = X_selected[train_idx], X_selected[test_idx]
            y_train, y_test = y[train_idx], y[test_idx]
            
            # Compute scale_pos_weight for imbalance
            scale_pos_weight = sum(y_train == 0) / max(1, sum(y_train == 1))
            
            clf = xgb.XGBClassifier(
                n_estimators=params['n_estimators'],
                max_depth=params['max_depth'],
                learning_rate=params['learning_rate'],
                subsample=params['subsample'],
                scale_pos_weight=scale_pos_weight,
                random_state=42+fold,
                eval_metric='logloss',
                verbosity=0
            )
            clf.fit(X_train, y_train)
            pred = clf.predict(X_test)
            cv_scores.append(balanced_accuracy_score(y_test, pred))
        
        mean_score = np.mean(cv_scores)
        std_score = np.std(cv_scores)
        xgb_results.append({'params': params, 'score': mean_score, 'std': std_score})
        print(f"XGB: {params} -> CV: {mean_score:.3f} (+/- {std_score:.3f})")
        
        if mean_score > best_score:
            best_score = mean_score
            best_params = params
    
    print(f"\nBest XGB: {best_params} -> CV: {best_score:.3f}")
    
    # Temporal test with best params
    test_size = 8
    X_train, X_test = X_selected[:-test_size], X_selected[-test_size:]
    y_train, y_test = y[:-test_size], y[-test_size:]
    
    scale_pos_weight = sum(y_train == 0) / max(1, sum(y_train == 1))
    final_xgb = xgb.XGBClassifier(
        n_estimators=best_params['n_estimators'],
        max_depth=best_params['max_depth'],
        learning_rate=best_params['learning_rate'],
        subsample=best_params['subsample'],
        scale_pos_weight=scale_pos_weight,
        random_state=42,
        eval_metric='logloss',
        verbosity=0
    )
    final_xgb.fit(X_train, y_train)
    y_pred = final_xgb.predict(X_test)
    
    temporal_acc = accuracy_score(y_test, y_pred)
    temporal_bal = balanced_accuracy_score(y_test, y_pred)
    temporal_f1 = f1_score(y_test, y_pred, average='macro', zero_division=0)
    temporal_mcc = matthews_corrcoef(y_test, y_pred)
    
    print(f"Temporal test ({test_size} samples): acc={temporal_acc:.3f}, balanced={temporal_bal:.3f}")
    
    # Feature importances
    importances = dict(zip(selected_features, final_xgb.feature_importances_))
    top_features = dict(sorted(importances.items(), key=lambda x: -x[1])[:8])
    print(f"Top features: {top_features}")
    
    # Save report
    report = {
        'timestamp_local': datetime.now().strftime('%Y-%m-%dT%H-%M-%S') + '-05:00',
        'step_executed': 'Train v20 - XGBoost model with expanded features',
        'samples': len(df),
        'train': len(df) - test_size,
        'test': test_size,
        'features': len(selected_features),
        'feature_names': selected_features,
        'class_distribution': {'winner_1': int(sum(y==1)), 'winner_0': int(sum(y==0))},
        'model': 'XGBoost',
        'best_params': best_params,
        'cv_balanced_accuracy': round(best_score, 3),
        'xgb_results': [{'params': str(r['params']), 'score': round(r['score'], 3), 'std': round(r['std'], 3)} for r in xgb_results],
        'temporal_test': {
            'accuracy': round(temporal_acc, 3),
            'balanced_accuracy': round(temporal_bal, 3),
            'f1_macro': round(temporal_f1, 3),
            'mcc': round(temporal_mcc, 3)
        },
        'top_features': {k: round(float(v), 3) for k, v in top_features.items()},
        'comparison_to_v19': 'v19 CV=0.733, v20 result shown above',
        'next_step': 'Compare results, try LightGBM or further feature engineering'
    }
else:
    # Fallback to enhanced RF
    from sklearn.ensemble import GradientBoostingClassifier
    
    print("\n=== Gradient Boosting (fallback) ===")
    gb_params = [
        {'n_estimators': 50, 'max_depth': 3, 'learning_rate': 0.1},
        {'n_estimators': 100, 'max_depth': 4, 'learning_rate': 0.1},
        {'n_estimators': 100, 'max_depth': 5, 'learning_rate': 0.05},
    ]
    
    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    best_score = 0
    best_params = None
    gb_results = []
    
    for params in gb_params:
        cv_scores = []
        for fold, (train_idx, test_idx) in enumerate(skf.split(X_selected, y)):
            X_train, X_test = X_selected[train_idx], X_selected[test_idx]
            y_train, y_test = y[train_idx], y[test_idx]
            
            clf = GradientBoostingClassifier(
                n_estimators=params['n_estimators'],
                max_depth=params['max_depth'],
                learning_rate=params['learning_rate'],
                random_state=42+fold
            )
            clf.fit(X_train, y_train)
            pred = clf.predict(X_test)
            cv_scores.append(balanced_accuracy_score(y_test, pred))
        
        mean_score = np.mean(cv_scores)
        std_score = np.std(cv_scores)
        gb_results.append({'params': params, 'score': mean_score, 'std': std_score})
        print(f"GB: {params} -> CV: {mean_score:.3f} (+/- {std_score:.3f})")
        
        if mean_score > best_score:
            best_score = mean_score
            best_params = params
    
    # Similar temporal test...
    test_size = 8
    X_train, X_test = X_selected[:-test_size], X_selected[-test_size:]
    y_train, y_test = y[:-test_size], y[-test_size:]
    
    final_gb = GradientBoostingClassifier(
        n_estimators=best_params['n_estimators'],
        max_depth=best_params['max_depth'],
        learning_rate=best_params['learning_rate'],
        random_state=42
    )
    final_gb.fit(X_train, y_train)
    y_pred = final_gb.predict(X_test)
    
    temporal_acc = accuracy_score(y_test, y_pred)
    temporal_bal = balanced_accuracy_score(y_test, y_pred)
    
    report = {
        'timestamp_local': datetime.now().strftime('%Y-%m-%dT%H-%M-%S') + '-05:00',
        'step_executed': 'Train v20 - GradientBoosting (XGBoost not available)',
        'samples': len(df),
        'cv_balanced_accuracy': round(best_score, 3),
        'temporal_test': {'accuracy': round(temporal_acc, 3), 'balanced_accuracy': round(temporal_bal, 3)},
    }

out_path = ROOT / 'reports' / f"train_v20_{datetime.now().strftime('%Y-%m-%dT%H-%M-%S')}.json"
out_path.write_text(json.dumps(report, indent=2))
print(f"\nSaved: {out_path}")