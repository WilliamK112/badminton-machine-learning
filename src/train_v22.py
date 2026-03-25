#!/usr/bin/env python3
"""Train v22: Ensemble of best approaches - voting from RF, GB, and Logistic Regression"""
import json
import gzip
import pandas as pd
import numpy as np
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import accuracy_score, balanced_accuracy_score, f1_score, matthews_corrcoef
from sklearn.impute import SimpleImputer
from sklearn.feature_selection import SelectKBest, f_classif
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier, VotingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from datetime import datetime
from pathlib import Path
import warnings
warnings.filterwarnings('ignore')

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

# Build rally-level features - EXACT same as v19 for fair comparison
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
    
    feat_dict['avg_motion'] = row.get('avg_motion', 0)
    feat_dict['max_motion'] = row.get('max_motion', 0)
    
    for player in ['X', 'Y']:
        stance_col = f'{player}_stance_width'
        if stance_col in rally_frames_frame.columns:
            stance_vals = pd.to_numeric(rally_frames_frame[stance_col], errors='coerce').dropna()
            if len(stance_vals) > 0:
                feat_dict[f'{player}_stance_mean'] = stance_vals.mean()
    
    features_list.append(feat_dict)

df = pd.DataFrame(features_list)
print(f"Built {len(df)} rally samples with features")

feature_cols = [c for c in df.columns if c not in ['rally_idx', 'winner']]
X = df[feature_cols].values
y = df['winner'].values

imputer = SimpleImputer(strategy='median')
X = imputer.fit_transform(X)

selector = SelectKBest(f_classif, k=min(10, len(feature_cols)))
X_selected = selector.fit_transform(X, y)
selected_features = [feature_cols[i] for i in selector.get_support(indices=True)]
print(f"Selected features: {selected_features}")

print(f"Class distribution: winner=1: {sum(y==1)}, winner=0: {sum(y==0)}")

# Scale for logistic regression
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X_selected)

# Try different approaches
skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

print("\n=== Individual Model CV ===")
models = {
    'RF': RandomForestClassifier(n_estimators=100, max_depth=4, min_samples_leaf=2, class_weight='balanced', random_state=42),
    'GB': GradientBoostingClassifier(n_estimators=100, max_depth=4, learning_rate=0.1, random_state=42),
    'LR': LogisticRegression(class_weight='balanced', max_iter=1000, random_state=42),
}

best_individual = {}
for name, model in models.items():
    cv_scores = []
    for fold, (train_idx, test_idx) in enumerate(skf.split(X_scaled if name == 'LR' else X_selected, y)):
        if name == 'LR':
            X_tr, X_te = X_scaled[train_idx], X_scaled[test_idx]
        else:
            X_tr, X_te = X_selected[train_idx], X_selected[test_idx]
        y_tr, y_te = y[train_idx], y[test_idx]
        
        m = models[name].__class__(**models[name].get_params())
        m.fit(X_tr, y_tr)
        pred = m.predict(X_te)
        cv_scores.append(balanced_accuracy_score(y_te, pred))
    
    mean_score = np.mean(cv_scores)
    std_score = np.std(cv_scores)
    best_individual[name] = {'score': mean_score, 'std': std_score}
    print(f"{name}: CV = {mean_score:.3f} (+/- {std_score:.3f})")

# Try voting ensemble
print("\n=== Voting Ensemble CV ===")
ensemble_params = [
    {'voting': 'hard', 'weights': [1, 1, 1]},
    {'voting': 'soft', 'weights': [1, 1, 1]},
    {'voting': 'soft', 'weights': [2, 1, 1]},
    {'voting': 'soft', 'weights': [1, 2, 1]},
]

best_ensemble_score = 0
best_ensemble_params = None
ensemble_results = []

for params in ensemble_params:
    cv_scores = []
    for fold, (train_idx, test_idx) in enumerate(skf.split(X_selected, y)):
        X_tr, X_te = X_selected[train_idx], X_selected[test_idx]
        X_tr_s, X_te_s = X_scaled[train_idx], X_scaled[test_idx]
        y_tr, y_te = y[train_idx], y[test_idx]
        
        # Create fresh estimators for each fold
        rf = RandomForestClassifier(n_estimators=100, max_depth=4, min_samples_leaf=2, class_weight='balanced', random_state=42+fold)
        gb = GradientBoostingClassifier(n_estimators=100, max_depth=4, learning_rate=0.1, random_state=42+fold)
        lr = LogisticRegression(class_weight='balanced', max_iter=1000, random_state=42+fold)
        
        rf.fit(X_tr, y_tr)
        gb.fit(X_tr, y_tr)
        lr.fit(X_tr_s, y_tr)
        
        if params['voting'] == 'hard':
            pred_rf = rf.predict(X_te)
            pred_gb = gb.predict(X_te)
            pred_lr = lr.predict(X_te)
            # Hard voting: majority
            from scipy import stats
            predictions = np.array([pred_rf, pred_gb, pred_lr])
            pred = stats.mode(predictions, axis=0, keepdims=False)[0]
        else:
            # Soft voting: average probabilities
            prob_rf = rf.predict_proba(X_te)[:, 1]
            prob_gb = gb.predict_proba(X_te)[:, 1]
            prob_lr = lr.predict_proba(X_te)[:, 1]
            
            w = params['weights']
            avg_prob = (w[0]*prob_rf + w[1]*prob_gb + w[2]*prob_lr) / sum(w)
            pred = (avg_prob > 0.5).astype(int)
        
        cv_scores.append(balanced_accuracy_score(y_te, pred))
    
    mean_score = np.mean(cv_scores)
    std_score = np.std(cv_scores)
    ensemble_results.append({'params': params, 'score': mean_score, 'std': std_score})
    print(f"Ensemble {params}: CV = {mean_score:.3f} (+/- {std_score:.3f})")
    
    if mean_score > best_ensemble_score:
        best_ensemble_score = mean_score
        best_ensemble_params = params

# Find overall best
all_scores = {**{n: v['score'] for n, v in best_individual.items()}, **{f"ensemble_{i}": r['score'] for i, r in enumerate(ensemble_results)}}
best_method = max(all_scores, key=all_scores.get)
print(f"\n>>> Best method: {best_method} with CV = {all_scores[best_method]:.3f}")

# Temporal test with best approach
test_size = 8
X_train, X_test = X_selected[:-test_size], X_selected[-test_size:]
X_train_s, X_test_s = X_scaled[:-test_size], X_scaled[-test_size:]
y_train, y_test = y[:-test_size], y[-test_size:]

# Use the best performing approach
if best_method.startswith('ensemble'):
    idx = int(best_method.split('_')[1])
    params = ensemble_results[idx]['params']
    
    rf = RandomForestClassifier(n_estimators=100, max_depth=4, min_samples_leaf=2, class_weight='balanced', random_state=42)
    gb = GradientBoostingClassifier(n_estimators=100, max_depth=4, learning_rate=0.1, random_state=42)
    lr = LogisticRegression(class_weight='balanced', max_iter=1000, random_state=42)
    
    rf.fit(X_train, y_train)
    gb.fit(X_train, y_train)
    lr.fit(X_train_s, y_train)
    
    if params['voting'] == 'hard':
        pred_rf = rf.predict(X_test)
        pred_gb = gb.predict(X_test)
        pred_lr = lr.predict(X_test)
        from scipy import stats
        predictions = np.array([pred_rf, pred_gb, pred_lr])
        y_pred = stats.mode(predictions, axis=0, keepdims=False)[0]
    else:
        prob_rf = rf.predict_proba(X_test)[:, 1]
        prob_gb = gb.predict_proba(X_test)[:, 1]
        prob_lr = lr.predict_proba(X_test)[:, 1]
        w = params['weights']
        avg_prob = (w[0]*prob_rf + w[1]*prob_gb + w[2]*prob_lr) / sum(w)
        y_pred = (avg_prob > 0.5).astype(int)
    
    final_score = best_ensemble_score
    final_params = params
else:
    if best_method == 'RF':
        final_model = RandomForestClassifier(n_estimators=100, max_depth=4, min_samples_leaf=2, class_weight='balanced', random_state=42)
        final_model.fit(X_train, y_train)
    elif best_method == 'GB':
        final_model = GradientBoostingClassifier(n_estimators=100, max_depth=4, learning_rate=0.1, random_state=42)
        final_model.fit(X_train, y_train)
    else:
        final_model = LogisticRegression(class_weight='balanced', max_iter=1000, random_state=42)
        final_model.fit(X_train_s, y_train)
    
    y_pred = final_model.predict(X_test if best_method != 'LR' else X_test_s)
    final_score = best_individual[best_method]['score']
    final_params = best_method

temporal_acc = accuracy_score(y_test, y_pred)
temporal_bal = balanced_accuracy_score(y_test, y_pred)
temporal_f1 = f1_score(y_test, y_pred, average='macro', zero_division=0)
temporal_mcc = matthews_corrcoef(y_test, y_pred)

print(f"\nTemporal test ({test_size} samples): acc={temporal_acc:.3f}, balanced={temporal_bal:.3f}")

# Save report
report = {
    'timestamp_local': datetime.now().strftime('%Y-%m-%dT%H-%M-%S') + '-05:00',
    'step_executed': 'Train v22 - Ensemble voting (RF+GB+LR)',
    'samples': len(df),
    'selected_features': len(selected_features),
    'feature_names': selected_features,
    'class_distribution': {'winner_1': int(sum(y==1)), 'winner_0': int(sum(y==0))},
    'best_method': best_method,
    'best_params': str(final_params),
    'cv_balanced_accuracy': round(final_score, 3),
    'individual_scores': {k: round(v['score'], 3) for k, v in best_individual.items()},
    'ensemble_results': [{'params': str(r['params']), 'score': round(r['score'], 3)} for r in ensemble_results],
    'temporal_test': {
        'accuracy': round(temporal_acc, 3),
        'balanced_accuracy': round(temporal_bal, 3),
        'f1_macro': round(temporal_f1, 3),
        'mcc': round(temporal_mcc, 3)
    },
    'comparison_to_v19': f'v19 CV=0.733, v22 CV={final_score:.3f}',
    'next_step': 'Try stacking or add more data features'
}

out_path = ROOT / 'reports' / f"train_v22_{datetime.now().strftime('%Y-%m-%dT%H-%M-%S')}.json"
out_path.write_text(json.dumps(report, indent=2))
print(f"\nSaved: {out_path}")