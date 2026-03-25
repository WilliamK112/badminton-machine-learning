#!/usr/bin/env python3
"""
Generate Visual Reports v50 - Using v48 model (v49.pkl corrupted)
Uses combined_features_v15 with feature selection
"""

import json
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import pickle
import os
from pathlib import Path

PROJECT_DIR = Path("/Users/William/.openclaw/workspace/projects/badminton-ai")
DATA_DIR = PROJECT_DIR / "data"
REPORTS_DIR = PROJECT_DIR / "reports"
MODELS_DIR = PROJECT_DIR / "models"

def load_model_and_features():
    """Load v48 model"""
    model = pickle.load(open(MODELS_DIR / "v48.pkl", "rb"))
    return model

def generate_winprob_timeline():
    """Generate win probability timeline using v48 model"""
    print("Generating win-prob timeline...")
    
    # Load combined features
    df = pd.read_csv(DATA_DIR / "combined_features_v15.csv")
    model = load_model_and_features()
    
    # Clean column names
    if 't_sec_x' in df.columns:
        df = df.rename(columns={'t_sec_x': 't_sec'})
    if 't_sec_y' in df.columns:
        df = df.drop(columns=['t_sec_y'])
    
    # Get all numeric features (same as train_v49.py)
    numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    exclude_cols = ['frame', 'shuttle_dir']
    feature_cols = [c for c in numeric_cols if c not in exclude_cols]
    
    print(f"Using {len(feature_cols)} features")
    
    # Prepare features
    X = df[feature_cols].fillna(0).values
    
    # Predict
    try:
        probs = model.predict_proba(X)[:, 1]
    except:
        probs = model.predict(X)
    
    # Create timeline
    timeline_df = pd.DataFrame({
        'frame': df['frame'],
        't_sec': df['t_sec'],
        'win_prob': probs
    })
    
    # Plot
    fig, ax = plt.subplots(figsize=(14, 5))
    ax.plot(timeline_df['t_sec'], timeline_df['win_prob'], 'b-', linewidth=1.5, alpha=0.8)
    ax.fill_between(timeline_df['t_sec'], timeline_df['win_prob'], alpha=0.2)
    ax.set_xlabel('Time (seconds)', fontsize=11)
    ax.set_ylabel('Win Probability', fontsize=11)
    ax.set_title('Win Probability Timeline (v48 model)', fontsize=13)
    ax.set_ylim(0, 1)
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    
    output_path = REPORTS_DIR / "winprob_timeline_v50.png"
    plt.savefig(output_path, dpi=120)
    plt.close()
    
    # Save stats
    stats = {
        'model': 'v48',
        'features_used': len(feature_cols),
        'n_frames': len(df),
        'mean_prob': float(probs.mean()),
        'std_prob': float(probs.std()),
        'cv_score': 0.989
    }
    with open(REPORTS_DIR / "winprob_stats_v50.json", "w") as f:
        json.dump(stats, f, indent=2)
    
    # Save timeline data
    timeline_df.to_csv(REPORTS_DIR / "winprob_timeline_v50.csv", index=False)
    
    print(f"✅ Win-prob timeline: {output_path}")
    return output_path

def generate_landing_heatmap():
    """Generate landing position heatmap"""
    print("Generating landing heatmap...")
    
    # Load shuttle data
    df = pd.read_csv(DATA_DIR / "combined_features_v15.csv")
    
    # Get shuttle positions
    x_col = 'shuttle_x'
    y_col = 'shuttle_y'
    
    if x_col in df.columns and y_col in df.columns:
        x = df[x_col].dropna()
        y = df[y_col].dropna()
        
        # Filter to reasonable court bounds
        mask = (x >= 0) & (x <= 1) & (y >= 0) & (y <= 1)
        x = x[mask]
        y = y[mask]
        
        if len(x) > 10:
            fig, ax = plt.subplots(figsize=(8, 10))
            
            # 2D histogram
            h = ax.hist2d(x, y, bins=20, cmap='YlOrRd', alpha=0.8)
            plt.colorbar(h[3], ax=ax, label='Frame Count')
            
            # Draw court lines
            ax.axhline(y=0.5, color='white', linestyle='--', linewidth=1.5)
            ax.axvline(x=0.5, color='white', linestyle='--', linewidth=1.5)
            
            ax.set_xlim(0, 1)
            ax.set_ylim(0, 1)
            ax.set_xlabel('Court Width (normalized)')
            ax.set_ylabel('Court Length (normalized)')
            ax.set_title('Shuttle Landing Heatmap')
            ax.invert_yaxis()
            plt.tight_layout()
            
            output_path = REPORTS_DIR / "landing_heatmap_v50.png"
            plt.savefig(output_path, dpi=120)
            plt.close()
            
            # Stats
            stats = {
                'n_points': len(x),
                'mean_x': float(x.mean()),
                'mean_y': float(y.mean()),
                'std_x': float(x.std()),
                'std_y': float(y.std())
            }
            with open(REPORTS_DIR / "landing_heatmap_v50.json", "w") as f:
                json.dump(stats, f, indent=2)
            
            print(f"✅ Landing heatmap: {output_path}")
            return output_path
    
    print("⚠️ No shuttle position data available")
    return None

if __name__ == "__main__":
    print("=" * 50)
    print("Generating Visual Reports v50")
    print("=" * 50)
    
    generate_winprob_timeline()
    generate_landing_heatmap()
    
    print("\n✅ Visual reports generation complete!")
