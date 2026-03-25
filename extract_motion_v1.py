#!/usr/bin/env python3
"""Extract motion quantification features: angular velocity from body features v14"""

import pandas as pd
import numpy as np
import json
import sys

def compute_angular_velocity(body_df):
    """Compute angular velocity features from body features"""
    
    # Define angle columns for each player
    angle_cols = [
        'shoulder_angle', 'shoulder_width', 'l_arm_angle', 'r_arm_angle',
        'torso_angle', 'torso_height', 'l_leg_angle', 'r_leg_angle',
        'l_reach', 'r_reach'
    ]
    
    # Time column
    t_col = 't_sec'
    
    # Compute angular velocity (delta angle / delta time)
    body_df = body_df.sort_values(t_col).reset_index(drop=True)
    body_df['dt'] = body_df[t_col].diff()
    
    # Compute velocities for X player
    x_cols = [f'X_{col}' for col in angle_cols]
    y_cols = [f'Y_{col}' for col in angle_cols]
    
    vel_cols = []
    for col in x_cols + y_cols:
        if col in body_df.columns:
            body_df[f'{col}_vel'] = body_df[col].diff() / body_df['dt']
            vel_cols.append(f'{col}_vel')
    
    # Also compute acceleration (delta velocity / delta time)
    for col in vel_cols:
        body_df[f'{col}_acc'] = body_df[col].diff() / body_df['dt']
    
    return body_df

def main():
    print("Loading body features v14...")
    body_df = pd.read_csv('data/body_features_v14.csv')
    print(f"Loaded {len(body_df)} frames")
    
    print("Computing angular velocity features...")
    body_df = compute_angular_velocity(body_df)
    
    # Select velocity and acceleration columns
    vel_cols = [c for c in body_df.columns if '_vel' in c or '_acc' in c]
    vel_df = body_df[['frame', 't_sec'] + vel_cols].copy()
    
    # Fill NaN with 0 (first rows)
    vel_df = vel_df.fillna(0)
    
    # Save
    output_path = 'data/motion_quant_v1.csv'
    vel_df.to_csv(output_path, index=False)
    print(f"Saved motion quantification features to {output_path}")
    print(f"Features: {len(vel_cols)} velocity/acceleration columns")
    
    # Summary stats
    stats = {
        'n_frames': len(vel_df),
        'n_velocity_features': len([c for c in vel_cols if '_vel' in c]),
        'n_acceleration_features': len([c for c in vel_cols if '_acc' in c]),
        'features': vel_cols
    }
    
    with open('data/motion_quant_v1.json', 'w') as f:
        json.dump(stats, f, indent=2)
    print(f"Saved stats to data/motion_quant_v1.json")
    
    print(f"\nMotion quantification complete!")
    print(f"- {stats['n_velocity_features']} velocity features")
    print(f"- {stats['n_acceleration_features']} acceleration features")

if __name__ == '__main__':
    main()
