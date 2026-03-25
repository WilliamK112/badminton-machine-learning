#!/usr/bin/env python3
"""
Court homography transformer - converts normalized video coordinates to real court positions.
Badminton court dimensions: 13.4m x 6.1m (singles)
"""

import json
import numpy as np

# Court corners in video (from data/court_corners.json)
VIDEO_CORNERS = np.array([
    [461, 151],    # top-left
    [1459, 151],  # top-right  
    [1804, 950],  # bottom-right
    [116, 950]    # bottom-left
], dtype=np.float64)

# Court real dimensions (meters) - singles court
# Origin at bottom-left (server's right court corner)
COURT_WIDTH = 6.1   # 6.1m (singles width)
COURT_LENGTH = 13.4 # 13.4m (full court length)

# Real court corners in meters (counter-clockwise from bottom-left)
REAL_CORNERS = np.array([
    [0, 0],           # bottom-left
    [COURT_WIDTH, 0], # bottom-right
    [COURT_WIDTH, COURT_LENGTH],  # top-right
    [0, COURT_LENGTH] # top-left
], dtype=np.float64)

def compute_homography():
    """Compute perspective transform from video to real court."""
    H, status = np.linalg.lstsq(
        np.vstack([
            np.column_stack([VIDEO_CORNERS[:,0], VIDEO_CORNERS[:,1], 
                           np.ones(4), np.zeros(4), np.zeros(4), np.zeros(4),
                           -VIDEO_CORNERS[:,0]*REAL_CORNERS[:,0], 
                           -VIDEO_CORNERS[:,1]*REAL_CORNERS[:,0]]),
            np.column_stack([np.zeros(4), VIDEO_CORNERS[:,0], VIDEO_CORNERS[:,1],
                           np.zeros(4), np.zeros(4), VIDEO_CORNERS[:,0], 
                           VIDEO_CORNERS[:,1], np.zeros(4),
                           -VIDEO_CORNERS[:,0]*REAL_CORNERS[:,1],
                           -VIDEO_CORNERS[:,1]*REAL_CORNERS[:,1]])
        ]),
        REAL_CORNERS.flatten(),
        rcond=None
    )
    # Actually use OpenCV-style computation
    H, _ = compute_homography_cv(VIDEO_CORNERS, REAL_CORNERS)
    return H

def compute_homography_cv(src, dst):
    """Compute homography using DLT algorithm."""
    n = src.shape[0]
    A = np.zeros((2*n, 9))
    
    for i in range(n):
        x, y = src[i]
        u, v = dst[i]
        A[2*i] = [-x, -y, -1, 0, 0, 0, u*x, u*y, u]
        A[2*i+1] = [0, 0, 0, -x, -y, -1, v*x, v*y, v]
    
    _, _, Vt = np.linalg.svd(A)
    H = Vt[-1].reshape(3, 3)
    H = H / H[2, 2]  # Normalize
    
    return H, np.ones(n, dtype=bool)

def transform_point(x, y, H):
    """Transform a single point from video to real court coordinates."""
    pt = np.array([x, y, 1])
    transformed = H @ pt
    return transformed[0] / transformed[2], transformed[1] / transformed[2]

def transform_normalized(nx, ny, H, image_width=1920, image_height=1080):
    """Transform normalized (0-1) coordinates to real court."""
    # Convert normalized to pixel coordinates
    px = nx * image_width
    py = ny * image_height
    return transform_point(px, py, H)

def analyze_landing_positions(features_file, output_file):
    """Analyze shuttle landing positions in real court coordinates."""
    H, _ = compute_homography_cv(VIDEO_CORNERS, REAL_CORNERS)
    
    # Load features
    landings = []
    with open(features_file) as f:
        for line in f:
            data = json.loads(line)
            shuttle = data.get('shuttle', {})
            if shuttle and shuttle.get('visible') and shuttle.get('xy'):
                # Check for low speed (potential landing)
                speed = shuttle.get('speed', 1.0)
                if speed < 0.02:  # Low speed = likely landed
                    nx, ny = shuttle['xy']
                    mx, my = transform_normalized(nx, ny, H)
                    if 0 <= mx <= COURT_WIDTH and 0 <= my <= COURT_LENGTH:
                        landings.append({
                            'frame': data['frame'],
                            't_sec': data['t_sec'],
                            'court_x': round(mx, 2),
                            'court_y': round(my, 2)
                        })
    
    result = {
        'homography': H.tolist(),
        'court_dimensions_meters': {'width': COURT_WIDTH, 'length': COURT_LENGTH},
        'total_landings': len(landings),
        'landings': landings[:20],  # Sample
        'analysis': {
            'court_sections': {
                'front_left': sum(1 for l in landings if l['court_x'] < COURT_WIDTH/2 and l['court_y'] < COURT_LENGTH/2),
                'front_right': sum(1 for l in landings if l['court_x'] >= COURT_WIDTH/2 and l['court_y'] < COURT_LENGTH/2),
                'back_left': sum(1 for l in landings if l['court_x'] < COURT_WIDTH/2 and l['court_y'] >= COURT_LENGTH/2),
                'back_right': sum(1 for l in landings if l['court_x'] >= COURT_WIDTH/2 and l['court_y'] >= COURT_LENGTH/2),
            }
        }
    }
    
    with open(output_file, 'w') as f:
        json.dump(result, f, indent=2)
    
    return result

if __name__ == '__main__':
    import sys
    H, _ = compute_homography_cv(VIDEO_CORNERS, REAL_CORNERS)
    print("Homography matrix:")
    print(H)
    
    # Test transform
    test_x, test_y = 0.5, 0.5  # Center of frame
    mx, my = transform_normalized(test_x, test_y, H)
    print(f"\nTest: normalized ({test_x}, {test_y}) -> court ({mx:.2f}m, {my:.2f}m)")
    
    # Analyze landings
    if len(sys.argv) > 1:
        result = analyze_landing_positions(sys.argv[1], sys.argv[2] if len(sys.argv) > 2 else 'court_analysis.json')
        print(f"\nAnalyzed {result['total_landings']} landing positions")
