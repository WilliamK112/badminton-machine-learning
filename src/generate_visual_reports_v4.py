#!/usr/bin/env python3
"""Generate visual reports: landing heatmap + win probability timeline."""
import gzip
import json
import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "data"
REPORTS = ROOT / "reports"

REPORTS.mkdir(parents=True, exist_ok=True)


def load_frame_features():
    """Load frame features from latest version."""
    path = DATA / "frame_features_v10.jsonl.gz"
    rows = []
    with gzip.open(path, 'rt') as f:
        for line in f:
            if line.strip():
                rows.append(json.loads(line))
    return rows


def load_rally_labels():
    """Load rally labels."""
    path = DATA / "rally_labels_v9.csv.gz"
    import csv
    rallies = {}
    with gzip.open(path, 'rt') as f:
        reader = csv.DictReader(f)
        for row in reader:
            rallies[int(row['rally_id'])] = row
    return rallies


def get_shuttle_pos(shuttle):
    """Extract shuttle (x, y) from shuttle dict."""
    if not shuttle:
        return None, None
    if isinstance(shuttle, dict):
        xy = shuttle.get('xy')
        if xy and isinstance(xy, list) and len(xy) >= 2:
            return float(xy[0]), float(xy[1])
    return None, None


def create_landing_heatmap(rows):
    """Create shuttle landing positions heatmap."""
    # Extract shuttle endpoints (where shuttle_dir_change indicates direction change)
    landing_positions = []
    prev_shuttle = None
    
    for row in rows:
        shuttle = row.get('shuttle', {})
        if shuttle and prev_shuttle:
            # Check if direction changed significantly (likely landing)
            if row.get('shuttle_dir_change', 0) > 0.8:
                sx, sy = get_shuttle_pos(shuttle)
                if sx is not None:
                    landing_positions.append((sx, sy))
        prev_shuttle = shuttle
    
    if not landing_positions:
        # Fallback: use last shuttle position when speed drops to near zero (landing)
        current_rally = None
        for row in rows:
            rally_id = row.get('frame', 0) // 1000
            shuttle = row.get('shuttle', {})
            speed = shuttle.get('speed', 0) if isinstance(shuttle, dict) else 0
            sx, sy = get_shuttle_pos(shuttle)
            if sx is not None and speed < 0.05:  # near stationary = landing
                if current_rally != rally_id:
                    landing_positions.append((sx, sy))
                    current_rally = rally_id
    
    if not landing_positions:
        print("No landing positions found")
        return None
    
    xs = [p[0] for p in landing_positions]
    ys = [p[1] for p in landing_positions]
    
    # Create heatmap
    fig, ax = plt.subplots(figsize=(10, 8))
    
    # Draw court
    court_width, court_length = 6.1, 13.4
    ax.add_patch(plt.Rectangle((0, 0), court_width, court_length, 
                                fill=False, edgecolor='black', linewidth=2))
    ax.add_patch(plt.Rectangle((0, 0), court_width/2, court_length/2, 
                                fill=False, edgecolor='black', linewidth=1, linestyle='--'))
    ax.add_patch(plt.Rectangle((court_width/2, court_length/2), court_width/2, court_length/2, 
                                fill=False, edgecolor='black', linewidth=1, linestyle='--'))
    
    # Plot landing positions
    if xs and ys:
        ax.scatter(xs, ys, alpha=0.6, s=50, c='red', edgecolors='darkred')
        
        # 2D histogram for density
        try:
            heatmap, xedges, yedges = np.histogram2d(xs, ys, bins=15, 
                                                      range=[[0, court_width], [0, court_length]])
            extent = [0, court_width, 0, court_length]
            ax.imshow(heatmap.T, origin='lower', extent=extent, 
                     cmap='hot', alpha=0.3, aspect='auto')
        except:
            pass
    
    ax.set_xlim(0, court_width)
    ax.set_ylim(0, court_length)
    ax.set_xlabel('Width (m)')
    ax.set_ylabel('Length (m)')
    ax.set_title('Shuttle Landing Positions Heatmap')
    ax.set_aspect('equal')
    
    output = REPORTS / "landing_heatmap.png"
    plt.savefig(output, dpi=100, bbox_inches='tight')
    plt.close()
    print(f"Saved: {output}")
    return len(landing_positions)


def create_win_prob_timeline(rows):
    """Create simplified win probability timeline."""
    # Create timeline based on shuttle position changes
    times = []
    prob_a = []
    
    court_width = 6.1
    court_center = court_width / 2
    
    for row in rows:
        t = row.get('t_sec', 0)
        shuttle = row.get('shuttle', {})
        
        sx, sy = get_shuttle_pos(shuttle)
        
        if sx is not None:
            # Player A is near x=0, Player B is near x=court_width
            # As shuttle moves to opponent side, their chance increases
            if sx < court_center:
                p_a = 0.5 + (court_center - sx) / court_center * 0.4
            else:
                p_a = 0.5 - (sx - court_center) / court_center * 0.4
            
            times.append(t)
            prob_a.append(max(0.1, min(0.9, p_a)))
    
    if not times:
        print("No timeline data")
        return None
    
    prob_b = [1 - p for p in prob_a]
    
    fig, ax = plt.subplots(figsize=(12, 5))
    ax.fill_between(times, prob_a, alpha=0.3, label='Player A', color='blue')
    ax.fill_between(times, prob_b, alpha=0.3, label='Player B', color='red')
    ax.plot(times, prob_a, color='blue', linewidth=1.5)
    ax.plot(times, prob_b, color='red', linewidth=1.5)
    
    ax.set_xlabel('Time (seconds)')
    ax.set_ylabel('Win Probability')
    ax.set_title('Rally Win Probability Timeline (Based on Shuttle Position)')
    ax.set_ylim(0, 1)
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    output = REPORTS / "win_probability_timeline.png"
    plt.savefig(output, dpi=100, bbox_inches='tight')
    plt.close()
    print(f"Saved: {output}")
    
    # Stats
    stats = {
        'duration_sec': max(times) if times else 0,
        'samples': len(times),
        'avg_win_prob_a': float(np.mean(prob_a)),
        'avg_win_prob_b': float(np.mean(prob_b)),
    }
    with open(REPORTS / "winprob_stats.json", 'w') as f:
        json.dump(stats, f, indent=2)
    print(f"Saved: {REPORTS / 'winprob_stats.json'}")
    
    return len(times)


def main():
    print("Loading frame features...")
    rows = load_frame_features()
    print(f"Loaded {len(rows)} frames")
    
    print("\nCreating landing heatmap...")
    n_landings = create_landing_heatmap(rows)
    
    print("\nCreating win probability timeline...")
    n_timeline = create_win_prob_timeline(rows)
    
    print(f"\n=== Visual Reports Generated ===")
    print(f"Landing positions: {n_landings}")
    print(f"Timeline points: {n_timeline}")
    print(f"Output: {REPORTS}")


if __name__ == '__main__':
    main()