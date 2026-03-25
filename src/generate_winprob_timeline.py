#!/usr/bin/env python3
"""Generate win-probability timeline visualization."""
import json
import matplotlib.pyplot as plt
import numpy as np

def main():
    with open('full_pipeline_output.json') as f:
        data = json.load(f)
    
    timeline = data.get('timeline', [])
    if not timeline:
        print("No timeline data found")
        return
    
    times = [e['t_sec'] for e in timeline]
    win_prob_a = [e.get('win_prob_a', 0.5) for e in timeline]
    win_prob_b = [e.get('win_prob_b', 0.5) for e in timeline]
    
    fig, ax = plt.subplots(figsize=(12, 5))
    ax.fill_between(times, win_prob_a, alpha=0.3, label='Player A', color='blue')
    ax.fill_between(times, win_prob_b, alpha=0.3, label='Player B', color='red')
    ax.plot(times, win_prob_a, color='blue', linewidth=1.5)
    ax.plot(times, win_prob_b, color='red', linewidth=1.5)
    
    ax.set_xlabel('Time (seconds)')
    ax.set_ylabel('Win Probability')
    ax.set_title('Badminton Rally Win Probability Timeline')
    ax.set_ylim(0, 1)
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    output_path = 'reports/win_probability_timeline.png'
    plt.savefig(output_path, dpi=100)
    print(f"Saved: {output_path}")
    
    # Also save summary stats
    stats = {
        'duration_sec': max(times) if times else 0,
        'samples': len(timeline),
        'avg_win_prob_a': np.mean(win_prob_a),
        'avg_win_prob_b': np.mean(win_prob_b),
        'max_lead_a': max(win_prob_a),
        'max_lead_b': max(win_prob_b),
    }
    with open('reports/winprob_stats.json', 'w') as f:
        json.dump(stats, f, indent=2)
    print(f"Saved: reports/winprob_stats.json")

if __name__ == '__main__':
    main()
