#!/usr/bin/env python3
"""
Generate combined visual report: landing heatmap + win-prob timeline
Priority #5: Generate visual reports
"""
import json
import numpy as np
from PIL import Image, ImageDraw, ImageFont
import os

# Load landing data
with open('reports/landing_heatmap_v13.json', 'r') as f:
    landings = json.load(f)['landings']

# Load win-prob timeline
with open('reports/winprob_timeline_v2.json', 'r') as f:
    wp_data = json.load(f)  # It's a list directly

# Create court template (13.4m x 6.1m scaled to 400x600)
court_w, court_h = 400, 600
court = Image.new('RGB', (court_w, court_h), '#1a1a2e')
draw = ImageDraw.Draw(court)

# Draw court lines
court_lines = [
    (40, 40, 360, 40, 360, 560, 40, 560),
    (40, 260, 360, 260),
    (160, 40, 160, 260),
    (240, 40, 240, 260),
    (60, 40, 60, 560),
    (340, 40, 340, 560),
]

for args in court_lines:
    draw.line(args, fill='#4a9eff', width=2)

# Create landing density heatmap
grid_w, grid_h = 20, 30
heatmap = np.zeros((grid_h, grid_w))
for land in landings:
    x_idx = int(land['x'] * grid_w)
    y_idx = int(land['y'] * grid_h)
    if 0 <= x_idx < grid_w and 0 <= y_idx < grid_h:
        heatmap[y_idx, x_idx] += 1

# Apply heatmap directly to image pixels
if heatmap.max() > 0:
    heatmap_norm = heatmap / heatmap.max()
    cell_w = court_w // grid_w
    cell_h = court_h // grid_h
    for gy in range(grid_h):
        for gx in range(grid_w):
            v = heatmap_norm[gy, gx]
            if v > 0.05:
                x1, y1 = gx * cell_w, gy * cell_h
                x2, y2 = x1 + cell_w, y1 + cell_h
                # Red-ish heat
                r = int(255 * v)
                g = int(100 * v)
                b = int(100 * (1-v))
                draw.rectangle([x1, y1, x2, y2], fill=(r, g, b))

# Draw landing points with winner colors
for land in landings:
    x = int(land['x'] * court_w)
    y = int(land['y'] * court_h)
    color = '#00ff88' if land['winner'] == 1 else '#ff4466'
    draw.ellipse([x-6, y-6, x+6, y+6], fill=color, outline='white', width=1)

# Add text
try:
    font = ImageFont.truetype('/System/Library/Fonts/Helvetica.ttc', 14)
except:
    font = ImageFont.load_default()

draw.text((10, 10), "🏸 Landing Heatmap (v14)", fill='white', font=font)
draw.text((10, court_h - 25), "🟢 P1  🔴 P2", fill='#aaa', font=font)

# Save
court.save('reports/landing_combined_v14.png')
print("✓ Generated: reports/landing_combined_v14.png")

# Generate win-prob summary stats
if wp_data:
    p1_wins = sum(1 for p in wp_data if p.get('winner') == 1)
    p2_wins = len(wp_data) - p1_wins
    avg_p1 = np.mean([p.get('p1_winprob', 0.5) for p in wp_data])
    
    stats = {
        'total_rallies': len(wp_data),
        'p1_wins': p1_wins,
        'p2_wins': p2_wins,
        'avg_p1_winprob': round(avg_p1, 3),
        'landing_count': len(landings)
    }
    with open('reports/combined_stats_v14.json', 'w') as f:
        json.dump(stats, f, indent=2)
    print(f"✓ Generated: reports/combined_stats_v14.json")
    print(f"  → {p1_wins} P1 wins, {p2_wins} P2 wins, avg P1 prob: {avg_p1:.1%}")
else:
    print("⚠ No win-prob timeline data found")

print("\n✅ Visual reports complete!")
