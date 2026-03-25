import json
from pathlib import Path
from datetime import datetime

ROOT = Path(__file__).resolve().parents[1]
in_path = ROOT / 'data' / 'frame_features_v7.jsonl'
out_path = ROOT / 'data' / 'frame_features_v8.jsonl'
report_path = ROOT / 'reports' / f'feature_extraction_upgrade_v8_{datetime.now().strftime("%Y-%m-%dT%H-%M-%S")}.json'
audit_path = ROOT / 'reports' / 'feature_quality_audit_v8.json'

rows = [json.loads(x) for x in in_path.read_text().splitlines() if x.strip()]

max_shuttle_gap = 3
ema_alpha = 0.35

last_vis_idx = None
last_xy = None
filled_shuttle = 0
raw_visible = 0

for i, r in enumerate(rows):
    sh = r.get('shuttle') or {}
    vis = bool(sh.get('visible', False) and sh.get('xy') is not None)

    if vis:
        raw_visible += 1
        xy = sh['xy']
        if last_xy is None:
            smooth = [float(xy[0]), float(xy[1])]
        else:
            smooth = [
                float(ema_alpha * xy[0] + (1.0 - ema_alpha) * last_xy[0]),
                float(ema_alpha * xy[1] + (1.0 - ema_alpha) * last_xy[1]),
            ]
        sh['xy_raw'] = [float(xy[0]), float(xy[1])]
        sh['xy'] = smooth
        sh['smoothed_v8'] = True
        r['shuttle'] = sh
        last_xy = smooth
        last_vis_idx = i
        continue

    if last_xy is not None and last_vis_idx is not None and (i - last_vis_idx) <= max_shuttle_gap:
        # Short-gap forward fill for trajectory continuity
        sh['xy'] = [float(last_xy[0]), float(last_xy[1])]
        sh['visible'] = True
        sh['source'] = 'interp'
        sh['filled_v8'] = True
        r['shuttle'] = sh
        filled_shuttle += 1
    else:
        r['shuttle'] = sh

N = len(rows)
x_detect = sum(1 for r in rows if r.get('players', {}).get('X') is not None)
y_detect = sum(1 for r in rows if r.get('players', {}).get('Y') is not None)
ball_vis = sum(1 for r in rows if r.get('shuttle', {}).get('visible', False))

score = (
    0.35 * (x_detect / N if N else 0)
    + 0.35 * (y_detect / N if N else 0)
    + 0.30 * (ball_vis / N if N else 0)
) * 100

with out_path.open('w') as f:
    for r in rows:
        f.write(json.dumps(r) + '\n')

report = {
    'input': str(in_path),
    'output': str(out_path),
    'frames': N,
    'player_X_detect_rate': round(x_detect / N, 4) if N else 0,
    'player_Y_detect_rate': round(y_detect / N, 4) if N else 0,
    'shuttle_visible_rate_raw': round(raw_visible / N, 4) if N else 0,
    'shuttle_visible_rate_post_v8': round(ball_vis / N, 4) if N else 0,
    'feature_quality_score_0_100': round(score, 2),
    'changes': {
        'shuttle_interp_filled_frames_v8': filled_shuttle,
        'max_shuttle_gap': max_shuttle_gap,
        'ema_alpha': ema_alpha,
    },
}

report_path.write_text(json.dumps(report, indent=2))
audit_path.write_text(json.dumps(report, indent=2))
print('saved', out_path)
print('saved', report_path)
print('saved', audit_path)
print(json.dumps(report, indent=2))
