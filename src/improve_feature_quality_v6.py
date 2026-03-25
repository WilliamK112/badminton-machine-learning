import json
from pathlib import Path
from copy import deepcopy
from datetime import datetime

ROOT = Path(__file__).resolve().parents[1]
in_path = ROOT / 'data' / 'frame_features_v5.jsonl'
out_path = ROOT / 'data' / 'frame_features_v6.jsonl'
report_path = ROOT / 'reports' / f'feature_extraction_upgrade_v6_{datetime.now().strftime("%Y-%m-%dT%H-%M-%S")}.json'

rows = [json.loads(x) for x in in_path.read_text().splitlines() if x.strip()]

# v6: shuttle gap fill with bounded velocity extrapolation + mild EMA smoothing
max_shuttle_gap = 5
shuttle_alpha = 0.6
max_step = 0.08  # clamp per-frame extrapolation for stability

filled_shuttle = 0
smoothed_shuttle = 0

last_shuttle = None
prev_shuttle = None
last_seen = -10_000


def clamp01(v):
    return min(max(v, 0.0), 1.0)


for i, r in enumerate(rows):
    sh = r.get('shuttle', {}) or {}
    visible = bool(sh.get('visible', False))

    if visible and sh.get('x') is not None and sh.get('y') is not None:
        cur_x = float(sh['x'])
        cur_y = float(sh['y'])

        if last_shuttle is not None:
            # EMA smoothing on detected position
            sm_x = (1 - shuttle_alpha) * last_shuttle['x'] + shuttle_alpha * cur_x
            sm_y = (1 - shuttle_alpha) * last_shuttle['y'] + shuttle_alpha * cur_y
            sh['x'] = clamp01(sm_x)
            sh['y'] = clamp01(sm_y)
            sh['smoothed_v6'] = True
            smoothed_shuttle += 1

        prev_shuttle = deepcopy(last_shuttle)
        last_shuttle = {'x': float(sh['x']), 'y': float(sh['y'])}
        last_seen = i

    else:
        # Fill short shuttle gaps using velocity extrapolation from last two visible points
        if last_shuttle is not None and (i - last_seen) <= max_shuttle_gap:
            fx, fy = last_shuttle['x'], last_shuttle['y']
            if prev_shuttle is not None:
                vx = last_shuttle['x'] - prev_shuttle['x']
                vy = last_shuttle['y'] - prev_shuttle['y']
                vx = min(max(vx, -max_step), max_step)
                vy = min(max(vy, -max_step), max_step)
                fx = clamp01(last_shuttle['x'] + vx)
                fy = clamp01(last_shuttle['y'] + vy)

            r['shuttle'] = {
                'x': fx,
                'y': fy,
                'visible': True,
                'source': 'interp',
                'filled_v6': True,
            }
            prev_shuttle = deepcopy(last_shuttle)
            last_shuttle = {'x': fx, 'y': fy}
            filled_shuttle += 1

N = len(rows)
x_detect = sum(1 for r in rows if r.get('players', {}).get('X') is not None)
y_detect = sum(1 for r in rows if r.get('players', {}).get('Y') is not None)
ball_vis = sum(1 for r in rows if r.get('shuttle', {}).get('visible', False))
ball_interp = sum(1 for r in rows if r.get('shuttle', {}).get('source') == 'interp')

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
    'shuttle_visible_rate': round(ball_vis / N, 4) if N else 0,
    'shuttle_interp_rate': round(ball_interp / N, 4) if N else 0,
    'feature_quality_score_0_100': round(score, 2),
    'changes': {
        'shuttle_filled_frames_v6': filled_shuttle,
        'shuttle_smoothed_frames_v6': smoothed_shuttle,
        'max_shuttle_gap': max_shuttle_gap,
        'shuttle_ema_alpha': shuttle_alpha,
        'max_step': max_step,
    },
}

report_path.write_text(json.dumps(report, indent=2))
print('saved', out_path)
print('saved', report_path)
print(json.dumps(report, indent=2))
