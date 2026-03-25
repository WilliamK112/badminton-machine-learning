import json
from pathlib import Path
from copy import deepcopy
from datetime import datetime

ROOT = Path(__file__).resolve().parents[1]
in_path = ROOT / 'data' / 'frame_features_v3.jsonl'
out_path = ROOT / 'data' / 'frame_features_v4.jsonl'
report_path = ROOT / 'reports' / f'feature_extraction_upgrade_v4_{datetime.now().strftime("%Y-%m-%dT%H-%M-%S")}.json'

rows = [json.loads(x) for x in in_path.read_text().splitlines() if x.strip()]

# temporal fill settings (slightly more aggressive than v3)
max_player_gap = {'X': 3, 'Y': 5}
max_shuttle_gap = 3
alpha = 0.45  # shuttle EMA smoothing

filled_players = {'X': 0, 'Y': 0}
smoothed_shuttle = 0
interpolated_shuttle = 0

last_player = {'X': None, 'Y': None}
last_player_seen = {'X': -10_000, 'Y': -10_000}

last_shuttle_xy = None
last_shuttle_v = [0.0, 0.0]
last_shuttle_seen = -10_000

for i, r in enumerate(rows):
    players = r.get('players', {})

    # 1) fill short player misses with last known state
    for side in ['X', 'Y']:
        if players.get(side) is None:
            if last_player[side] is not None and (i - last_player_seen[side]) <= max_player_gap[side]:
                players[side] = deepcopy(last_player[side])
                players[side]['filled_v4'] = True
                filled_players[side] += 1
        else:
            last_player[side] = deepcopy(players[side])
            last_player_seen[side] = i

    # 2) shuttle smoothing + short-gap interpolation
    shuttle = r.get('shuttle', {})
    xy = shuttle.get('xy')
    visible = bool(shuttle.get('visible', False) and xy is not None)

    if visible:
        if last_shuttle_xy is not None:
            # EMA smooth to reduce jitter
            sx = (1 - alpha) * last_shuttle_xy[0] + alpha * xy[0]
            sy = (1 - alpha) * last_shuttle_xy[1] + alpha * xy[1]
            if abs(sx - xy[0]) + abs(sy - xy[1]) > 1e-8:
                smoothed_shuttle += 1
            xy = [sx, sy]

            vx = xy[0] - last_shuttle_xy[0]
            vy = xy[1] - last_shuttle_xy[1]
            shuttle['v'] = [vx, vy]
            shuttle['speed'] = float((vx * vx + vy * vy) ** 0.5)
            last_shuttle_v = [vx, vy]

        shuttle['xy'] = xy
        shuttle['visible'] = True
        last_shuttle_xy = xy
        last_shuttle_seen = i

    else:
        # short-gap interpolation from last velocity
        if last_shuttle_xy is not None and (i - last_shuttle_seen) <= max_shuttle_gap:
            pred = [last_shuttle_xy[0] + last_shuttle_v[0], last_shuttle_xy[1] + last_shuttle_v[1]]
            pred = [min(max(pred[0], 0.0), 1.0), min(max(pred[1], 0.0), 1.0)]
            shuttle['xy'] = pred
            shuttle['v'] = list(last_shuttle_v)
            shuttle['speed'] = float((last_shuttle_v[0] ** 2 + last_shuttle_v[1] ** 2) ** 0.5)
            shuttle['visible'] = True
            shuttle['source'] = 'interp_v4'
            interpolated_shuttle += 1
            last_shuttle_xy = pred
            last_shuttle_seen = i

    r['players'] = players
    r['shuttle'] = shuttle

with out_path.open('w') as f:
    for r in rows:
        f.write(json.dumps(r) + '\n')

N = len(rows)
x_detect = sum(1 for r in rows if r.get('players', {}).get('X') is not None)
y_detect = sum(1 for r in rows if r.get('players', {}).get('Y') is not None)
ball_vis = sum(1 for r in rows if r.get('shuttle', {}).get('visible', False))

score = (
    0.35 * (x_detect / N if N else 0)
    + 0.35 * (y_detect / N if N else 0)
    + 0.30 * (ball_vis / N if N else 0)
) * 100

report = {
    'input': str(in_path),
    'output': str(out_path),
    'frames': N,
    'player_X_detect_rate': round(x_detect / N, 4) if N else 0,
    'player_Y_detect_rate': round(y_detect / N, 4) if N else 0,
    'shuttle_visible_rate': round(ball_vis / N, 4) if N else 0,
    'feature_quality_score_0_100': round(score, 2),
    'changes': {
        'player_X_filled_frames_v4': filled_players['X'],
        'player_Y_filled_frames_v4': filled_players['Y'],
        'shuttle_smoothed_frames_v4': smoothed_shuttle,
        'shuttle_interpolated_frames_v4': interpolated_shuttle,
        'max_player_gap': max_player_gap,
        'max_shuttle_gap': max_shuttle_gap,
        'ema_alpha': alpha,
    },
}

report_path.write_text(json.dumps(report, indent=2))
print('saved', out_path)
print('saved', report_path)
print(json.dumps(report, indent=2))
