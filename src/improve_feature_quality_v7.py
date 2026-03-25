import json
from pathlib import Path
from copy import deepcopy
from datetime import datetime

ROOT = Path(__file__).resolve().parents[1]
in_path = ROOT / 'data' / 'frame_features_v6.jsonl'
out_path = ROOT / 'data' / 'frame_features_v7.jsonl'
report_path = ROOT / 'reports' / f'feature_extraction_upgrade_v7_{datetime.now().strftime("%Y-%m-%dT%H-%M-%S")}.json'
audit_path = ROOT / 'reports' / 'feature_quality_audit_v7.json'

rows = [json.loads(x) for x in in_path.read_text().splitlines() if x.strip()]

max_player_gap = 4
max_center_step = 0.06

filled_players = {'X': 0, 'Y': 0}


def clamp01(v):
    return min(max(v, 0.0), 1.0)


def center_from_bbox(b):
    if not b:
        return None
    x1, y1, x2, y2 = b
    return ((x1 + x2) / 2.0, (y1 + y2) / 2.0)


def shift_bbox(b, dx, dy):
    x1, y1, x2, y2 = b
    return [clamp01(x1 + dx), clamp01(y1 + dy), clamp01(x2 + dx), clamp01(y2 + dy)]


state = {
    'X': {'last': None, 'prev': None, 'last_seen': -10_000},
    'Y': {'last': None, 'prev': None, 'last_seen': -10_000},
}

for i, r in enumerate(rows):
    players = r.get('players', {}) or {}

    for pid in ('X', 'Y'):
        p = players.get(pid)
        has_bbox = bool(p and p.get('bbox'))

        st = state[pid]

        if has_bbox:
            c = center_from_bbox(p['bbox'])
            st['prev'] = deepcopy(st['last'])
            st['last'] = {'bbox': deepcopy(p['bbox']), 'center': c}
            st['last_seen'] = i
            continue

        gap = i - st['last_seen']
        if st['last'] is None or gap > max_player_gap:
            continue

        dx, dy = 0.0, 0.0
        if st['prev'] is not None and st['prev'].get('center') and st['last'].get('center'):
            vx = st['last']['center'][0] - st['prev']['center'][0]
            vy = st['last']['center'][1] - st['prev']['center'][1]
            dx = min(max(vx, -max_center_step), max_center_step)
            dy = min(max(vy, -max_center_step), max_center_step)

        fb = shift_bbox(st['last']['bbox'], dx, dy)

        if 'players' not in r or not isinstance(r['players'], dict):
            r['players'] = {}

        existing = r['players'].get(pid) or {}
        r['players'][pid] = {
            'bbox': fb,
            'kpts': existing.get('kpts', []),
            'source': 'interp',
            'filled_v7': True,
        }

        st['prev'] = deepcopy(st['last'])
        st['last'] = {'bbox': deepcopy(fb), 'center': center_from_bbox(fb)}
        filled_players[pid] += 1

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
    'shuttle_visible_rate': round(ball_vis / N, 4) if N else 0,
    'feature_quality_score_0_100': round(score, 2),
    'changes': {
        'player_interp_filled_frames_v7': filled_players,
        'max_player_gap': max_player_gap,
        'max_center_step': max_center_step,
    },
}

report_path.write_text(json.dumps(report, indent=2))
audit_path.write_text(json.dumps(report, indent=2))
print('saved', out_path)
print('saved', report_path)
print('saved', audit_path)
print(json.dumps(report, indent=2))
