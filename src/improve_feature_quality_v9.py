import json
from pathlib import Path
from datetime import datetime

ROOT = Path(__file__).resolve().parents[1]
in_path = ROOT / 'data' / 'frame_features_v6.jsonl'
out_path = ROOT / 'data' / 'frame_features_v9.jsonl'
report_path = ROOT / 'reports' / f'feature_extraction_upgrade_v9_{datetime.now().strftime("%Y-%m-%dT%H-%M-%S")}.json'
audit_path = ROOT / 'reports' / 'feature_quality_audit_v9.json'

rows = [json.loads(x) for x in in_path.read_text().splitlines() if x.strip()]
N = len(rows)

# Collect shuttle track
track = []
raw_visible = 0
for i, r in enumerate(rows):
    sh = r.get('shuttle') or {}
    vis = bool(sh.get('visible', False) and sh.get('xy') is not None)
    if vis:
        xy = [float(sh['xy'][0]), float(sh['xy'][1])]
        track.append((i, xy))
        raw_visible += 1

max_interp_gap = 6
interp_filled = 0
smoothed_points = 0

# 1) Bridge short missing gaps using linear interpolation between two real visible points.
for k in range(len(track) - 1):
    i0, p0 = track[k]
    i1, p1 = track[k + 1]
    gap = i1 - i0 - 1
    if gap <= 0 or gap > max_interp_gap:
        continue

    dx = p1[0] - p0[0]
    dy = p1[1] - p0[1]
    for t in range(1, gap + 1):
        j = i0 + t
        a = t / (gap + 1)
        px = p0[0] + a * dx
        py = p0[1] + a * dy

        sh = rows[j].get('shuttle') or {}
        vis = bool(sh.get('visible', False) and sh.get('xy') is not None)
        if not vis:
            sh['xy'] = [px, py]
            sh['visible'] = True
            sh['source'] = 'interp_linear_v9'
            sh['filled_v9'] = True
            rows[j]['shuttle'] = sh
            interp_filled += 1

# 2) Smooth post-filled trajectory with a 3-point weighted moving average.
for i in range(1, N - 1):
    sh_prev = rows[i - 1].get('shuttle') or {}
    sh_cur = rows[i].get('shuttle') or {}
    sh_next = rows[i + 1].get('shuttle') or {}

    ok_prev = bool(sh_prev.get('visible', False) and sh_prev.get('xy') is not None)
    ok_cur = bool(sh_cur.get('visible', False) and sh_cur.get('xy') is not None)
    ok_next = bool(sh_next.get('visible', False) and sh_next.get('xy') is not None)
    if not (ok_prev and ok_cur and ok_next):
        continue

    x = 0.2 * float(sh_prev['xy'][0]) + 0.6 * float(sh_cur['xy'][0]) + 0.2 * float(sh_next['xy'][0])
    y = 0.2 * float(sh_prev['xy'][1]) + 0.6 * float(sh_cur['xy'][1]) + 0.2 * float(sh_next['xy'][1])
    sh_cur['xy_raw_v9'] = [float(sh_cur['xy'][0]), float(sh_cur['xy'][1])]
    sh_cur['xy'] = [x, y]
    sh_cur['smoothed_v9'] = True
    rows[i]['shuttle'] = sh_cur
    smoothed_points += 1

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
    'shuttle_visible_rate_post_v9': round(ball_vis / N, 4) if N else 0,
    'feature_quality_score_0_100': round(score, 2),
    'changes': {
        'interp_filled_frames_v9': interp_filled,
        'smoothed_points_v9': smoothed_points,
        'max_interp_gap': max_interp_gap,
    },
}

report_path.write_text(json.dumps(report, indent=2))
audit_path.write_text(json.dumps(report, indent=2))
print('saved', out_path)
print('saved', report_path)
print('saved', audit_path)
print(json.dumps(report, indent=2))
