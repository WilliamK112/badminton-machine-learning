import json
from pathlib import Path
from copy import deepcopy
from datetime import datetime

ROOT = Path(__file__).resolve().parents[1]
in_path = ROOT / 'data' / 'frame_features_v4.jsonl'
out_path = ROOT / 'data' / 'frame_features_v5.jsonl'
report_path = ROOT / 'reports' / f'feature_extraction_upgrade_v5_{datetime.now().strftime("%Y-%m-%dT%H-%M-%S")}.json'

rows = [json.loads(x) for x in in_path.read_text().splitlines() if x.strip()]

# v5: velocity-aware fill for players + keypoint EMA smoothing
max_player_gap = {'X': 4, 'Y': 7}
player_alpha = 0.55

filled_players = {'X': 0, 'Y': 0}
smoothed_players = {'X': 0, 'Y': 0}

last_player = {'X': None, 'Y': None}
prev_player = {'X': None, 'Y': None}
last_seen = {'X': -10_000, 'Y': -10_000}


def clamp01(v):
    return min(max(v, 0.0), 1.0)


def smooth_kpts(prev_kpts, curr_kpts, alpha):
    if not prev_kpts or not curr_kpts or len(prev_kpts) != len(curr_kpts):
        return curr_kpts
    out = []
    for p, c in zip(prev_kpts, curr_kpts):
        if p is None or c is None or len(p) < 2 or len(c) < 2:
            out.append(c)
            continue
        sx = (1 - alpha) * p[0] + alpha * c[0]
        sy = (1 - alpha) * p[1] + alpha * c[1]
        conf = c[2] if len(c) > 2 else 1.0
        out.append([clamp01(sx), clamp01(sy), conf])
    return out


for i, r in enumerate(rows):
    players = r.get('players', {})

    for side in ['X', 'Y']:
        cur = players.get(side)

        if cur is None:
            if last_player[side] is not None and (i - last_seen[side]) <= max_player_gap[side]:
                filled = deepcopy(last_player[side])

                # velocity-aware extrapolation from previous frame pair
                if prev_player[side] is not None:
                    c1 = prev_player[side].get('center')
                    c2 = last_player[side].get('center')
                    if c1 and c2:
                        vx = c2[0] - c1[0]
                        vy = c2[1] - c1[1]
                        new_center = [clamp01(c2[0] + vx), clamp01(c2[1] + vy)]
                        dx = new_center[0] - c2[0]
                        dy = new_center[1] - c2[1]
                        filled['center'] = new_center
                        if filled.get('bbox'):
                            x1, y1, x2, y2 = filled['bbox']
                            filled['bbox'] = [clamp01(x1 + dx), clamp01(y1 + dy), clamp01(x2 + dx), clamp01(y2 + dy)]
                        if filled.get('kpts'):
                            shifted = []
                            for k in filled['kpts']:
                                if k and len(k) >= 2:
                                    conf = k[2] if len(k) > 2 else 1.0
                                    shifted.append([clamp01(k[0] + dx), clamp01(k[1] + dy), conf])
                                else:
                                    shifted.append(k)
                            filled['kpts'] = shifted

                filled['filled_v5'] = True
                players[side] = filled
                filled_players[side] += 1

        else:
            # smooth noisy keypoints for detected players
            if last_player[side] is not None and cur.get('kpts') and last_player[side].get('kpts'):
                cur['kpts'] = smooth_kpts(last_player[side]['kpts'], cur['kpts'], player_alpha)
                smoothed_players[side] += 1

            prev_player[side] = deepcopy(last_player[side])
            last_player[side] = deepcopy(cur)
            last_seen[side] = i

    r['players'] = players

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
        'player_X_filled_frames_v5': filled_players['X'],
        'player_Y_filled_frames_v5': filled_players['Y'],
        'player_X_smoothed_frames_v5': smoothed_players['X'],
        'player_Y_smoothed_frames_v5': smoothed_players['Y'],
        'max_player_gap': max_player_gap,
        'player_kpt_ema_alpha': player_alpha,
    },
}

report_path.write_text(json.dumps(report, indent=2))
print('saved', out_path)
print('saved', report_path)
print(json.dumps(report, indent=2))
