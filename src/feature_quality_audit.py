import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
feat = ROOT / 'data' / 'frame_features.jsonl'
out = ROOT / 'reports' / 'feature_quality_audit.json'

rows = [json.loads(x) for x in feat.read_text().splitlines() if x.strip()]
N = len(rows)

x_detect = sum(1 for r in rows if r.get('players',{}).get('X') is not None)
y_detect = sum(1 for r in rows if r.get('players',{}).get('Y') is not None)
ball_vis = sum(1 for r in rows if r.get('shuttle',{}).get('visible', False))

# simple quality score (0-100)
if N == 0:
    score = 0
else:
    score = (
        0.4 * (x_detect / N) +
        0.4 * (y_detect / N) +
        0.2 * (ball_vis / N)
    ) * 100

report = {
    'frames': N,
    'player_X_detect_rate': round(x_detect / N, 4) if N else 0,
    'player_Y_detect_rate': round(y_detect / N, 4) if N else 0,
    'shuttle_visible_rate': round(ball_vis / N, 4) if N else 0,
    'feature_quality_score_0_100': round(score, 2),
    'next_tuning': {
        'sample_every': 'try 4 instead of 6 if shuttle_visible_rate < 0.55',
        'video_resolution': 'prefer 1080p when possible',
        'camera_angle': 'full-court steady top view'
    }
}

out.write_text(json.dumps(report, indent=2))
print('saved', out)
print(report)
