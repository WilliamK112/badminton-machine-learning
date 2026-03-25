import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
feat = ROOT / 'data' / 'frame_features_v2.jsonl'
out = ROOT / 'reports' / 'feature_quality_audit_v2.json'

rows = [json.loads(x) for x in feat.read_text().splitlines() if x.strip()]
N = len(rows)

x_detect = sum(1 for r in rows if r.get('players',{}).get('X') is not None)
y_detect = sum(1 for r in rows if r.get('players',{}).get('Y') is not None)
ball_vis = sum(1 for r in rows if r.get('shuttle',{}).get('visible', False))
ball_det = sum(1 for r in rows if r.get('shuttle',{}).get('source') == 'det')
ball_motion = sum(1 for r in rows if r.get('shuttle',{}).get('source') == 'motion')

score = (
    0.35 * (x_detect / N if N else 0) +
    0.35 * (y_detect / N if N else 0) +
    0.30 * (ball_vis / N if N else 0)
) * 100

report = {
    'frames': N,
    'player_X_detect_rate': round(x_detect / N, 4) if N else 0,
    'player_Y_detect_rate': round(y_detect / N, 4) if N else 0,
    'shuttle_visible_rate': round(ball_vis / N, 4) if N else 0,
    'shuttle_source_breakdown': {
        'det_rate': round(ball_det / N, 4) if N else 0,
        'motion_rate': round(ball_motion / N, 4) if N else 0,
    },
    'feature_quality_score_0_100': round(score, 2)
}

out.write_text(json.dumps(report, indent=2))
print('saved', out)
print(report)
