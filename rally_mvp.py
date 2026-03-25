import json
from pathlib import Path

p = Path(__file__).with_name('pose_summary.json')
if not p.exists():
    raise SystemExit('pose_summary.json not found. Run quickstart.py first.')

data = json.loads(p.read_text())
samples = data.get('samples', [])

# MVP heuristic:
# - Assume 2 players should exist; confidence increases when 2 detected.
# - Win probability proxy = normalized stability score per frame.
# - Player ratings initialized to 50/50 and updated by pseudo-rally evidence.

rating_a = 50.0
rating_b = 50.0
timeline = []

for s in samples:
    people = s.get('people_detected', 0)
    frame = s.get('frame')

    # simple confidence heuristic
    if people >= 2:
        p_a = min(0.7, 0.5 + 0.02 * (people - 1))
    elif people == 1:
        p_a = 0.5
    else:
        p_a = 0.45
    p_b = 1 - p_a

    # tiny rating update (MVP only)
    rating_a += (p_a - 0.5) * 0.2
    rating_b += (p_b - 0.5) * 0.2

    timeline.append({
        'frame': frame,
        'win_prob_a': round(p_a, 3),
        'win_prob_b': round(p_b, 3),
        'players_detected': people,
    })

# clamp
rating_a = max(0, min(100, rating_a))
rating_b = max(0, min(100, rating_b))

out = {
    'mvp_note': 'Heuristic-only baseline. Replace with shuttle/racket + motion features next.',
    'player_rating': {
        'player_a': round(rating_a, 2),
        'player_b': round(rating_b, 2)
    },
    'timeline': timeline[:300]
}

out_path = Path(__file__).with_name('rally_mvp_output.json')
out_path.write_text(json.dumps(out, indent=2))
print('Saved', out_path)
print('Ratings:', out['player_rating'])
