import json
from pathlib import Path
from datetime import datetime

ROOT = Path(__file__).resolve().parents[1]
R = ROOT / 'reports'


def load_json(name):
    p = R / name
    return json.loads(p.read_text()) if p.exists() else None


def load_text(name):
    p = R / name
    return p.read_text() if p.exists() else ''


# Prefer newest compressed-first era metrics.
metrics_map = [
    ('quant_v5', 'quant_model_metrics_v5.json'),
    ('quant_v4', 'quant_model_metrics_v4.json'),
    ('rally_v4', 'rally_metrics_v4_compare.json'),
    ('temporal', 'temporal_model_metrics.json'),
    # legacy fallback
    ('quant_v3', 'quant_model_metrics_v3.json'),
]

loaded = []
for label, filename in metrics_map:
    m = load_json(filename)
    if m:
        loaded.append((label, filename, m))

summary = load_json('report_summary.json') or {}
changelog = load_text('model_changelog.md')
infer = load_text('inference_recommendation.md')

candidates = [row for row in loaded if row[2].get('landing_rmse') is not None]
best = sorted(candidates, key=lambda x: x[2]['landing_rmse'])[0][0] if candidates else 'n/a'

html = f"""<!doctype html><html><head><meta charset='utf-8'><title>Badminton AI Full Report</title>
<style>
body{{font-family:Inter,Arial,sans-serif;max-width:1150px;margin:24px auto;padding:0 12px;line-height:1.45}}
img{{max-width:100%;border:1px solid #ddd;border-radius:8px}}
.card{{border:1px solid #e5e7eb;border-radius:12px;padding:14px;margin:12px 0}}
table{{border-collapse:collapse;width:100%}}th,td{{border:1px solid #ddd;padding:8px;text-align:left}}th{{background:#f9fafb}}
.best{{background:#ecfeff}}
pre{{white-space:pre-wrap;background:#f8fafc;border:1px solid #e5e7eb;border-radius:10px;padding:12px}}
</style></head><body>
<h1>🏸 Badminton AI Full Report</h1>
<div class='card'>
<p><b>Best model by landing RMSE:</b> {best}</p>
<p><b>Frames analyzed:</b> {summary.get('frames_analyzed')}</p>
<p><b>Current ratings:</b> {summary.get('ratings')}</p>
</div>

<h2>Model Metrics Comparison</h2>
<table>
<tr><th>Model</th><th>Source</th><th>Winner ACC</th><th>Landing RMSE</th><th>Samples</th><th>Note</th></tr>
"""

for name, filename, m in loaded:
    cls = 'best' if name == best else ''
    html += (
        f"<tr class='{cls}'><td>{name}</td><td>{filename}</td>"
        f"<td>{m.get('winner_acc')}</td><td>{m.get('landing_rmse')}</td>"
        f"<td>{m.get('samples')}</td><td>{m.get('note','')}</td></tr>"
    )

html += """
</table>

<h2>Dynamic Win Probability</h2>
<img src='win_prob_timeline_v3.png' alt='win probability timeline v3'/>

<h2>Landing Heatmap</h2>
<img src='landing_heatmap_v3.png' alt='landing heatmap v3'/>

<h2>Model Changelog</h2>
<pre>
""" + changelog + """
</pre>

<h2>Inference Recommendation</h2>
<pre>
""" + infer + """
</pre>

</body></html>
"""

index_out = R / 'index.html'
index_out.write_text(html)

stamp = datetime.now().astimezone().strftime('%Y-%m-%dT%H-%M-%S%z')
meta = {
    'created_at': stamp,
    'entrypoint': 'update_report_index_v3.py',
    'best_model': best,
    'models_included': [
        {
            'name': n,
            'source': f,
            'winner_acc': m.get('winner_acc'),
            'landing_rmse': m.get('landing_rmse'),
            'samples': m.get('samples'),
        }
        for n, f, m in loaded
    ],
    'artifacts': {
        'index_html': str(index_out.relative_to(ROOT)),
        'win_prob_plot': 'reports/win_prob_timeline_v3.png',
        'landing_heatmap': 'reports/landing_heatmap_v3.png',
    },
}
meta_out = R / f'report_index_v3_{stamp}.json'
meta_out.write_text(json.dumps(meta, indent=2))

print('saved', index_out)
print('saved', meta_out)
print('best_model', best)
