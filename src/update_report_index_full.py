import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
R = ROOT / 'reports'

def load_json(name):
    p = R / name
    return json.loads(p.read_text()) if p.exists() else None

def load_text(name):
    p = R / name
    return p.read_text() if p.exists() else ''

v1 = load_json('quant_model_metrics.json')
v2 = load_json('quant_model_metrics_v2.json')
v3 = load_json('quant_model_metrics_v3.json')
tm = load_json('temporal_model_metrics.json')
summary = load_json('report_summary.json') or {}
changelog = load_text('model_changelog.md')
infer = load_text('inference_recommendation.md')

rows = []
for name, m in [('v1', v1), ('v2', v2), ('v3', v3), ('temporal', tm)]:
    if m:
        rows.append((name, m.get('winner_acc'), m.get('landing_rmse'), m.get('samples')))

best = sorted([r for r in rows if r[2] is not None], key=lambda x: x[2])[0][0] if rows else 'n/a'

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
<tr><th>Model</th><th>Winner ACC</th><th>Landing RMSE</th><th>Samples</th><th>Note</th></tr>
"""

for name, m in [('v1', v1), ('v2', v2), ('v3', v3), ('temporal', tm)]:
    if not m:
        continue
    cls = "best" if name == best else ""
    html += f"<tr class='{cls}'><td>{name}</td><td>{m.get('winner_acc')}</td><td>{m.get('landing_rmse')}</td><td>{m.get('samples')}</td><td>{m.get('note','')}</td></tr>"

html += """
</table>

<h2>Dynamic Win Probability</h2>
<img src='win_prob_curve.png' alt='win prob curve'/>

<h2>Shuttle Heatmap</h2>
<img src='shuttle_heatmap.png' alt='shuttle heatmap'/>

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

out = R / 'index.html'
out.write_text(html)
print('saved', out)
