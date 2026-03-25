import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
R = ROOT / 'reports'

# load metrics safely

def load(name):
    p = R / name
    return json.loads(p.read_text()) if p.exists() else None

v1 = load('quant_model_metrics.json')
v2 = load('quant_model_metrics_v2.json')
v3 = load('quant_model_metrics_v3.json')
tm = load('temporal_model_metrics.json')
summary = load('report_summary.json') or {}

rows = []
for name, m in [('v1', v1), ('v2', v2), ('v3', v3), ('temporal', tm)]:
    if m:
        rows.append((name, m.get('winner_acc'), m.get('landing_rmse'), m.get('samples')))

# best by rmse (lower better)
best = None
if rows:
    best = sorted([r for r in rows if r[2] is not None], key=lambda x: x[2])[0][0]

html = f"""<!doctype html>
<html><head><meta charset='utf-8'><title>Badminton AI Report</title>
<style>
body{{font-family:Inter,Arial,sans-serif;max-width:1100px;margin:24px auto;padding:0 12px}}
img{{max-width:100%;border:1px solid #ddd;border-radius:8px}}
.card{{border:1px solid #e5e7eb;border-radius:12px;padding:14px;margin:12px 0}}
table{{border-collapse:collapse;width:100%}}th,td{{border:1px solid #ddd;padding:8px;text-align:left}}th{{background:#f9fafb}}
.best{{background:#ecfeff}}
</style></head><body>
<h1>🏸 Badminton AI Report (Autopilot)</h1>
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

</body></html>
"""

out = R / 'index.html'
out.write_text(html)
print('saved', out)
