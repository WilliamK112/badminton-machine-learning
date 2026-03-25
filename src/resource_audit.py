import json
import os
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
reports = ROOT / 'reports'
reports.mkdir(exist_ok=True)

start = time.time()

# file sizes
targets = [
    ROOT / 'data' / 'frame_features_v2.jsonl',
    ROOT / 'data' / 'quant_features_v2.csv',
    ROOT / 'data' / 'quant_features_v3.csv',
    ROOT / 'reports' / 'index.html',
    ROOT / 'reports' / 'win_prob_curve.png',
    ROOT / 'reports' / 'shuttle_heatmap.png',
    ROOT / 'reports' / 'shuttle_heatmap_denoised.png',
]

size_map = {}
for p in targets:
    if p.exists():
        size_map[str(p.relative_to(ROOT))] = p.stat().st_size

# total project footprint
project_bytes = 0
for root, _, files in os.walk(ROOT):
    for f in files:
        fp = Path(root) / f
        try:
            project_bytes += fp.stat().st_size
        except FileNotFoundError:
            pass

elapsed = time.time() - start

out = {
    'project_total_bytes': project_bytes,
    'project_total_mb': round(project_bytes / (1024*1024), 2),
    'tracked_files_bytes': size_map,
    'audit_runtime_sec': round(elapsed, 4),
    'recommendations': [
        'Keep only v3 artifacts for daily use; archive old intermediates.',
        'Prefer JSONL/CSV features over raw frame dumps.',
        'Use 720p + sample_every=4 for quality/cost balance.'
    ]
}

out_path = reports / 'resource_audit.json'
out_path.write_text(json.dumps(out, indent=2))
print('saved', out_path)
print(out)
