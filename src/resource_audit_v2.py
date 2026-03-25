import json
import os
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
reports = ROOT / 'reports'
reports.mkdir(exist_ok=True)

start = time.time()

# Key artifacts to track (latest pipeline)
targets = [
    ROOT / 'data' / 'frame_features_v6.jsonl',
    ROOT / 'data' / 'quant_features_v5.csv',
    ROOT / 'reports' / 'index.html',
    ROOT / 'reports' / 'quant_model_metrics_v4_compare.json',
    ROOT / 'reports' / 'win_prob_curve.png',
    ROOT / 'reports' / 'shuttle_heatmap.png',
    ROOT / 'reports' / 'shuttle_heatmap_denoised.png',
]

size_map = {}
for p in targets:
    if p.exists():
        size_map[str(p.relative_to(ROOT))] = p.stat().st_size

# total project footprint + top heavy files
total_bytes = 0
files = []
for root, _, fs in os.walk(ROOT):
    for f in fs:
        fp = Path(root) / f
        try:
            sz = fp.stat().st_size
            total_bytes += sz
            files.append((sz, fp))
        except FileNotFoundError:
            pass

files.sort(reverse=True, key=lambda x: x[0])
top_heavy = [
    {
        'path': str(fp.relative_to(ROOT)),
        'bytes': sz,
        'mb': round(sz / (1024 * 1024), 2),
    }
    for sz, fp in files[:12]
]

safe_delete_candidates = []
for sz, fp in files:
    rel = str(fp.relative_to(ROOT))
    if rel in {
        'reports/index.html',
        'reports/quant_model_metrics_v3.json',
        'reports/quant_model_metrics_v4_compare.json',
        'reports/temporal_model_metrics.json',
        'reports/win_prob_curve.png',
        'reports/shuttle_heatmap_denoised.png',
        'data/frame_features_v6.jsonl',
        'data/quant_features_v5.csv',
    }:
        continue
    if rel.endswith('.pt') or rel.startswith('data/frame_features_v') and not rel.endswith('v6.jsonl'):
        safe_delete_candidates.append({'path': rel, 'bytes': sz, 'mb': round(sz / (1024 * 1024), 2)})

reclaimable_bytes = sum(x['bytes'] for x in safe_delete_candidates)

elapsed = time.time() - start

out = {
    'project_total_bytes': total_bytes,
    'project_total_mb': round(total_bytes / (1024 * 1024), 2),
    'tracked_files_bytes': size_map,
    'top_heavy_files': top_heavy,
    'safe_delete_candidates': safe_delete_candidates,
    'reclaimable_bytes': reclaimable_bytes,
    'reclaimable_mb': round(reclaimable_bytes / (1024 * 1024), 2),
    'audit_runtime_sec': round(elapsed, 4),
    'recommendations': [
        'Keep latest artifacts only: frame_features_v6 + quant_features_v5 + report core files.',
        'If disk is tight, remove YOLO .pt cache files; they are regenerable/downloadable.',
        'Delete older frame_features_v*.jsonl snapshots except v6 to reduce storage.',
    ]
}

out_path = reports / 'resource_audit_v2.json'
out_path.write_text(json.dumps(out, indent=2))
print('saved', out_path)
print(json.dumps({'project_total_mb': out['project_total_mb'], 'reclaimable_mb': out['reclaimable_mb']}, indent=2))
