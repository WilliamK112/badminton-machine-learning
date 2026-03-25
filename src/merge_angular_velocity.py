"""
Merge angular velocity from enhanced_features into frame_features_v10
"""
import json
from pathlib import Path

# Load enhanced features (has angular velocity)
enhanced = json.loads(Path('enhanced_features_output.json').read_text())
frame_to_ang = {}
for entry in enhanced['timeline']:
    if entry.get('angular_vel'):
        frame_to_ang[entry['frame']] = entry['angular_vel']

print(f"Merging {len(frame_to_ang)} angular velocity samples into frame_features_v10")

# Read and merge
input_path = Path('data/frame_features_v10.jsonl')
output_path = Path('data/frame_features_v11.jsonl')

updated_count = 0
with open(output_path, 'w') as out:
    for line in input_path.read_text().splitlines():
        if not line.strip():
            continue
        r = json.loads(line)
        frame = r['frame']
        if frame in frame_to_ang:
            r['angular_vel'] = frame_to_ang[frame]
            updated_count += 1
        out.write(json.dumps(r) + '\n')

print(f"Updated {updated_count} frames with angular velocity")
print(f"Saved to {output_path}")

# Quick audit
rows = [json.loads(x) for x in output_path.read_text().splitlines() if x.strip()]
N = len(rows)
ang_vel = sum(1 for r in rows if r.get('angular_vel'))
print(f"Angular velocity coverage: {ang_vel}/{N} = {ang_vel/N*100:.1f}%")

# Compress
import gzip
with open(output_path, 'rb') as f_in:
    with gzip.open(str(output_path) + '.gz', 'wb') as f_out:
        f_out.write(f_in.read())
print(f"Compressed to {output_path}.gz")
