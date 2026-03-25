import json
from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parents[1]
feat = ROOT / 'data' / 'frame_features_v2.jsonl'
out_img = ROOT / 'reports' / 'shuttle_heatmap_denoised.png'
out_json = ROOT / 'reports' / 'heatmap_denoise_stats.json'

pts = []
for line in feat.read_text().splitlines():
    r = json.loads(line)
    sh = r.get('shuttle', {})
    xy = sh.get('xy')
    sp = float(sh.get('speed', 0) or 0)
    vis = bool(sh.get('visible', False))
    if not vis or xy is None:
        continue
    x, y = xy
    # basic bounds
    if not (0 <= x <= 1 and 0 <= y <= 1):
        continue
    # denoise: reject near-static and hyper-jump artifacts
    if sp < 0.001 or sp > 0.25:
        continue
    pts.append((x, y, sp))

if not pts:
    raise SystemExit('No points after denoise')

arr = np.array(pts)
xy = arr[:, :2]

# robust clip by quantiles to remove spatial outliers
qx1, qx2 = np.quantile(xy[:,0], [0.01, 0.99])
qy1, qy2 = np.quantile(xy[:,1], [0.01, 0.99])
mask = (xy[:,0] >= qx1) & (xy[:,0] <= qx2) & (xy[:,1] >= qy1) & (xy[:,1] <= qy2)
xy2 = xy[mask]

plt.figure(figsize=(5,9))
plt.hist2d(xy2[:,0], xy2[:,1], bins=50, cmap='viridis')
plt.colorbar(label='Density')
plt.gca().invert_yaxis()
plt.xlabel('Court X')
plt.ylabel('Court Y')
plt.title('Shuttle Position Heatmap (Denoised)')
plt.tight_layout()
plt.savefig(out_img, dpi=150)
plt.close()

stats = {
    'raw_points': int(len(pts)),
    'after_spatial_clip': int(len(xy2)),
    'x_clip': [float(qx1), float(qx2)],
    'y_clip': [float(qy1), float(qy2)]
}
out_json.write_text(json.dumps(stats, indent=2))
print('saved', out_img)
print('saved', out_json)
print(stats)
