import json
from pathlib import Path
import matplotlib.pyplot as plt
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
fp = ROOT / "full_pipeline_output.json"
if not fp.exists():
    raise SystemExit("full_pipeline_output.json not found")

data = json.loads(fp.read_text())
tl = data.get("timeline", [])
if not tl:
    raise SystemExit("timeline empty")

t = [x["t_sec"] for x in tl]
p = [x["win_prob_a"] for x in tl]

reports = ROOT / "reports"
reports.mkdir(exist_ok=True)

# 1) Win probability curve
plt.figure(figsize=(10,4))
plt.plot(t, p, label="P(X wins next)")
plt.ylim(0,1)
plt.xlabel("Time (sec)")
plt.ylabel("Probability")
plt.title("Dynamic Win Probability (Player X)")
plt.grid(alpha=0.25)
plt.legend()
curve = reports / "win_prob_curve.png"
plt.tight_layout()
plt.savefig(curve, dpi=140)
plt.close()

# 2) Landing proxy heatmap using shuttle positions
#    (from full_pipeline timeline we only have counts; fallback to MVP output if available)
rally = ROOT / "rally_mvp_output.json"
points = []
if rally.exists():
    rv = json.loads(rally.read_text())
    for row in rv.get("timeline", []):
        # no landing coordinates in MVP timeline, skip
        pass

# Use feature file shuttle positions as heatmap proxy
feat = ROOT / "data" / "frame_features.jsonl"
if feat.exists():
    for line in feat.read_text().splitlines():
        r = json.loads(line)
        xy = r.get("shuttle", {}).get("xy")
        if xy is not None:
            points.append(xy)

if points:
    arr = np.array(points)
    plt.figure(figsize=(5,9))
    plt.hist2d(arr[:,0], arr[:,1], bins=40, cmap="magma")
    plt.colorbar(label="Density")
    plt.gca().invert_yaxis()
    plt.title("Shuttle Position Heatmap (Proxy)")
    plt.xlabel("Court X")
    plt.ylabel("Court Y")
    heat = reports / "shuttle_heatmap.png"
    plt.tight_layout()
    plt.savefig(heat, dpi=140)
    plt.close()
else:
    heat = None

summary = {
    "ratings": data.get("ratings", {}),
    "frames_analyzed": data.get("frames_analyzed"),
    "artifacts": {
        "win_prob_curve": str(curve),
        "shuttle_heatmap": str(heat) if heat else None,
    }
}

summary_path = reports / "report_summary.json"
summary_path.write_text(json.dumps(summary, indent=2))

print("saved", curve)
if heat:
    print("saved", heat)
print("saved", summary_path)
