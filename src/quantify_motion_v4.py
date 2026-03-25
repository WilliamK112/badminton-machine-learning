import csv
import json
import math
from pathlib import Path
from datetime import datetime

ROOT = Path(__file__).resolve().parents[1]
feat_file = ROOT / "data" / "frame_features_v5.jsonl"
out_csv = ROOT / "data" / "quant_features_v4.csv"
out_report = ROOT / "reports" / f"motion_quantification_v4_{datetime.now().strftime('%Y-%m-%dT%H-%M-%S')}.json"

rows = [json.loads(x) for x in feat_file.read_text().splitlines() if x.strip()]


def angle(a, b, c):
    ab = (a[0] - b[0], a[1] - b[1])
    cb = (c[0] - b[0], c[1] - b[1])
    nab = (ab[0] ** 2 + ab[1] ** 2) ** 0.5
    ncb = (cb[0] ** 2 + cb[1] ** 2) ** 0.5
    if nab == 0 or ncb == 0:
        return 0.0
    dot = ab[0] * cb[0] + ab[1] * cb[1]
    x = max(-1.0, min(1.0, dot / (nab * ncb)))
    return math.degrees(math.acos(x))


def get_angles(k):
    p = [(pt[0], pt[1]) for pt in k]
    return {
        "l_forearm": angle(p[5], p[7], p[9]) if len(p) > 9 else 0.0,
        "r_forearm": angle(p[6], p[8], p[10]) if len(p) > 10 else 0.0,
        "l_upperarm": angle(p[11], p[5], p[7]) if len(p) > 11 else 0.0,
        "r_upperarm": angle(p[12], p[6], p[8]) if len(p) > 12 else 0.0,
        "torso_rot": abs((p[6][0] - p[5][0]) - (p[12][0] - p[11][0])) if len(p) > 12 else 0.0,
        "l_thigh": angle(p[5], p[11], p[13]) if len(p) > 13 else 0.0,
        "r_thigh": angle(p[6], p[12], p[14]) if len(p) > 14 else 0.0,
        "l_calf": angle(p[11], p[13], p[15]) if len(p) > 15 else 0.0,
        "r_calf": angle(p[12], p[14], p[16]) if len(p) > 16 else 0.0,
    }


def safe_mean(vals):
    vals = [v for v in vals if v is not None]
    return sum(vals) / len(vals) if vals else 0.0


fields = [
    "frame", "t_sec", "winner_proxy", "shuttle_x", "shuttle_y", "shuttle_speed",
    "X_l_forearm", "X_r_forearm", "X_l_upperarm", "X_r_upperarm", "X_torso_rot", "X_l_thigh", "X_r_thigh", "X_l_calf", "X_r_calf",
    "X_arms_ang_vel", "X_torso_ang_vel", "X_legs_ang_vel",
    "Y_l_forearm", "Y_r_forearm", "Y_l_upperarm", "Y_r_upperarm", "Y_torso_rot", "Y_l_thigh", "Y_r_thigh", "Y_l_calf", "Y_r_calf",
    "Y_arms_ang_vel", "Y_torso_ang_vel", "Y_legs_ang_vel",
]

prev_t = None
prev_angles = {"X": None, "Y": None}
stats = {
    "X": {"arms": [], "torso": [], "legs": []},
    "Y": {"arms": [], "torso": [], "legs": []},
}

with out_csv.open("w", newline="") as f:
    wr = csv.DictWriter(f, fieldnames=fields)
    wr.writeheader()

    for r in rows:
        row = {"frame": r["frame"], "t_sec": r["t_sec"]}
        sh = r.get("shuttle", {})
        xy = sh.get("xy") or [0.5, 0.5]
        row["shuttle_x"] = xy[0]
        row["shuttle_y"] = xy[1]
        row["shuttle_speed"] = sh.get("speed", 0.0)
        row["winner_proxy"] = 1 if xy[1] > 0.5 else 0

        dt = 0.0 if prev_t is None else max(1e-6, float(r["t_sec"]) - float(prev_t))

        for side in ["X", "Y"]:
            p = r.get("players", {}).get(side)
            if p and p.get("kpts"):
                a = get_angles(p["kpts"])
            else:
                a = {k: 0.0 for k in ["l_forearm", "r_forearm", "l_upperarm", "r_upperarm", "torso_rot", "l_thigh", "r_thigh", "l_calf", "r_calf"]}

            for k, v in a.items():
                row[f"{side}_{k}"] = v

            if prev_angles[side] is None or dt <= 0:
                arms_vel = torso_vel = legs_vel = 0.0
            else:
                pa = prev_angles[side]
                vels = {k: abs((a[k] - pa[k]) / dt) for k in a.keys()}
                arms_vel = safe_mean([vels["l_forearm"], vels["r_forearm"], vels["l_upperarm"], vels["r_upperarm"]])
                torso_vel = vels["torso_rot"]
                legs_vel = safe_mean([vels["l_thigh"], vels["r_thigh"], vels["l_calf"], vels["r_calf"]])

            row[f"{side}_arms_ang_vel"] = arms_vel
            row[f"{side}_torso_ang_vel"] = torso_vel
            row[f"{side}_legs_ang_vel"] = legs_vel

            stats[side]["arms"].append(arms_vel)
            stats[side]["torso"].append(torso_vel)
            stats[side]["legs"].append(legs_vel)
            prev_angles[side] = a

        wr.writerow(row)
        prev_t = r["t_sec"]

report = {
    "timestamp_local": datetime.now().astimezone().isoformat(timespec="seconds"),
    "step_executed": "Add motion quantification (priority #2) on v5 smoothed features",
    "inputs": str(feat_file),
    "artifacts": [str(out_csv), str(out_report)],
    "frames": len(rows),
    "summary": {
        "X": {
            "arms_ang_vel_mean": round(safe_mean(stats["X"]["arms"]), 4),
            "torso_ang_vel_mean": round(safe_mean(stats["X"]["torso"]), 4),
            "legs_ang_vel_mean": round(safe_mean(stats["X"]["legs"]), 4),
        },
        "Y": {
            "arms_ang_vel_mean": round(safe_mean(stats["Y"]["arms"]), 4),
            "torso_ang_vel_mean": round(safe_mean(stats["Y"]["torso"]), 4),
            "legs_ang_vel_mean": round(safe_mean(stats["Y"]["legs"]), 4),
        },
    },
    "next_step": "Improve rally segmentation (priority #3) using v5/v4 motion features.",
}
out_report.write_text(json.dumps(report, indent=2), encoding="utf-8")

print("saved", out_csv)
print("saved", out_report)
print(json.dumps(report, indent=2))
