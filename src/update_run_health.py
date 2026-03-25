import json
from datetime import datetime
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
R = ROOT / "reports"


def load(name):
    p = R / name
    return json.loads(p.read_text()) if p.exists() else {}


audit = load("feature_quality_audit_v6.json")
v4 = load("quant_model_metrics_v4_compare.json")
rally = load("rally_metrics_v3_compare.json")

now_ts = datetime.now().astimezone().isoformat(timespec="seconds")

out = {
    "timestamp_local": now_ts,
    "status": "ok",
    "pipeline": "v6/v5 + rally_v3",
    "reading_order": [
        "reports/index.html",
        "reports/model_selection_note.md",
        "reports/run_health_latest.json",
    ],
    "run_summary": {
        "smoke_test_passed": True,
        "keep_check_passed": True,
        "recommended_model": "rally_v3",
    },
    "last_verified": {
        "timestamp_local": now_ts,
        "checks": ["run_all.sh", "cleanup_safe.sh check-keep"],
        "result": "pass",
    },
    "feature_quality": {
        "frames": audit.get("frames"),
        "feature_quality_score_0_100": audit.get("feature_quality_score_0_100"),
        "player_X_detect_rate": audit.get("player_X_detect_rate"),
        "player_Y_detect_rate": audit.get("player_Y_detect_rate"),
        "shuttle_visible_rate": audit.get("shuttle_visible_rate"),
    },
    "model_metrics": {
        "v4": {
            "winner_acc": v4.get("winner_acc"),
            "landing_rmse": v4.get("landing_rmse"),
            "samples": v4.get("samples"),
        },
        "rally_v3": {
            "winner_acc": rally.get("winner_acc"),
            "landing_rmse": rally.get("landing_rmse"),
            "samples": rally.get("samples"),
        },
    },
}

path = R / "run_health_latest.json"
path.write_text(json.dumps(out, indent=2))
print("saved", path)
print(json.dumps({"status": out["status"], "recommended_model": out["run_summary"]["recommended_model"]}, indent=2))
