# Badminton AI (X vs Y Court)

Lightweight local pipeline for badminton video analysis:
- Player skeleton tracking (X top court, Y bottom court)
- Shuttle tracking (detector + motion fallback)
- Motion quantization (arms/torso/legs)
- Next-step prediction (winner proxy + landing point)
- Auto-generated visual report

## Quick Start (one command)
```bash
~/\.openclaw/workspace/projects/badminton-ai/run_all.sh
```

## Input Video
Put one of these on Desktop:
- `badminton_sample.mp4` (preferred)
- `badminton_hd.mp4` (auto-copied to sample)

## Main Artifacts
- `data/frame_features_v6.jsonl` — per-frame compact features (latest)
- `data/quant_features_v5.csv` — quantized motion features (latest)
- `reports/index.html` — full report dashboard
- `reports/feature_quality_audit_v6.json` — extraction quality (latest)
- `reports/rally_metrics_v3_compare.json` — current recommended model metrics (rally-aware)
- `reports/model_selection_note.md` — why current recommendation is rally_v3 (and when to fallback)
- `reports/run_health_latest.json` — latest pipeline health/status snapshot

Recommended reading order: `reports/index.html` → `reports/model_selection_note.md` → `reports/run_health_latest.json`

## Current Best (realistic)
- Model: **rally_v3** (event-segmented rally-aware)
- Winner ACC: **1.0000**
- Landing RMSE: **0.0250**
- Baseline reference: v4 winner ACC 0.9951 / RMSE 0.0896

## Model Versions
- v1: compact baseline (small sample, optimistic)
- v2: improved shuttle visibility (hybrid tracking)
- v3: adds racket proxy features
- temporal: window-based predictor

## Constraints / Cost
- Mostly free and local
- No full-frame storage required
- Store only JSONL/CSV + report images

## Safe Cleanup
Use:
```bash
./cleanup_safe.sh check-keep   # verify required artifacts exist
./cleanup_safe.sh dry-run      # preview reclaimable files
./cleanup_safe.sh apply        # move cleanup candidates to ~/.Trash
```

### Keep (do not delete)
- `reports/index.html`
- `reports/rally_metrics_v3_compare.json`
- `reports/quant_model_metrics_v4_compare.json`
- `reports/model_selection_note.md`
- `reports/win_prob_curve.png`
- `reports/shuttle_heatmap_denoised.png`
- `data/frame_features_v6.jsonl`
- `data/quant_features_v5.csv`

### Regenerable (safe to delete)
- old intermediate files (`frame_features.jsonl`, `quant_features.csv`)
- model cache weights (`yolov8n*.pt`) if disk is tight

## Next Recommended Upgrades
1. Manual labels for 50–100 rallies (true winner + landing)
2. Court homography (map to real court coordinates)
3. Explicit racket detector/keypoints model
4. Temporal model retrain on true labels
