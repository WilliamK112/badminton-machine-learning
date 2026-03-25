import json
import os
from pathlib import Path
import cv2
from ultralytics import YOLO
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
VIDEO = Path.home() / "Desktop" / "badminton_sample.mp4"
if not VIDEO.exists():
    VIDEO = Path.home() / "Desktop" / "badminton_hd.mp4"

pose_model = YOLO(str(ROOT / "yolov8n-pose.pt"))
det_model = YOLO(str(ROOT / "yolov8n.pt"))

out_path = ROOT / "data" / "frame_features.jsonl"
out_path.parent.mkdir(parents=True, exist_ok=True)

cap = cv2.VideoCapture(str(VIDEO))
fps = cap.get(cv2.CAP_PROP_FPS) or 30
W = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH) or 1)
H = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT) or 1)

sample_every = int(os.getenv("SAMPLE_EVERY", "6"))
frame_idx = 0
prev_shuttle = None
prev_shuttle_v = [0.0, 0.0]

# quality controls
pose_conf_th = 0.35
kpt_conf_th = 0.25
shuttle_conf_th = 0.15
max_shuttle_jump = 0.18


def norm_xy(x, y):
    return [float(x) / W, float(y) / H]

with out_path.open("w") as f:
    while True:
        ok, frame = cap.read()
        if not ok:
            break
        frame_idx += 1
        if frame_idx % sample_every != 0:
            continue

        pose = pose_model(frame, verbose=False)[0]
        det = det_model(frame, verbose=False)[0]

        # players by y-center split (top=X, bottom=Y), filtered by detection confidence
        players = {"X": None, "Y": None}
        if pose.boxes is not None and pose.keypoints is not None and len(pose.boxes):
            boxes = pose.boxes.xyxy.cpu().numpy()
            box_conf = pose.boxes.conf.cpu().numpy() if pose.boxes.conf is not None else np.ones(len(boxes))
            kpts_xy = pose.keypoints.xy.cpu().numpy() if len(pose.keypoints) else np.empty((0, 17, 2))
            kpts_cf = pose.keypoints.conf.cpu().numpy() if pose.keypoints.conf is not None else np.ones((len(boxes), 17))
            candidates = []
            for i, b in enumerate(boxes):
                conf = float(box_conf[i])
                if conf < pose_conf_th:
                    continue
                x1, y1, x2, y2 = b.tolist()
                if x2 <= x1 or y2 <= y1:
                    continue
                area = (x2 - x1) * (y2 - y1)
                cy = (y1 + y2) / 2
                kpt_rows = []
                for j, p in enumerate(kpts_xy[i].tolist()):
                    kp_conf = float(kpts_cf[i][j])
                    vis = 1.0 if kp_conf >= kpt_conf_th else 0.0
                    kpt_rows.append([p[0] / W, p[1] / H, vis])
                candidates.append((cy, area, conf, [x1, y1, x2, y2], kpt_rows))

            candidates.sort(key=lambda x: x[0])
            if len(candidates) >= 1:
                c = candidates[0]
                players["X"] = {
                    "bbox": [v / W if i % 2 == 0 else v / H for i, v in enumerate(c[3])],
                    "kpts": c[4],
                    "center": [((c[3][0] + c[3][2]) / 2) / W, ((c[3][1] + c[3][3]) / 2) / H],
                    "conf": c[2],
                }
            if len(candidates) >= 2:
                c = candidates[-1]
                players["Y"] = {
                    "bbox": [v / W if i % 2 == 0 else v / H for i, v in enumerate(c[3])],
                    "kpts": c[4],
                    "center": [((c[3][0] + c[3][2]) / 2) / W, ((c[3][1] + c[3][3]) / 2) / H],
                    "conf": c[2],
                }

        # shuttle from sports-ball class=32 (confidence + temporal consistency)
        shuttle = {"xy": None, "v": [0.0, 0.0], "speed": 0.0, "visible": False, "conf": 0.0}
        if det.boxes is not None and len(det.boxes) > 0:
            cls = det.boxes.cls.cpu().numpy().astype(int)
            bxy = det.boxes.xyxy.cpu().numpy()
            bcf = det.boxes.conf.cpu().numpy() if det.boxes.conf is not None else np.ones(len(bxy))
            balls = []
            for c, b, conf in zip(cls, bxy, bcf):
                if c != 32 or conf < shuttle_conf_th:
                    continue
                cx, cy = (b[0] + b[2]) / 2, (b[1] + b[3]) / 2
                area = max((b[2] - b[0]) * (b[3] - b[1]), 1.0)
                balls.append((norm_xy(cx, cy), float(conf), float(area)))

            if balls:
                if prev_shuttle is None:
                    # bootstrap: smallest reasonable candidate with best confidence
                    balls.sort(key=lambda x: (x[2], -x[1]))
                    cand_xy, cand_conf, _ = balls[0]
                else:
                    pred_xy = [prev_shuttle[0] + prev_shuttle_v[0], prev_shuttle[1] + prev_shuttle_v[1]]
                    scored = []
                    for xy, conf, area in balls:
                        dx = xy[0] - pred_xy[0]
                        dy = xy[1] - pred_xy[1]
                        dist = float((dx * dx + dy * dy) ** 0.5)
                        # prefer trajectory consistency; lightly penalize larger blobs
                        score = dist + 0.01 * (area / (W * H)) - 0.02 * conf
                        scored.append((score, dist, xy, conf))
                    scored.sort(key=lambda x: x[0])
                    _, dist, cand_xy, cand_conf = scored[0]
                    if dist > max_shuttle_jump:
                        cand_xy, cand_conf = None, None

                if cand_xy is not None:
                    shuttle["xy"] = cand_xy
                    shuttle["visible"] = True
                    shuttle["conf"] = cand_conf
                    if prev_shuttle is not None:
                        vx = cand_xy[0] - prev_shuttle[0]
                        vy = cand_xy[1] - prev_shuttle[1]
                        shuttle["v"] = [vx, vy]
                        shuttle["speed"] = float((vx * vx + vy * vy) ** 0.5)
                        prev_shuttle_v = shuttle["v"]
                    prev_shuttle = cand_xy

        rec = {
            "frame": frame_idx,
            "t_sec": round(frame_idx / fps, 3),
            "court": {"x_line_y": 0.5},
            "players": players,
            "shuttle": shuttle,
        }
        f.write(json.dumps(rec) + "\n")

cap.release()
print("saved", out_path)
