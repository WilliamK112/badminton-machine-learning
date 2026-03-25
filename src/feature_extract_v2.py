import json
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

out_path = ROOT / "data" / "frame_features_v2.jsonl"
out_path.parent.mkdir(parents=True, exist_ok=True)

cap = cv2.VideoCapture(str(VIDEO))
fps = cap.get(cv2.CAP_PROP_FPS) or 30
W = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH) or 1)
H = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT) or 1)

sample_every = 4  # denser sampling for shuttle
frame_idx = 0
prev_shuttle = None
prev_gray = None

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

        players = {"X": None, "Y": None}
        if pose.boxes is not None and pose.keypoints is not None:
            boxes = pose.boxes.xyxy.cpu().numpy() if len(pose.boxes) else np.empty((0,4))
            kpts = pose.keypoints.xy.cpu().numpy() if len(pose.keypoints) else np.empty((0,17,2))
            candidates = []
            for i, b in enumerate(boxes):
                x1,y1,x2,y2 = b.tolist()
                cy = (y1+y2)/2
                candidates.append((cy, [x1,y1,x2,y2], kpts[i].tolist()))
            candidates.sort(key=lambda x: x[0])
            if len(candidates) >= 1:
                c = candidates[0]
                players["X"] = {
                    "bbox": [v/W if i%2==0 else v/H for i,v in enumerate(c[1])],
                    "kpts": [[p[0]/W, p[1]/H, 1.0] for p in c[2]],
                    "center": [((c[1][0]+c[1][2])/2)/W, ((c[1][1]+c[1][3])/2)/H],
                }
            if len(candidates) >= 2:
                c = candidates[-1]
                players["Y"] = {
                    "bbox": [v/W if i%2==0 else v/H for i,v in enumerate(c[1])],
                    "kpts": [[p[0]/W, p[1]/H, 1.0] for p in c[2]],
                    "center": [((c[1][0]+c[1][2])/2)/W, ((c[1][1]+c[1][3])/2)/H],
                }

        shuttle = {"xy": None, "v": [0.0,0.0], "speed": 0.0, "visible": False, "source": None}

        # A) detector sports-ball
        if det.boxes is not None and len(det.boxes) > 0:
            cls = det.boxes.cls.cpu().numpy().astype(int)
            bxy = det.boxes.xyxy.cpu().numpy()
            balls = [b for c,b in zip(cls,bxy) if c == 32]
            if balls:
                balls = sorted(balls, key=lambda b: (b[2]-b[0])*(b[3]-b[1]))
                b = balls[0]
                cx, cy = (b[0]+b[2])/2, (b[1]+b[3])/2
                xy = norm_xy(cx, cy)
                shuttle.update({"xy": xy, "visible": True, "source": "det"})

        # B) fallback: motion blob for tiny fast object
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        if not shuttle['visible'] and prev_gray is not None:
            diff = cv2.absdiff(gray, prev_gray)
            _, th = cv2.threshold(diff, 25, 255, cv2.THRESH_BINARY)
            th = cv2.medianBlur(th, 3)
            cnts, _ = cv2.findContours(th, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            small = []
            for c in cnts:
                x,y,w,h = cv2.boundingRect(c)
                area = w*h
                if 4 <= area <= 120:  # tiny moving blob
                    cx, cy = x + w/2, y + h/2
                    small.append((area, cx, cy))
            if small:
                small.sort(key=lambda z: z[0])
                _, cx, cy = small[0]
                shuttle.update({"xy": norm_xy(cx, cy), "visible": True, "source": "motion"})
        prev_gray = gray

        if shuttle['xy'] is not None and prev_shuttle is not None:
            vx = shuttle['xy'][0] - prev_shuttle[0]
            vy = shuttle['xy'][1] - prev_shuttle[1]
            shuttle['v'] = [vx, vy]
            shuttle['speed'] = float((vx*vx + vy*vy) ** 0.5)
        if shuttle['xy'] is not None:
            prev_shuttle = shuttle['xy']

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
