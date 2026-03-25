import cv2
import json
from pathlib import Path
from ultralytics import YOLO

VIDEO = Path.home() / "Desktop" / "badminton_sample.mp4"
if not VIDEO.exists():
    VIDEO = Path.home() / "Desktop" / "test_video.mp4"

# Models
pose_model = YOLO("yolov8n-pose.pt")
det_model = YOLO("yolov8n.pt")  # COCO classes incl. person, sports ball

cap = cv2.VideoCapture(str(VIDEO))
fps = cap.get(cv2.CAP_PROP_FPS) or 30

sample_every = 10
frame_idx = 0
timeline = []

rating_a = 50.0
rating_b = 50.0

while True:
    ok, frame = cap.read()
    if not ok:
        break
    frame_idx += 1
    if frame_idx % sample_every != 0:
        continue

    # Pose
    pose_pred = pose_model(frame, verbose=False)[0]
    people = int(len(pose_pred.keypoints)) if pose_pred.keypoints is not None else 0

    # Detection
    det_pred = det_model(frame, verbose=False)[0]
    classes = det_pred.boxes.cls.tolist() if det_pred.boxes is not None and det_pred.boxes.cls is not None else []
    # COCO: person=0, sports ball=32
    n_person = sum(1 for c in classes if int(c) == 0)
    n_ball = sum(1 for c in classes if int(c) == 32)

    # Heuristic win probability (upgrade vs MVP)
    # More confidence when 2 players + ball visible
    score_a = 0.5
    if people >= 2 or n_person >= 2:
        score_a += 0.05
    if n_ball >= 1:
        score_a += 0.03
    score_a = max(0.35, min(0.75, score_a))
    score_b = 1 - score_a

    rating_a += (score_a - 0.5) * 0.15
    rating_b += (score_b - 0.5) * 0.15

    timeline.append({
        "t_sec": round(frame_idx / fps, 2),
        "frame": frame_idx,
        "people_pose": people,
        "people_det": n_person,
        "ball_det": n_ball,
        "win_prob_a": round(score_a, 3),
        "win_prob_b": round(score_b, 3),
    })

cap.release()

rating_a = max(0, min(100, rating_a))
rating_b = max(0, min(100, rating_b))

out = {
    "video": str(VIDEO),
    "sample_every": sample_every,
    "frames_analyzed": len(timeline),
    "ratings": {"player_a": round(rating_a, 2), "player_b": round(rating_b, 2)},
    "timeline": timeline[:1000],
    "notes": [
        "COCO sports ball is imperfect for badminton shuttlecock.",
        "Next: fine-tune shuttlecock detector or use small-object tracker.",
        "Next: add rally segmentation + per-rally win prob curve."
    ]
}

out_path = Path(__file__).with_name("full_pipeline_output.json")
out_path.write_text(json.dumps(out, indent=2))
print("Saved", out_path)
print("Frames analyzed:", len(timeline))
print("Ratings:", out["ratings"])
