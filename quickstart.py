import cv2
import json
from pathlib import Path
from ultralytics import YOLO

VIDEO = Path.home() / "Desktop" / "badminton_sample.mp4"
if not VIDEO.exists():
    VIDEO = Path.home() / "Desktop" / "test_video.mp4"

model = YOLO("yolov8n-pose.pt")  # auto-download once
cap = cv2.VideoCapture(str(VIDEO))

frame_count = 0
sample_every = 20
samples = []

while True:
    ok, frame = cap.read()
    if not ok:
        break
    frame_count += 1
    if frame_count % sample_every != 0:
        continue

    pred = model(frame, verbose=False)[0]
    num_people = 0
    keypoints_per_person = []

    if pred.keypoints is not None and len(pred.keypoints) > 0:
        num_people = len(pred.keypoints)
        for kp in pred.keypoints.xy:
            keypoints_per_person.append(len(kp))

    samples.append({
        "frame": frame_count,
        "people_detected": num_people,
        "keypoints_per_person": keypoints_per_person,
    })

cap.release()

out = {
    "video": str(VIDEO),
    "sample_every": sample_every,
    "total_samples": len(samples),
    "samples": samples[:200],
}
Path("pose_summary.json").write_text(json.dumps(out, indent=2))

detected_frames = sum(1 for s in samples if s["people_detected"] > 0)
print(f"Video: {VIDEO}")
print(f"Sampled frames: {len(samples)}")
print(f"Frames with players detected: {detected_frames}")
print("Saved: pose_summary.json")
