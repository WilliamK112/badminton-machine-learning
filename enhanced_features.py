"""
Enhanced feature extraction: shuttlecock detection + racket + angular velocity
"""
import cv2
import json
import numpy as np
from pathlib import Path
from collections import deque
from ultralytics import YOLO

VIDEO = Path.home() / "Desktop" / "badminton_sample.mp4"
if not VIDEO.exists():
    VIDEO = Path.home() / "Desktop" / "test_video.mp4"

# Models
pose_model = YOLO("yolov8n-pose.pt")
det_model = YOLO("yolov8n.pt")

# Shuttle tracking
shuttle_history = deque(maxlen=10)
MIN_SHUTTLE_AREA = 50
MAX_SHUTTLE_AREA = 800

def detect_shuttle(frame, prev_pos=None):
    """Detect shuttlecock using motion + appearance cues."""
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    
    # Method 1: Bright object detection (shuttle is light colored)
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
    # White/light objects
    lower_white = np.array([0, 0, 200])
    upper_white = np.array([180, 30, 255])
    mask_white = cv2.inRange(hsv, lower_white, upper_white)
    
    # Method 2: If we have previous position, search nearby
    if prev_pos is not None:
        h, w = frame.shape[:2]
        search_radius = 150
        x, y = prev_pos
        x = int(max(0, min(w-1, x)))
        y = int(max(0, min(h-1, y)))
        
        roi = frame[max(0,y-search_radius):min(h,y+search_radius),
                    max(0,x-search_radius):min(w,x+search_radius)]
        if roi.size > 0:
            roi_gray = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
            # Look for bright small objects
            blur = cv2.GaussianBlur(roi_gray, (5, 5), 0)
            _, thresh = cv2.threshold(blur, 200, 255, cv2.THRESH_BINARY)
            contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            
            for cnt in contours:
                area = cv2.contourArea(cnt)
                if MIN_SHUTTLE_AREA < area < MAX_SHUTTLE_AREA:
                    moments = cv2.moments(cnt)
                    if moments["m00"] > 0:
                        cx = int(moments["m10"] / moments["m00"]) + max(0, x - search_radius)
                        cy = int(moments["m01"] / moments["m00"]) + max(0, y - search_radius)
                        return (cx, cy), True
    
    # Fallback: search full frame
    blur = cv2.GaussianBlur(gray, (5, 5), 0)
    _, thresh = cv2.threshold(blur, 220, 255, cv2.THRESH_BINARY)
    contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    best_candidate = None
    best_score = 0
    
    for cnt in contours:
        area = cv2.contourArea(cnt)
        if MIN_SHUTTLE_AREA < area < MAX_SHUTTLE_AREA:
            # Check circularity (shuttle is somewhat round)
            perimeter = cv2.arcLength(cnt, True)
            if perimeter > 0:
                circularity = 4 * np.pi * area / (perimeter ** 2)
                if circularity > 0.3:
                    moments = cv2.moments(cnt)
                    if moments["m00"] > 0:
                        cx = int(moments["m10"] / moments["m00"])
                        cy = int(moments["m01"] / moments["m00"])
                        score = circularity * (area / MAX_SHUTTLE_AREA)
                        if score > best_score:
                            best_score = score
                            best_candidate = (cx, cy)
    
    return best_candidate, best_candidate is not None


def compute_angular_velocity(kpts, prev_kpts, dt):
    """Compute angular velocity for body joints."""
    if prev_kpts is None or dt == 0:
        return {}
    
    angular_vel = {}
    
    # Key joint pairs for angle computation (shoulder, elbow, wrist indices)
    joint_pairs = [
        ("l_forearm", 5, 7, 9),   # left shoulder, elbow, wrist
        ("r_forearm", 6, 8, 10),  # right shoulder, elbow, wrist
        ("torso", 5, 6, 7),       # shoulder line (approximate)
    ]
    
    def angle_from_points(p1, p2, p3):
        """Angle at p2 formed by p1-p2-p3."""
        v1 = np.array([p1[0] - p2[0], p1[1] - p2[1]])
        v2 = np.array([p3[0] - p2[0], p3[1] - p2[1]])
        norm = np.linalg.norm(v1) * np.linalg.norm(v2)
        if norm < 1e-6:
            return 0
        cos_angle = np.dot(v1, v2) / norm
        cos_angle = np.clip(cos_angle, -1, 1)
        return np.degrees(np.arccos(cos_angle))
    
    # Extract keypoints as numpy arrays
    keypoints = kpts.keypoints.xy[0].cpu().numpy()
    prev_keypoints = prev_kpts.keypoints.xy[0].cpu().numpy()
    
    for name, i, j, k in joint_pairs:
        try:
            curr_ang = angle_from_points(keypoints[i], keypoints[j], keypoints[k])
            prev_ang = angle_from_points(prev_keypoints[i], prev_keypoints[j], prev_keypoints[k])
            angular_vel[name] = round(float(abs(curr_ang - prev_ang) / dt), 2)
        except Exception:
            pass
    
    return angular_vel


def detect_racket(frame, person_bbox=None):
    """Simple racket detection - looks for elongated objects near player."""
    # This is a placeholder - real implementation would need custom model
    return None


# Main processing
cap = cv2.VideoCapture(str(VIDEO))
fps = cap.get(cv2.CAP_PROP_FPS) or 30
width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

sample_every = 10
frame_idx = 0
timeline = []
prev_pose = None

print(f"Processing {VIDEO}")
print(f"Resolution: {width}x{height}, FPS: {fps}")

while True:
    ok, frame = cap.read()
    if not ok:
        break
    
    frame_idx += 1
    if frame_idx % sample_every != 0:
        continue
    
    t_sec = frame_idx / fps
    
    # Pose detection
    pose_pred = pose_model(frame, verbose=False)[0]
    people = len(pose_pred.keypoints) if pose_pred.keypoints is not None else 0
    
    # Angular velocity - use previous sampled frame's pose
    angular_vel = {}
    if people > 0 and prev_pose is not None:
        try:
            angular_vel = compute_angular_velocity(pose_pred, prev_pose, sample_every / fps)
        except Exception as e:
            print(f"Angular velocity error: {e}")
    
    # Store for next sample (only if people detected)
    if people > 0:
        prev_pose = pose_pred
    
    # Person detection
    det_pred = det_model(frame, verbose=False)[0]
    classes = det_pred.boxes.cls.tolist() if det_pred.boxes is not None else []
    n_person = sum(1 for c in classes if int(c) == 0)
    n_ball = sum(1 for c in classes if int(c) == 32)
    
    # Shuttle detection
    prev_shuttle = shuttle_history[-1] if shuttle_history else None
    shuttle_pos, shuttle_visible = detect_shuttle(frame, prev_shuttle)
    if shuttle_visible and shuttle_pos:
        shuttle_history.append(shuttle_pos)
    
    # Compute shuttle velocity
    shuttle_vel = [0, 0]
    shuttle_speed = 0
    if len(shuttle_history) >= 2:
        dx = shuttle_history[-1][0] - shuttle_history[-2][0]
        dy = shuttle_history[-1][1] - shuttle_history[-2][1]
        dt = sample_every / fps
        shuttle_vel = [round(dx / dt / width, 4), round(dy / dt / height, 4)]
        shuttle_speed = round(np.sqrt(dx**2 + dy**2) / dt / 1000, 3)  # pixels per sec / 1000
    
    # Win probability
    score_a = 0.5
    if people >= 2:
        score_a += 0.05
    if shuttle_visible:
        score_a += 0.08
    if n_ball >= 1:
        score_a += 0.03
    score_a = max(0.35, min(0.75, score_a))
    
    record = {
        "frame": frame_idx,
        "t_sec": round(t_sec, 2),
        "players_detected": people,
        "shuttle": {
            "xy": [round(shuttle_pos[0]/width, 3), round(shuttle_pos[1]/height, 3)] if shuttle_pos else None,
            "v": shuttle_vel,
            "speed": shuttle_speed,
            "visible": shuttle_visible
        },
        "angular_vel": angular_vel,
        "win_prob_a": round(score_a, 3),
        "win_prob_b": round(1 - score_a, 3)
    }
    timeline.append(record)
    
    if frame_idx % 100 == 0:
        print(f"Frame {frame_idx}: players={people}, shuttle={'yes' if shuttle_visible else 'no'}")

cap.release()

out = {
    "video": str(VIDEO),
    "frames_analyzed": len(timeline),
    "shuttle_detected_frames": sum(1 for r in timeline if r["shuttle"]["visible"]),
    "angular_velocity_samples": sum(1 for r in timeline if r["angular_vel"]),
    "timeline": timeline[:500],
    "notes": ["Added shuttle detection + tracking", "Added angular velocity estimation", "Next: add racket detection model"]
}

out_path = Path(__file__).with_name("enhanced_features_output.json")
out_path.write_text(json.dumps(out, indent=2))
print(f"\nSaved {out_path}")
print(f"Frames: {len(timeline)}, Shuttle detected: {out['shuttle_detected_frames']}, Angular vel: {out['angular_velocity_samples']}")