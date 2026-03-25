"""
Simple court annotation tool using OpenCV
Click and drag to draw court bounding boxes
"""
import cv2
import os
from pathlib import Path

# Settings
FRAMES_DIR = Path("data/training_frames")
LABELS_DIR = Path("data/training_labels")
LABELS_DIR.mkdir(exist_ok=True)

# Get all frame files
frame_files = sorted(FRAMES_DIR.glob("*.jpg"))
print(f"Found {len(frame_files)} frames")

# State
current_idx = 0
drawing = False
start_x, start_y = -1, -1
rects = []  # rectangles for current frame

# Mouse callback
def mouse_event(event, x, y, flags, param):
    global drawing, start_x, start_y, rects
    
    if event == cv2.EVENT_LBUTTONDOWN:
        drawing = True
        start_x, start_y = x, y
        rects = []  # reset for new frame
    
    elif event == cv2.EVENT_MOUSEMOVE:
        if drawing:
            img = param.copy()
            cv2.rectangle(img, (start_x, start_y), (x, y), (0, 255, 0), 2)
            cv2.imshow("Annotate", img)
    
    elif event == cv2.EVENT_LBUTTONUP:
        drawing = False
        end_x, end_y = x, y
        # Normalize rect
        x1 = min(start_x, end_x)
        y1 = min(start_y, end_y)
        x2 = max(start_x, end_x)
        y2 = max(start_y, end_y)
        rects.append((x1, y1, x2, y2))

# Main loop
while current_idx < len(frame_files):
    frame_path = frame_files[current_idx]
    img = cv2.imread(str(frame_path))
    h, w = img.shape[:2]
    
    print(f"\n[{current_idx+1}/{len(frame_files)}] {frame_path.name}")
    print("Drag to draw court box. Press 's' to save & next, 'r' to redo, 'q' to quit")
    
    rects = []
    display = img.copy()
    
    cv2.imshow("Annotate", display)
    cv2.setMouseCallback("Annotate", mouse_event, display)
    
    key = cv2.waitKey(0) & 0xFF
    
    if key == ord('q'):
        break
    elif key == ord('r'):
        rects = []
        continue
    elif key == ord('s') and rects:
        # Save YOLO format
        x1, y1, x2, y2 = rects[0]
        
        # YOLO format: class center_x center_y width height (normalized)
        center_x = ((x1 + x2) / 2) / w
        center_y = ((y1 + y2) / 2) / h
        bw = (x2 - x1) / w
        bh = (y2 - y1) / h
        
        label_path = LABELS_DIR / f"{frame_path.stem}.txt"
        with open(label_path, 'w') as f:
            f.write(f"0 {center_x:.6f} {center_y:.6f} {bw:.6f} {bh:.6f}\n")
        
        print(f"  Saved: {label_path.name}")
        current_idx += 1

cv2.destroyAllWindows()
print("\nDone!")
