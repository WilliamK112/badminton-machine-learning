#!/usr/bin/env python3
"""Shuttle annotation - s:save, k:next, p:prev"""
import cv2, json
from pathlib import Path

VIDEO = '/Users/William/Desktop/badminton_sample.mp4'
OUT = Path('/Users/William/.openclaw/workspace/projects/badminton-ai/data/shuttle_annotations/shuttle_positions.jsonl')

data = {}
if OUT.exists():
    for line in open(OUT):
        d = json.loads(line)
        data[d['frame_idx']] = (d['x'], d['y'])

cap = cv2.VideoCapture(VIDEO)
total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
frame = 0

print(f"Total: {total}, Annotated: {len(data)}")

def save():
    with open(OUT, 'w') as f:
        for k,v in sorted(data.items()):
            f.write(json.dumps({'frame_idx':k,'x':v[0],'y':v[1]})+'\n')

def mouse(e, x, y, flags, param):
    if e == cv2.EVENT_LBUTTONDOWN:
        data[frame] = (x, y)
        save()
        print(f"Saved frame {frame}")

cv2.namedWindow('Annotate')
cv2.setMouseCallback('Annotate', mouse)

while True:
    cap.set(cv2.CAP_PROP_POS_FRAMES, frame)
    ret, img = cap.read()
    if not ret: break
    
    disp = img.copy()
    
    if frame in data:
        cv2.circle(disp, data[frame], 15, (0,255,0), -1)
        cv2.putText(disp, f"OK {frame}", (10,40), cv2.FONT_HERSHEY_SIMPLEX, 1, (0,255,0), 2)
    else:
        cv2.putText(disp, f"{frame}", (10,40), cv2.FONT_HERSHEY_SIMPLEX, 1, (0,0,255), 2)
    
    cv2.putText(disp, "k:next p:prev 0-9:jump d:del s:save q:quit", (10,disp.shape[0]-20), 
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255,255,255), 1)
    
    cv2.imshow('Annotate', disp)
    
    k = cv2.waitKey(0) & 0xFF
    if k == ord('q'): break
    elif k == ord('k'): frame = min(frame+1, total-1)
    elif k == ord('p'): frame = max(frame-1, 0)
    elif k == ord('d') and frame in data: del data[frame]; save()
    elif k == ord('s'): save(); print(f"Saved {len(data)}")
    elif 48 <= k <= 57: frame = int(total * (k-48) / 10)

cap.release()
cv2.destroyAllWindows()
save()
print(f"Done! {len(data)} annotations")
