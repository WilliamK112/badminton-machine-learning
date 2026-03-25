"""
Court Detection V4 - 改进版颜色/边缘检测
更robust的羽毛球场地检测
"""
import cv2
import numpy as np
from pathlib import Path
from ultralytics import YOLO


def detect_white_lines_improved(frame, debug=False):
    """
    改进的白色场地线检测
    """
    h, w = frame.shape[:2]
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    
    # 方法1: 在HSV空间检测白色
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
    
    # 扩展白色范围（包含浅灰白色）
    lower_white1 = np.array([0, 0, 200])
    upper_white1 = np.array([180, 30, 255])
    lower_white2 = np.array([0, 0, 150])
    upper_white2 = np.array([180, 50, 200])
    
    mask1 = cv2.inRange(hsv, lower_white1, upper_white1)
    mask2 = cv2.inRange(hsv, lower_white2, upper_white2)
    white_mask = cv2.bitwise_or(mask1, mask2)
    
    # 方法2: 结合Laplacian边缘检测（增强细线）
    laplacian = cv2.Laplacian(gray, cv2.CV_64F)
    laplacian = np.uint8(np.absolute(laplacian))
    
    # 方法3: 在蓝色通道检测（羽毛球场地通常是蓝色/绿色）
    blue = frame[:,:,0]
    green = frame[:,:,1]
    red = frame[:,:,2]
    
    # 羽毛球球场通常是蓝色 cushion + 白色线条
    # 检测蓝色区域
    blue_mask = cv2.inRange(blue, 100, 180)
    
    # 组合
    combined = cv2.bitwise_and(white_mask, blue_mask) if debug else white_mask
    
    # 去噪
    kernel = np.ones((3,3), np.uint8)
    combined = cv2.morphologyEx(combined, cv2.MORPH_CLOSE, kernel, iterations=2)
    combined = cv2.morphologyEx(combined, cv2.MORPH_OPEN, kernel, iterations=1)
    
    # 边缘检测
    edges = cv2.Canny(combined, 50, 150)
    
    # 霍夫直线检测
    lines = cv2.HoughLinesP(edges, 1, np.pi/180, 80, minLineLength=60, maxLineGap=20)
    
    h_lines = []
    v_lines = []
    
    if lines is not None:
        for line in lines:
            x1, y1, x2, y2 = line[0]
            angle = np.abs(np.arctan2(y2-y1, x2-x1) * 180 / np.pi)
            length = np.sqrt((x2-x1)**2 + (y2-y1)**2)
            
            if length > 80:
                if angle < 15 or angle > 165:
                    h_lines.append((x1, y1, x2, y2))
                elif 75 < angle < 105:
                    v_lines.append((x1, y1, x2, y2))
    
    return h_lines, v_lines


def find_court_corners_from_lines(h_lines, v_lines, frame_shape):
    """
    从检测到的线找到场地角点
    """
    h, w = frame_shape[:2]
    
    # 如果线不够，用默认
    if len(h_lines) < 2 or len(v_lines) < 2:
        # 使用透视比例（基于典型羽毛球视频）
        aspect = 13.4 / 6.1  # 长宽比
        
        # 假设网在中间
        mid_y = h // 2
        
        # 估算场地边界（透视效果：远处窄，近处宽）
        top_width = int(w * 0.50)  # 远处
        bottom_width = int(w * 0.85)  # 近处
        
        top_center_x = w // 2
        bottom_center_x = w // 2
        
        corners = {
            'top_left': (top_center_x - top_width//2, int(h * 0.12)),
            'top_right': (top_center_x + top_width//2, int(h * 0.12)),
            'bottom_left': (bottom_center_x - bottom_width//2, int(h * 0.88)),
            'bottom_right': (bottom_center_x + bottom_width//2, int(h * 0.88))
        }
        return corners
    
    # 使用线来估计角点
    # 取水平线的中点
    h_mid_y = [(y1 + y2) / 2 for (x1, y1, x2, y2) in h_lines]
    v_mid_x = [(x1 + x2) / 2 for (x1, y1, x2, y2) in v_lines]
    
    h_mid_y.sort()
    v_mid_x.sort()
    
    # 取最外侧的线
    top_y = int(h_mid_y[1] if len(h_mid_y) > 1 else h * 0.15)
    bottom_y = int(h_mid_y[-2] if len(h_mid_y) > 1 else h * 0.85)
    left_x = int(v_mid_x[1] if len(v_mid_x) > 1 else w * 0.15)
    right_x = int(v_mid_x[-2] if len(v_mid_x) > 1 else w * 0.85)
    
    corners = {
        'top_left': (left_x, top_y),
        'top_right': (right_x, top_y),
        'bottom_left': (left_x, bottom_y),
        'bottom_right': (right_x, bottom_y)
    }
    
    return corners


def draw_court_improved(frame, corners, in_court, out_court):
    """
    绘制改进的场地
    """
    debug = frame.copy()
    h, w = frame.shape[:2]
    
    # 填充场地（透视渐变效果）
    pts = np.array([
        corners['top_left'], corners['top_right'],
        corners['bottom_right'], corners['bottom_left']
    ], np.int32)
    
    # 半透明绿色填充
    overlay = debug.copy()
    cv2.fillPoly(overlay, [pts], (0, 200, 0))
    cv2.addWeighted(overlay, 0.2, debug, 0.8, 0, debug)
    
    # 粗边框
    cv2.polylines(debug, [pts], True, (0, 255, 0), 4)
    
    # 画网的位置
    net_y = (corners['top_left'][1] + corners['bottom_left'][1]) // 2
    cv2.line(debug, (corners['top_left'][0], net_y), 
             (corners['top_right'][0], net_y), (255, 0, 255), 3)
    cv2.putText(debug, "NET", (w//2 - 30, net_y - 10), 
               cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 255), 2)
    
    # 画角点
    for name, (x, y) in corners.items():
        cv2.circle(debug, (int(x), int(y)), 12, (0, 255, 255), -1)
    
    # 画场内球员
    h, w = frame.shape[:2]
    for side, data in in_court.items():
        foot = data['foot']
        color = (0, 255, 0)  # 绿色
        cv2.circle(debug, (int(foot[0]), int(foot[1])), 15, color, -1)
        cv2.putText(debug, f"P{side} IN", (int(foot[0])+20, int(foot[1])), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.8, color, 2)
        
        # 画骨架
        kpts = data['keypoints']
        for x, y in kpts:
            if x > 0 and y > 0:
                cv2.circle(debug, (int(x), int(y)), 5, color, -1)
    
    # 画场外球员
    for side, data in out_court.items():
        foot = data['foot']
        color = (0, 0, 255)  # 红色
        cv2.circle(debug, (int(foot[0]), int(foot[1])), 15, color, -1)
        cv2.putText(debug, f"P{side} OUT", (int(foot[0])+20, int(foot[1])), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.8, color, 2)
    
    return debug


if __name__ == "__main__":
    video_path = Path.home() / "Desktop" / "badminton_sample.mp4"
    if not video_path.exists():
        video_path = Path.home() / "Desktop" / "badminton_hd.mp4"
    
    cap = cv2.VideoCapture(str(video_path))
    model = YOLO("yolov8n-pose.pt")
    
    saved = 0
    for i in range(5):
        ret, frame = cap.read()
        if not ret:
            break
        
        h, w = frame.shape[:2]
        
        # 检测场地线
        h_lines, v_lines = detect_white_lines_improved(frame, debug=False)
        
        # 估计角点
        corners = find_court_corners_from_lines(h_lines, v_lines, frame.shape)
        
        # 检测球员
        results = model(frame, verbose=False)
        
        in_court = {}
        out_court = {}
        
        if results and len(results) > 0 and results[0].keypoints is not None:
            kpts_all = results[0].keypoints.xy.cpu().numpy()
            
            for person_kpts in kpts_all:
                # 获取脚点
                if len(person_kpts) > 16:
                    left_ankle = person_kpts[15]
                    right_ankle = person_kpts[16]
                    valid = []
                    if left_ankle[0] > 0: valid.append(left_ankle)
                    if right_ankle[0] > 0: valid.append(right_ankle)
                    
                    if valid:
                        foot = (sum(p[0] for p in valid)/len(valid), 
                               sum(p[1] for p in valid)/len(valid))
                    else:
                        foot = None
                else:
                    foot = None
                
                if foot:
                    # 检查是否在场地内
                    pts = np.array([
                        corners['top_left'], corners['top_right'],
                        corners['bottom_right'], corners['bottom_left']
                    ], np.float32)
                    
                    inside = cv2.pointPolygonTest(pts, foot, False) >= 0
                    
                    if foot[1] < h * 0.5:
                        side = 'X'
                    else:
                        side = 'Y'
                    
                    if inside:
                        in_court[side] = {'foot': foot, 'keypoints': person_kpts}
                    else:
                        out_court[side] = {'foot': foot, 'keypoints': person_kpts}
        
        # 绘制
        debug_frame = draw_court_improved(frame, corners, in_court, out_court)
        cv2.imwrite(f"reports/court_v4_demo_{saved}.jpg", debug_frame)
        saved += 1
        
        print(f"Frame {i}: IN={list(in_court.keys())}, OUT={list(out_court.keys())}")
    
    cap.release()
    print(f"Done! {saved} frames saved")
