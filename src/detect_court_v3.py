"""
Badminton Court Detection V3 - 透视校正 + 脚点检测
基于真实场地线的透视校正，只标注脚点在场地内的人
"""
import cv2
import numpy as np
from pathlib import Path
from ultralytics import YOLO


# 羽毛球场地标准尺寸 (米)
COURT_WIDTH = 6.1   # 双打场地宽 6.1m
COURT_LENGTH = 13.4 # 场地长 13.4m

# 视频帧的原始尺寸
ORIGINAL_W = 1920
ORIGINAL_H = 1080


def detect_court_lines_advanced(frame, debug=False):
    """
    高级场地线检测 - 使用颜色过滤 + 边缘检测
    """
    h, w = frame.shape[:2]
    
    # 1. 检测白色区域（场地线是白色的）
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
    
    # 白色范围（羽毛球场地线）
    lower_white = np.array([0, 0, 180])
    upper_white = np.array([180, 40, 255])
    white_mask = cv2.inRange(hsv, lower_white, upper_white)
    
    # 去噪
    kernel = np.ones((3,3), np.uint8)
    white_mask = cv2.morphologyEx(white_mask, cv2.MORPH_CLOSE, kernel, iterations=2)
    white_mask = cv2.morphologyEx(white_mask, cv2.MORPH_OPEN, kernel, iterations=1)
    
    # 2. 边缘检测
    edges = cv2.Canny(white_mask, 80, 200)
    
    # 3. 霍夫直线检测
    lines = cv2.HoughLinesP(edges, 1, np.pi/180, 100, minLineLength=80, maxLineGap=15)
    
    h_lines = []  # 水平线列表 [(x1,y1,x2,y2), ...]
    v_lines = []  # 垂直线列表
    
    if lines is not None:
        for line in lines:
            x1, y1, x2, y2 = line[0]
            angle = np.abs(np.arctan2(y2-y1, x2-x1) * 180 / np.pi)
            length = np.sqrt((x2-x1)**2 + (y2-y1)**2)
            
            if length > 100:
                if angle < 12 or angle > 168:
                    h_lines.append((y1 + y2) / 2)  # y坐标
                elif 78 < angle < 102:
                    v_lines.append((x1 + x2) / 2)  # x坐标
    
    return h_lines, v_lines, white_mask


def estimate_court_corners_perspective(h_lines, v_lines, frame_shape):
    """
    透视校正的角点估计
    根据检测到的线位置，考虑透视效果
    """
    h, w = frame_shape[:2]
    
    # 如果检测到足够多的线
    if len(h_lines) >= 2 and len(v_lines) >= 2:
        # 取最外侧的四条线
        h_sorted = sorted(h_lines)
        v_sorted = sorted(v_lines)
        
        top_y = int(h_sorted[1])  # 第二条水平线（从上往下）
        bottom_y = int(h_sorted[-2])  # 倒数第二条水平线
        left_x = int(v_sorted[1])  # 第二条垂直线
        right_x = int(v_sorted[-2])  # 倒数第二条垂直线
    else:
        # 使用透视比例估计
        # 透视效果：远处的边更窄
        top_width = int(w * 0.55)
        bottom_width = int(w * 0.85)
        
        left_x = int((w - top_width) // 2)
        right_x = int(left_x + top_width)
        
        top_y = int(h * 0.15)
        bottom_y = int(h * 0.88)
    
    # 四个角点（按顺时针）
    corners = {
        'top_left': (left_x, top_y),
        'top_right': (right_x, top_y),
        'bottom_right': (right_x, bottom_y),
        'bottom_left': (left_x, bottom_y)
    }
    
    return corners


def get_foot_point(keypoints):
    """
    提取人的脚点位置
    使用脚踝关键点 (14, 16) 的中心
    """
    # COCO keypoints: 15=left_ankle, 16=right_ankle
    if len(keypoints) > 16:
        left_ankle = keypoints[15]  # left_ankle
        right_ankle = keypoints[16]  # right_ankle
        
        # 计算脚中心点
        valid = []
        if left_ankle[0] > 0 and left_ankle[1] > 0:
            valid.append(left_ankle)
        if right_ankle[0] > 0 and right_ankle[1] > 0:
            valid.append(right_ankle)
        
        if valid:
            foot_x = sum(p[0] for p in valid) / len(valid)
            foot_y = sum(p[1] for p in valid) / len(valid)
            return (foot_x, foot_y)
    
    # 如果没有脚踝，用躯干底部近似
    if len(keypoints) > 12:
        left_hip = keypoints[11]
        right_hip = keypoints[12]
        
        if left_hip[0] > 0 and right_hip[0] > 0:
            foot_x = (left_hip[0] + right_hip[0]) / 2
            foot_y = (left_hip[1] + right_hip[1]) / 2
            return (foot_x, foot_y)
    
    return None


def is_point_in_court(point, court_corners):
    """
    判断点是否在透视校正后的场地多边形内
    """
    if point is None:
        return False
    
    x, y = point
    
    # 获取四个角点
    pts = np.array([
        court_corners['top_left'],
        court_corners['top_right'],
        court_corners['bottom_right'],
        court_corners['bottom_left']
    ], np.float32)
    
    # 射线法
    result = cv2.pointPolygonTest(pts, (x, y), False)
    return result >= 0  # 在内部或边界上


def draw_perspective_court(frame, corners, players_in_court, players_outside):
    """
    绘制透视场地和球员
    """
    debug = frame.copy()
    
    # 1. 画透视场地边界（四边形）
    pts = np.array([
        corners['top_left'], corners['top_right'],
        corners['bottom_right'], corners['bottom_left']
    ], np.int32)
    
    # 填充场地区域
    overlay = debug.copy()
    cv2.fillPoly(overlay, [pts], (0, 255, 0))  # 绿色填充
    cv2.addWeighted(overlay, 0.15, debug, 0.85, 0, debug)
    
    # 画边界线
    cv2.polylines(debug, [pts], True, (0, 255, 0), 3)
    
    # 2. 画场地内的球员（绿色）
    for side, data in players_in_court.items():
        foot = data['foot']
        kpts = data['keypoints']
        
        if foot:
            cv2.circle(debug, (int(foot[0]), int(foot[1])), 15, (0, 255, 0), -1)
            cv2.putText(debug, f"Player {side} (IN)", 
                       (int(foot[0])+20, int(foot[1])), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)
        
        # 画骨架
        if len(kpts) > 0:
            for x, y in kpts:
                if x > 0 and y > 0:
                    cv2.circle(debug, (int(x), int(y)), 4, (0, 255, 0), -1)
    
    # 3. 画场地外的球员（红色）
    for side, data in players_outside.items():
        foot = data['foot']
        kpts = data['keypoints']
        
        if foot:
            cv2.circle(debug, (int(foot[0]), int(foot[1])), 15, (0, 0, 255), -1)
            cv2.putText(debug, f"Player {side} (OUT)", 
                       (int(foot[0])+20, int(foot[1])), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2)
    
    # 4. 画角点标注
    for name, (x, y) in corners.items():
        cv2.circle(debug, (int(x), int(y)), 10, (0, 255, 255), -1)
        cv2.putText(debug, name, (int(x)+15, int(y)), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 1)
    
    return debug


if __name__ == "__main__":
    # 测试
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
        
        # 1. 检测场地线
        h_lines, v_lines, _ = detect_court_lines_advanced(frame, debug=True)
        
        # 2. 估计透视场地角点
        corners = estimate_court_corners_perspective(h_lines, v_lines, frame.shape)
        
        # 3. 检测球员和脚点
        results = model(frame, verbose=False)
        
        in_court = {}
        out_court = {}
        
        if results and len(results) > 0 and results[0].keypoints is not None:
            kpts_all = results[0].keypoints.xy.cpu().numpy()
            
            for person_idx, person_kpts in enumerate(kpts_all):
                foot_point = get_foot_point(person_kpts)
                
                # 判断脚点是否在场地内
                if foot_point and is_point_in_court(foot_point, corners):
                    # 根据y位置判断是X还是Y
                    if foot_point[1] < frame.shape[0] * 0.5:
                        side = 'X'
                    else:
                        side = 'Y'
                    in_court[side] = {'foot': foot_point, 'keypoints': person_kpts}
                else:
                    # 场外的人
                    if foot_point:
                        if foot_point[1] < frame.shape[0] * 0.5:
                            side = 'X'
                        else:
                            side = 'Y'
                        out_court[side] = {'foot': foot_point, 'keypoints': person_kpts}
        
        # 4. 绘制结果
        debug_frame = draw_perspective_court(frame, corners, in_court, out_court)
        
        # 保存
        cv2.imwrite(f"reports/court_v3_demo_{saved}.jpg", debug_frame)
        saved += 1
        
        print(f"Frame {i}: IN={list(in_court.keys())}, OUT={list(out_court.keys())}")
    
    cap.release()
    print(f"Done! Saved {saved} frames to reports/court_v3_demo_*.jpg")
