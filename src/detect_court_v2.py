"""
Badminton Court Detection - 真实场地检测 + 透视变换
检测真实场地边界线，映射到实际场地坐标
"""
import cv2
import numpy as np
from pathlib import Path
import json


def detect_white_lines(frame, debug=False):
    """
    通过颜色过滤检测羽毛球场地白色边界线
    """
    h, w = frame.shape[:2]
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    
    # 方法1: 颜色过滤 - 找出白色/浅色区域（羽毛球场线是白色的）
    # 在HSV空间过滤白色
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
    lower_white = np.array([0, 0, 200])
    upper_white = np.array([180, 50, 255])
    white_mask = cv2.inRange(hsv, lower_white, upper_white)
    
    # 形态学操作去噪
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3,3))
    white_mask = cv2.morphologyEx(white_mask, cv2.MORPH_CLOSE, kernel)
    white_mask = cv2.morphologyEx(white_mask, cv2.MORPH_OPEN, kernel)
    
    # 边缘检测
    edges = cv2.Canny(white_mask, 50, 150)
    
    # 霍夫变换找直线
    lines = cv2.HoughLinesP(edges, 1, np.pi/180, 80, minLineLength=50, maxLineGap=10)
    
    h_lines = []  # 水平线
    v_lines = []  # 垂直线
    
    if lines is not None:
        for line in lines:
            x1, y1, x2, y2 = line[0]
            angle = np.abs(np.arctan2(y2-y1, x2-x1) * 180 / np.pi)
            length = np.sqrt((x2-x1)**2 + (y2-y1)**2)
            
            if length > 100:  # 只保留较长的线
                if angle < 15 or angle > 165:  # 水平线
                    h_lines.append((y1 + y2) / 2)  # y坐标
                elif 75 < angle < 105:  # 垂直线
                    v_lines.append((x1 + x2) / 2)  # x坐标
    
    return h_lines, v_lines, white_mask


def estimate_court_corners(h_lines, v_lines, frame_shape):
    """
    根据检测到的线估计场地四个角
    """
    h, w = frame_shape[:2]
    
    if len(h_lines) >= 2 and len(v_lines) >= 2:
        # 有足够的线，取最外侧的四条
        top_y = sorted(h_lines)[:2][-1]  # 取偏下的两条水平线的中间
        bottom_y = sorted(h_lines)[-2:][0]  # 取偏上的两条水平线的中间
        left_x = sorted(v_lines)[:2][-1]
        right_x = sorted(v_lines)[-2:][0]
    else:
        # 线不够，用默认比例估计（双打场地标准比例）
        # 羽毛球场地: 13.4m x 6.1m (长13.4m, 宽6.1m)
        # 画面中大致比例
        top_y = int(h * 0.15)
        bottom_y = int(h * 0.85)
        left_x = int(w * 0.12)
        right_x = int(w * 0.88)
    
    return {
        'top_left': (left_x, top_y),
        'top_right': (right_x, top_y),
        'bottom_left': (left_x, bottom_y),
        'bottom_right': (right_x, bottom_y)
    }


def get_court_mask(frame, corners):
    """
    创建场地mask，只保留场地内的区域
    """
    pts = np.array([
        corners['top_left'],
        corners['top_right'],
        corners['bottom_right'],
        corners['bottom_left']
    ], np.int32)
    
    mask = np.zeros(frame.shape[:2], np.uint8)
    cv2.fillPoly(mask, [pts], 255)
    
    return mask


def filter_players_in_court(players, corners):
    """
    只保留在真实场地内的球员
    使用射线法判断点是否在四边形内
    """
    def point_in_polygon(point, polygon):
        """射线法判断点是否在多边形内"""
        x, y = point
        n = len(polygon)
        inside = False
        
        j = n - 1
        for i in range(n):
            xi, yi = polygon[i]
            xj, yj = polygon[j]
            
            if ((yi > y) != (yj > y)) and (x < (xj - xi) * (y - yi) / (yj - yi) + xi):
                inside = not inside
            j = i
        
        return inside
    
    polygon = [
        corners['top_left'],
        corners['top_right'],
        corners['bottom_right'],
        corners['bottom_left']
    ]
    
    filtered = {}
    for side, p in players.items():
        if p is None:
            continue
        center = p.get('center')
        if center and point_in_polygon(center, polygon):
            filtered[side] = p
        # else: player is outside court, filter out
    
    return filtered


def detect_court_and_filter(frame, pose_results, debug=False):
    """
    主函数：检测场地 + 过滤场外球员
    """
    # 1. 检测场地线
    h_lines, v_lines, white_mask = detect_white_lines(frame, debug)
    
    # 2. 估计场地角点
    corners = estimate_court_corners(h_lines, v_lines, frame.shape)
    
    # 3. 获取场地mask
    court_mask = get_court_mask(frame, corners)
    
    # 4. 过滤球员
    players = {}
    if pose_results is not None and len(pose_results.keypoints) > 0:
        kpts = pose_results.keypoints.xy.cpu().numpy()
        
        for person_kpts in kpts:
            # 计算人的中心点（用躯干关键点）
            # keypoints: 0=nose, 5=left_shoulder, 6=right_shoulder, 11=left_hip, 12=right_hip
            shoulders = []
            hips = []
            
            if len(person_kpts) > 6:
                if person_kpts[5][0] > 0: shoulders.append(person_kpts[5])
                if person_kpts[6][0] > 0: shoulders.append(person_kpts[6])
                if person_kpts[11][0] > 0: hips.append(person_kpts[11])
                if person_kpts[12][0] > 0: hips.append(person_kpts[12])
            
            if shoulders and hips:
                center_x = (sum(p[0] for p in shoulders) / len(shoulders) + sum(p[0] for p in hips) / len(hips)) / 2
                center_y = (sum(p[1] for p in shoulders) / len(shoulders) + sum(p[1] for p in hips) / len(hips)) / 2
                
                # 判断是否在场内
                is_inside = cv2.pointPolygonTest(
                    np.array([
                        corners['top_left'], corners['top_right'],
                        corners['bottom_right'], corners['bottom_left']
                    ], np.float32),
                    (center_x, center_y),
                    False
                ) >= 0
                
                if is_inside:
                    # 根据y位置判断是X还是Y
                    h = frame.shape[0]
                    if center_y < h * 0.5:
                        players['X'] = {'center': [center_x, center_y], 'kpts': person_kpts}
                    else:
                        players['Y'] = {'center': [center_x, center_y], 'kpts': person_kpts}
    
    return players, corners, court_mask


def draw_court_debug(frame, corners, players=None):
    """
    绘制调试信息
    """
    debug = frame.copy()
    
    # 画场地边界
    pts = np.array([
        corners['top_left'], corners['top_right'],
        corners['bottom_right'], corners['bottom_left']
    ], np.int32)
    cv2.polylines(debug, [pts], True, (0, 255, 0), 3)
    
    # 画场地角点
    for name, (x, y) in corners.items():
        cv2.circle(debug, (int(x), int(y)), 8, (0, 255, 255), -1)
        cv2.putText(debug, name, (int(x)+10, int(y)), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0,255,255), 2)
    
    # 画球员
    if players:
        for side, p in players.items():
            center = p.get('center')
            if center:
                color = (0, 255, 0) if side == 'X' else (255, 0, 0)
                cv2.circle(debug, (int(center[0]), int(center[1])), 12, color, -1)
                cv2.putText(debug, f"Player {side}", (int(center[0])+15, int(center[1])), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.8, color, 2)
    
    return debug


if __name__ == "__main__":
    from ultralytics import YOLO
    
    video_path = Path.home() / "Desktop" / "badminton_sample.mp4"
    if not video_path.exists():
        video_path = Path.home() / "Desktop" / "badminton_hd.mp4"
    
    cap = cv2.VideoCapture(str(video_path))
    model = YOLO("yolov8n-pose.pt")
    
    # 读取几帧测试
    for i in range(3):
        ret, frame = cap.read()
        if not ret:
            break
        
        # 运行检测
        results = model(frame, verbose=False)
        pose = results[0] if results else None
        
        players, corners, mask = detect_court_and_filter(frame, pose, debug=True)
        
        # 画调试图
        debug_frame = draw_court_debug(frame, corners, players)
        
        # 保存
        cv2.imwrite(f"reports/court_realistic_demo_{i}.jpg", debug_frame)
        print(f"Frame {i}: Detected {len(players)} players in court: {list(players.keys())}")
    
    cap.release()
    print("Done! Check reports/court_realistic_demo_*.jpg")
