"""
Badminton Court & Net Detection
使用 OpenCV 检测羽毛球场地边界和网，聚焦比赛区域
"""
import cv2
import numpy as np
from pathlib import Path


def detect_court_lines(frame, debug=False):
    """
    通过颜色和边缘检测羽毛球场地白线
    返回: court_mask, homography_matrix
    """
    h, w = frame.shape[:2]
    
    # 1. 转换为灰度
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    
    # 2. 高斯模糊 + Canny 边缘检测
    blurred = cv2.GaussianBlur(gray, (5, 5), 0)
    edges = cv2.Canny(blurred, 50, 150)
    
    # 3. 霍夫变换检测直线
    lines = cv2.HoughLinesP(edges, 1, np.pi/180, 100, minLineLength=100, maxLineGap=10)
    
    # 4. 过滤出接近水平和垂直的线（场地线通常是水平/垂直的）
    h_lines, v_lines = [], []
    if lines is not None:
        for line in lines:
            x1, y1, x2, y2 = line[0]
            angle = np.abs(np.arctan2(y2-y1, x2-x1) * 180 / np.pi)
            if angle < 10 or angle > 170:  # ~水平
                h_lines.append(line[0])
            elif 80 < angle < 100:  # ~垂直
                v_lines.append(line[0])
    
    # 5. 创建场地 mask（简单版：假设场地在画面中心区域）
    # 实际可用更复杂的线框检测
    mask = np.zeros_like(edges)
    
    # 默认：取画面中间 60% 区域作为场地
    y1, y2 = int(h * 0.2), int(h * 0.85)
    x1, x2 = int(w * 0.15), int(w * 0.85)
    mask[y1:y2, x1:x2] = 255
    
    return mask, (x1, y1, x2, y2)


def detect_net(frame, debug=False):
    """
    检测羽毛球网（通常是一条水平细线）
    返回: net_y (网的高度位置)
    """
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    
    # 网通常是画面中间的一条细水平线
    h, w = frame.shape[:2]
    mid_y = int(h * 0.5)  # 假设网在画面中间
    
    # 在中间区域检测细水平线
    search_region = gray[mid_y-20:mid_y+20, :]
    blurred = cv2.GaussianBlur(search_region, (3, 3), 0)
    edges = cv2.Canny(blurred, 30, 90)
    
    # 霍夫变换找水平线
    lines = cv2.HoughLinesP(edges, 1, np.pi/180, 50, minLineLength=w*0.3, maxLineGap=5)
    
    if lines is not None:
        for line in lines:
            x1, y1, x2, y2 = line[0]
            angle = np.abs(np.arctan2(y2-y1, x2-x1) * 180 / np.pi)
            if angle < 15:  # 接近水平
                net_y = mid_y + (y1 + y2) // 2 - 20
                return net_y
    
    return mid_y  # 默认


def crop_court_region(frame, debug=False):
    """
    裁剪到羽毛球场地区域
    返回: cropped_frame, court_box
    """
    h, w = frame.shape[:2]
    
    # 检测网的位置
    net_y = detect_net(frame, debug)
    
    # 场地区域：网上方 + 网下方（双打场地）
    # 上半区（球员 X）
    top_court = frame[:net_y, :]
    # 下半区（球员 Y）  
    bottom_court = frame[net_y:, :]
    
    # 可以分别处理上下两个半区
    return top_court, bottom_court, net_y


def filter_players_in_court(players, court_box):
    """
    只保留在场地范围内的球员
    court_box: (x1, y1, x2, y2)
    """
    x1, y1, x2, y2 = court_box
    filtered = {}
    
    for side, p in players.items():
        if p is None:
            continue
        center = p.get('center')
        if center and x1 <= center[0] <= x2 and y1 <= center[1] <= y2:
            filtered[side] = p
    
    return filtered


if __name__ == "__main__":
    import sys
    
    video_path = Path.home() / "Desktop" / "badminton_sample.mp4"
    if not video_path.exists():
        video_path = Path.home() / "Desktop" / "badminton_hd.mp4"
    
    cap = cv2.VideoCapture(str(video_path))
    
    # 读取第一帧测试
    ret, frame = cap.read()
    if ret:
        # 检测
        net_y = detect_net(frame, debug=True)
        print(f"Detected net at y={net_y}")
        
        # 可视化
        debug_frame = frame.copy()
        cv2.line(debug_frame, (0, net_y), (frame.shape[1], net_y), (0, 255, 0), 2)
        cv2.putText(debug_frame, f"Net: y={net_y}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0,255,0), 2)
        
        cv2.imwrite("reports/court_detection_demo.jpg", debug_frame)
        print("Saved: reports/court_detection_demo.jpg")
        
        # 打开看看
        import os
        os.system("open reports/court_detection_demo.jpg")
    
    cap.release()
