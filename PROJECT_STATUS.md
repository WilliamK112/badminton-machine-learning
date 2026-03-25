# Badminton AI - 项目状态

## 训练好的模型

| 模型 | 用途 | mAP50 | mAP50-95 | Precision | Recall |
|------|------|-------|----------|-----------|--------|
| shuttle_v2 | 羽毛球检测 | 67.7% | 24.7% | 67.7% | 65.9% |
| shuttle_v22 | 羽毛球检测 | 65.8% | 21.8% | 62.6% | 64.2% |
| court_detector_v2 | 场地检测 | 94.9%* | 84.1%* | - | - |
| yolo11n-pose | 运动员骨架 | - | - | - | - |

*注: court模型用已知数据训练，实际新视频检测效果待验证

## 检测结果示例
- frame_00000.jpg: 1 shuttle
- frame_02441.jpg: 4 shuttles
- frame_03662.jpg: 3 shuttles

## 组合检测
- Court: 使用已知corner点
- Shuttle: shuttle_v2模型
- Players: yolo11n-pose (2人)

## 特征数据
- body_features_v14.csv
- combined_features_v16.csv  
- motion_quant_v1.csv
- rally_labels_v13.csv

## 可视化报告
- `reports/rally_timeline.png` - Rally 时间线
- `reports/landing_heatmap.png` - 羽毛球落点热图
- `reports/win_prob_timeline.png` - 胜负概率时间线

## 下一步
1. 改进场地检测模型（需要更多标注数据）
2. 收集更多 rally outcomes 数据（当前只有 9 个落点，8:1 不平衡）
