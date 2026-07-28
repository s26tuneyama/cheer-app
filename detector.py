# detector.py
import cv2
from ultralytics import YOLO

def detect_and_filter_frames(video_path, conf_threshold=0.15, margin_ratio=0.10, top_margin_ratio=0.02, frame_skip=2):
    """
    CPU環境向けに最適化したYOLO高速検出処理
    - yolov8n-pose (Nanoモデル) を使用して爆速化
    - frame_skip=2 で2コマに1コマ処理して処理時間を半減
    """
    # 最軽量のNanoモデルを採用 (CPUで超高速に動作)
    model = YOLO('yolov8n-pose.pt')
    cap = cv2.VideoCapture(video_path)
    
    frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    
    left_bound = frame_width * margin_ratio
    right_bound = frame_width * (1.0 - margin_ratio)
    top_bound = frame_height * top_margin_ratio
    
    raw_frames = []
    frame_idx = 0
    
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret: 
            break
        
        # 間引き処理（指定コマ数ごとに解析を実施）
        if frame_idx % frame_skip == 0:
            # imgsz=480 でCPUの負担を大幅低減
            results = model.predict(frame, classes=[0], conf=conf_threshold, imgsz=480, verbose=False)
            
            detections = []
            if len(results[0].boxes) > 0:
                boxes = results[0].boxes.xyxy.cpu().numpy()
                confs = results[0].boxes.conf.cpu().numpy()
                
                for bbox, conf in zip(boxes, confs):
                    x1, y1, x2, y2 = bbox
                    x_center = (x1 + x2) / 2.0
                    y_center = (y1 + y2) / 2.0
                    box_height = y2 - y1
                    
                    is_edge = not (left_bound <= x_center <= right_bound) or (y_center < top_bound)
                    
                    detections.append({
                        'bbox': bbox.tolist(),
                        'center': [x_center, y_center],
                        'box_height': float(box_height),
                        'conf': float(conf),
                        'is_edge': is_edge
                    })
                    
            raw_frames.append({
                'frame_idx': frame_idx, 
                'detections': detections,
                'frame_height': frame_height
            })
            
        frame_idx += 1
        
    cap.release()
    return raw_frames

