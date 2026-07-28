# detector.py
import cv2
from ultralytics import YOLO

def detect_and_filter_frames(video_path, conf_threshold=0.10, margin_ratio=0.15):
    """
    1. YOLO(conf=0.10)で低閾値検知
    2. 動画を通じて「ずっと画面端にしかいなかったノイズ」を軽量化のために排除する
    """
    model = YOLO('yolov8n.pt')
    cap = cv2.VideoCapture(video_path)
    frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    
    left_bound = frame_width * margin_ratio
    right_bound = frame_width * (1.0 - margin_ratio)
    
    raw_frames = []
    central_active_detections = []
    
    frame_idx = 0
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret: break
        
        results = model.predict(frame, classes=[0], conf=conf_threshold, verbose=False)
        
        detections = []
        if len(results[0].boxes) > 0:
            boxes = results[0].boxes.xyxy.cpu().numpy()
            confs = results[0].boxes.conf.cpu().numpy()
            
            for bbox, conf in zip(boxes, confs):
                x_center = (bbox[0] + bbox[2]) / 2.0
                y_center = (bbox[1] + bbox[3]) / 2.0
                
                # 【端の排除ロジック】
                # 一度も画面中央エリア（左右15%より内側）に入らない背景ノイズは最初から対象外
                is_edge = not (left_bound <= x_center <= right_bound)
                
                detections.append({
                    'bbox': bbox.tolist(),
                    'center': [x_center, y_center],
                    'conf': float(conf),
                    'is_edge': is_edge
                })
                
        raw_frames.append({'frame_idx': frame_idx, 'detections': detections})
        frame_idx += 1
        
    cap.release()
    return raw_frames
