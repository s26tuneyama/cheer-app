# detector.py
import cv2
from ultralytics import YOLO

def detect_and_filter_frames(video_path, conf_threshold=0.10, margin_ratio=0.15):
    """
    yolov8n-pose.pt を使用し、BBox・信頼度・関節座標(17点)をまとめて取得
    """
    model = YOLO('yolov8n-pose.pt')  # 👈 骨格推定モデルに変更！
    cap = cv2.VideoCapture(video_path)
    frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    
    left_bound = frame_width * margin_ratio
    right_bound = frame_width * (1.0 - margin_ratio)
    
    raw_frames = []
    frame_idx = 0
    
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret: 
            break
        
        results = model.predict(frame, classes=[0], conf=conf_threshold, verbose=False)
        
        detections = []
        if len(results[0].boxes) > 0:
            boxes = results[0].boxes.xyxy.cpu().numpy()
            confs = results[0].boxes.conf.cpu().numpy()
            
            # 関節座標の取得 (NumPy配列)
            has_kpts = results[0].keypoints is not None
            keypoints_data = results[0].keypoints.xy.cpu().numpy() if has_kpts else []
            
            for i, (bbox, conf) in enumerate(zip(boxes, confs)):
                x_center = (bbox[0] + bbox[2]) / 2.0
                y_center = (bbox[1] + bbox[3]) / 2.0
                
                is_edge = not (left_bound <= x_center <= right_bound)
                
                kpts = keypoints_data[i].tolist() if i < len(keypoints_data) else []
                
                detections.append({
                    'bbox': bbox.tolist(),
                    'center': [x_center, y_center],
                    'conf': float(conf),
                    'is_edge': is_edge,
                    'keypoints': kpts  # 👈 17個の関節座標 [[x, y], ...]
                })
                
        raw_frames.append({'frame_idx': frame_idx, 'detections': detections})
        frame_idx += 1
        
    cap.release()
    return raw_frames

