# detector.py
import cv2
from ultralytics import YOLO

def detect_and_filter_frames(video_path, conf_threshold=0.15, margin_ratio=0.10, top_margin_ratio=0.02):
    """
    yolov8m-pose (Mediumモデル) を使用して高精度に骨格推定
    """
    # ★ 高精度な Medium モデルに変更
    model = YOLO('yolov8m-pose.pt')
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
        
        results = model.predict(frame, classes=[0], conf=conf_threshold, verbose=False)
        
        detections = []
        if len(results[0].boxes) > 0:
            boxes = results[0].boxes.xyxy.cpu().numpy()
            confs = results[0].boxes.conf.cpu().numpy()
            
            has_kpts = results[0].keypoints is not None
            if has_kpts:
                kpts_xy = results[0].keypoints.xy.cpu().numpy()
                kpts_conf = results[0].keypoints.conf.cpu().numpy() if results[0].keypoints.conf is not None else None
            else:
                kpts_xy, kpts_conf = [], None
            
            for i, (bbox, conf) in enumerate(zip(boxes, confs)):
                x1, y1, x2, y2 = bbox
                x_center = (x1 + x2) / 2.0
                y_center = (y1 + y2) / 2.0
                box_height = y2 - y1
                
                is_edge = not (left_bound <= x_center <= right_bound) or (y_center < top_bound)
                
                # 関節データ [x, y, conf] のリストを作成
                kpts_data = []
                if i < len(kpts_xy):
                    for j in range(len(kpts_xy[i])):
                        x, y = kpts_xy[i][j]
                        c = kpts_conf[i][j] if kpts_conf is not None else 1.0
                        kpts_data.append([float(x), float(y), float(c)])
                
                detections.append({
                    'bbox': bbox.tolist(),
                    'center': [x_center, y_center],
                    'box_height': float(box_height),
                    'conf': float(conf),
                    'is_edge': is_edge,
                    'keypoints': kpts_data
                })
                
        raw_frames.append({
            'frame_idx': frame_idx, 
            'detections': detections,
            'frame_height': frame_height
        })
        frame_idx += 1
        
    cap.release()
    return raw_frames

