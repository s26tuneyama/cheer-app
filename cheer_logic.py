# cheer_logic.py
import numpy as np

def analyze_cheer_flyer_descent(raw_frames, max_jump_distance=150.0):
    """
    チア専門ロジック：
    ピーク（最高到達点）を検出し、ID無視の空間バトンタッチで下降を追跡する
    """
    # Step 1: 端のノイズを除外した上で、最高到達点（ピーク）を探索
    peak_frame_idx = -1
    min_y = float('inf')
    peak_detection = None

    for frame_info in raw_frames:
        f_idx = frame_info['frame_idx']
        for det in frame_info['detections']:
            # 画面端に固定されている観客等はピーク判定から除外
            if det['is_edge']: 
                continue
                
            y_center = det['center'][1]
            if y_center < min_y:
                min_y = y_center
                peak_frame_idx = f_idx
                peak_detection = det

    if peak_detection is None:
        return []

    # Step 2: ピークから近接バトンタッチで追跡
    trajectory = []
    current_target = peak_detection.copy()
    current_target['frame_idx'] = peak_frame_idx
    current_target['valid_for_scoring'] = current_target['conf'] >= 0.40
    trajectory.append(current_target)

    prev_center = current_target['center']

    for frame_info in raw_frames[peak_frame_idx + 1:]:
        f_idx = frame_info['frame_idx']
        # 画面端以外の検出だけに絞る（端の観客への誤バトンタッチを防止）
        candidates = [d for d in frame_info['detections'] if not d['is_edge']]
        
        if not candidates: break

        best_candidate = None
        min_dist = float('inf')

        for cand in candidates:
            dist = np.sqrt(
                (cand['center'][0] - prev_center[0])**2 + 
                (cand['center'][1] - prev_center[1])**2
            )
            if dist < min_dist:
                min_dist = dist
                best_candidate = cand

        if best_candidate and min_dist <= max_jump_distance:
            tracked_flyer = best_candidate.copy()
            tracked_flyer['frame_idx'] = f_idx
            tracked_flyer['valid_for_scoring'] = tracked_flyer['conf'] >= 0.40
            
            trajectory.append(tracked_flyer)
            prev_center = tracked_flyer['center']
        else:
            break # キャッチ完了（移動停止）

    return trajectory
