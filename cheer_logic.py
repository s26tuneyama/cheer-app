# cheer_logic.py
import numpy as np

def calculate_angle(a, b, c):
    """
    bを中心とした 3点 (a, b, c) のなす角度（度数法 0〜180度）を計算
    """
    a, b, c = np.array(a), np.array(b), np.array(c)
    if np.all(a == 0) or np.all(b == 0) or np.all(c == 0):
        return None  # 座標が取れていない場合
    
    ba = a - b
    bc = c - b
    
    cosine_angle = np.dot(ba, bc) / (np.linalg.norm(ba) * np.linalg.norm(bc) + 1e-6)
    cosine_angle = np.clip(cosine_angle, -1.0, 1.0)
    angle = np.arccos(cosine_angle)
    return np.degrees(angle)


def extract_cheer_angles(keypoints):
    """
    17個の関節からチアに必要な主要角度を計算
    5:左肩, 6:右肩, 11:左腰, 12:右腰, 13:左膝, 14:右膝, 15:左足首, 16:右足首
    """
    if not keypoints or len(keypoints) < 17:
        return {'body_angle': None, 'split_angle': None}
    
    # 中点の計算（両肩・両腰・両膝）
    shoulder_mid = [(keypoints[5][0] + keypoints[6][0]) / 2, (keypoints[5][1] + keypoints[6][1]) / 2]
    hip_mid = [(keypoints[11][0] + keypoints[12][0]) / 2, (keypoints[11][1] + keypoints[12][1]) / 2]
    knee_mid = [(keypoints[13][0] + keypoints[14][0]) / 2, (keypoints[13][1] + keypoints[14][1]) / 2]
    
    left_ankle = keypoints[15]
    right_ankle = keypoints[16]
    
    # 1. 体幹角度 (肩 - 腰 - 膝)
    body_angle = calculate_angle(shoulder_mid, hip_mid, knee_mid)
    
    # 2. 開脚角度 (左足首 - 腰 - 右足首)
    split_angle = calculate_angle(left_ankle, hip_mid, right_ankle)
    
    return {
        'body_angle': round(body_angle, 1) if body_angle else None,
        'split_angle': round(split_angle, 1) if split_angle else None
    }


def analyze_cheer_flyer_descent(raw_frames, max_jump_distance=150.0):
    peak_frame_idx = -1
    min_y = float('inf')
    peak_detection = None

    for frame_info in raw_frames:
        f_idx = frame_info['frame_idx']
        for det in frame_info['detections']:
            if det['is_edge']: continue
            y_center = det['center'][1]
            if y_center < min_y:
                min_y = y_center
                peak_frame_idx = f_idx
                peak_detection = det

    if peak_detection is None:
        return []

    trajectory = []
    current_target = peak_detection.copy()
    current_target['frame_idx'] = peak_frame_idx
    current_target['valid_for_scoring'] = current_target['conf'] >= 0.40
    
    # 角度計算を追加
    angles = extract_cheer_angles(current_target.get('keypoints'))
    current_target['body_angle'] = angles['body_angle']
    current_target['split_angle'] = angles['split_angle']
    
    trajectory.append(current_target)
    prev_center = current_target['center']

    for frame_info in raw_frames[peak_frame_idx + 1:]:
        f_idx = frame_info['frame_idx']
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
            
            # 角度計算を追加
            angles = extract_cheer_angles(tracked_flyer.get('keypoints'))
            tracked_flyer['body_angle'] = angles['body_angle']
            tracked_flyer['split_angle'] = angles['split_angle']
            
            trajectory.append(tracked_flyer)
            prev_center = tracked_flyer['center']
        else:
            break

    return trajectory

