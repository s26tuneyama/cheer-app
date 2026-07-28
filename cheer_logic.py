# cheer_logic.py
import numpy as np
import cv2

SKELETON_CONNECTIONS = [
    (5, 6), (5, 7), (7, 9), (6, 8), (8, 10),
    (5, 11), (6, 12), (11, 12), (11, 13), (13, 15), (12, 14), (14, 16)
]

def calculate_angle(a, b, c):
    a, b, c = np.array(a[:2]), np.array(b[:2]), np.array(c[:2])
    if np.all(a == 0) or np.all(b == 0) or np.all(c == 0):
        return None
    ba, bc = a - b, c - b
    norm_ba, norm_bc = np.linalg.norm(ba), np.linalg.norm(bc)
    if norm_ba == 0 or norm_bc == 0:
        return None
    cosine_angle = np.clip(np.dot(ba, bc) / (norm_ba * norm_bc), -1.0, 1.0)
    return float(np.degrees(np.arccos(cosine_angle)))

def extract_cheer_angles(keypoints, kpt_conf_thresh=0.25):
    if not keypoints or len(keypoints) < 17:
        return {'body_angle': None, 'split_angle': None}
    
    def valid_pt(pt_idx):
        if pt_idx >= len(keypoints): return False
        pt = keypoints[pt_idx]
        conf = pt[2] if len(pt) > 2 else 1.0
        return (pt[0] > 0 and pt[1] > 0 and conf >= kpt_conf_thresh)

    body_angle = None
    if valid_pt(5) and valid_pt(6) and valid_pt(11) and valid_pt(12):
        shoulder_mid = [(keypoints[5][0] + keypoints[6][0]) / 2, (keypoints[5][1] + keypoints[6][1]) / 2]
        hip_mid = [(keypoints[11][0] + keypoints[12][0]) / 2, (keypoints[11][1] + keypoints[12][1]) / 2]
        
        knee_pt = None
        if valid_pt(13) and valid_pt(14):
            knee_pt = [(keypoints[13][0] + keypoints[14][0]) / 2, (keypoints[13][1] + keypoints[14][1]) / 2]
        elif valid_pt(13): knee_pt = keypoints[13]
        elif valid_pt(14): knee_pt = keypoints[14]

        if knee_pt is not None:
            body_angle = calculate_angle(shoulder_mid, hip_mid, knee_pt)

    split_angle = None
    if valid_pt(11) and valid_pt(12) and valid_pt(15) and valid_pt(16):
        hip_mid = [(keypoints[11][0] + keypoints[12][0]) / 2, (keypoints[11][1] + keypoints[12][1]) / 2]
        split_angle = calculate_angle(keypoints[15], hip_mid, keypoints[16])

    return {
        'body_angle': round(body_angle, 1) if body_angle is not None else None,
        'split_angle': round(split_angle, 1) if split_angle is not None else None
    }

def smooth_angles_median(trajectory, window_size=3):
    """
    【安全策】座標は触らず、算出された「角度数値」に対して
    前後コマの中央値（メディアン）をとって一瞬の誤認識スパイクだけを弾く
    """
    if len(trajectory) < window_size:
        return trajectory

    num_frames = len(trajectory)
    
    # 開脚角度の平滑化
    split_angles = [pt.get('split_angle') for pt in trajectory]
    body_angles = [pt.get('body_angle') for pt in trajectory]

    for i in range(num_frames):
        start = max(0, i - window_size // 2)
        end = min(num_frames, i + window_size // 2 + 1)
        
        # 開脚角度の中央値
        valid_splits = [split_angles[k] for k in range(start, end) if split_angles[k] is not None]
        if valid_splits:
            trajectory[i]['split_angle'] = round(float(np.median(valid_splits)), 1)

        # 体幹角度の中央値
        valid_bodies = [body_angles[k] for k in range(start, end) if body_angles[k] is not None]
        if valid_bodies:
            trajectory[i]['body_angle'] = round(float(np.median(valid_bodies)), 1)

    return trajectory

def analyze_cheer_flyer_descent(raw_frames, max_jump_distance=180.0, min_size_ratio=0.01, min_peak_conf=0.35):
    peak_frame_idx = -1
    min_y = float('inf')
    peak_detection = None

    for frame_info in raw_frames:
        f_idx = frame_info['frame_idx']
        f_height = frame_info.get('frame_height', 1080)
        min_box_h = f_height * min_size_ratio

        for det in frame_info['detections']:
            if det['is_edge']: continue
            if det.get('box_height', 0) < min_box_h: continue
            if det['conf'] < min_peak_conf: continue

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
    current_target['valid_for_scoring'] = current_target['conf'] >= 0.35
    
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
            dist = np.sqrt((cand['center'][0] - prev_center[0])**2 + (cand['center'][1] - prev_center[1])**2)
            if dist < min_dist:
                min_dist = dist
                best_candidate = cand

        if best_candidate and min_dist <= max_jump_distance:
            tracked_flyer = best_candidate.copy()
            tracked_flyer['frame_idx'] = f_idx
            tracked_flyer['valid_for_scoring'] = tracked_flyer['conf'] >= 0.35
            angles = extract_cheer_angles(tracked_flyer.get('keypoints'))
            tracked_flyer['body_angle'] = angles['body_angle']
            tracked_flyer['split_angle'] = angles['split_angle']
            trajectory.append(tracked_flyer)
            prev_center = tracked_flyer['center']
        else:
            break

    # ★ 角度の数値に対してメディアンフィルタを適用（歪み・誤判定の完全排除）
    trajectory = smooth_angles_median(trajectory, window_size=3)

    return trajectory

def render_flyer_capture(video_path, flyer_data, kpt_conf_thresh=0.20):
    if not flyer_data: return None
    cap = cv2.VideoCapture(video_path)
    cap.set(cv2.CAP_PROP_POS_FRAMES, flyer_data['frame_idx'])
    ret, frame = cap.read()
    cap.release()
    if not ret or frame is None: return None

    bbox = flyer_data.get('bbox')
    if bbox:
        x1, y1, x2, y2 = map(int, bbox)
        cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 255), 2)

    kpts = flyer_data.get('keypoints', [])
    if kpts and len(kpts) >= 17:
        for pt in kpts:
            x, y = int(pt[0]), int(pt[1])
            conf = pt[2] if len(pt) > 2 else 1.0
            if x > 0 and y > 0 and conf >= kpt_conf_thresh:
                cv2.circle(frame, (x, y), 5, (0, 0, 255), -1)

        for p1_idx, p2_idx in SKELETON_CONNECTIONS:
            pt1, pt2 = kpts[p1_idx], kpts[p2_idx]
            conf1 = pt1[2] if len(pt1) > 2 else 1.0
            conf2 = pt2[2] if len(pt2) > 2 else 1.0
            if conf1 >= kpt_conf_thresh and conf2 >= kpt_conf_thresh:
                x1, y1 = int(pt1[0]), int(pt1[1])
                x2, y2 = int(pt2[0]), int(pt2[1])
                if x1 > 0 and y1 > 0 and x2 > 0 and y2 > 0:
                    cv2.line(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)

    info_text = f"Frame: {flyer_data['frame_idx']} | Conf: {int(flyer_data['conf']*100)}%"
    cv2.putText(frame, info_text, (20, 40), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2, cv2.LINE_AA)
    
    b_ang = flyer_data.get('body_angle')
    s_ang = flyer_data.get('split_angle')
    
    cv2.putText(frame, f"Body: {b_ang if b_ang is not None else 'N/A'} deg", (20, 75), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2, cv2.LINE_AA)
    cv2.putText(frame, f"Split: {s_ang if s_ang is not None else 'N/A'} deg", (20, 110), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2, cv2.LINE_AA)

    return cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

