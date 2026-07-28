# cheer_logic.py
import cv2
import numpy as np
import mediapipe as mp

mp_pose = mp.solutions.pose

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

def extract_mediapipe_angles(frame, bbox, pose_estimator):
    """
    180度開脚やつま先を逃さない広範囲クロップ ＋ 正確な「開脚」「反り」の分離計算
    """
    h_orig, w_orig, _ = frame.shape
    x1, y1, x2, y2 = map(int, bbox)
    
    bw = x2 - x1
    bh = y2 - y1
    max_dim = max(bw, bh)
    
    pad_w = int(max_dim * 1.2)
    pad_h = int(max_dim * 1.0)
    
    cx1, cy1 = max(0, x1 - pad_w), max(0, y1 - pad_h)
    cx2, cy2 = min(w_orig, x2 + pad_w), min(h_orig, y2 + pad_h)
    
    crop = frame[cy1:cy2, cx1:cx2]
    if crop.size == 0:
        return {'split_angle': None, 'arch_angle': None, 'mp_kpts': {}, 'full_bbox': bbox}

    crop_rgb = cv2.cvtColor(crop, cv2.COLOR_BGR2RGB)
    results = pose_estimator.process(crop_rgb)
    
    if not results.pose_landmarks:
        return {'split_angle': None, 'arch_angle': None, 'mp_kpts': {}, 'full_bbox': bbox}

    landmarks = results.pose_landmarks.landmark
    kpts = {}
    crop_h, crop_w, _ = crop.shape
    x_coords, y_coords = [], []

    for idx, lm in enumerate(landmarks):
        if lm.visibility > 0.2:
            abs_x = int(lm.x * crop_w + cx1)
            abs_y = int(lm.y * crop_h + cy1)
            kpts[idx] = [abs_x, abs_y, lm.visibility]
            x_coords.append(abs_x)
            y_coords.append(abs_y)

    # 1. 最大開脚角度（左足 - 腰 - 右足）
    split_angle = None
    hip_center = None
    if 23 in kpts and 24 in kpts:
        hip_center = [(kpts[23][0] + kpts[24][0])/2, (kpts[23][1] + kpts[24][1])/2]
    
    left_foot = kpts.get(31, kpts.get(27))
    right_foot = kpts.get(32, kpts.get(28))

    if hip_center and left_foot and right_foot:
        split_angle = calculate_angle(left_foot, hip_center, right_foot)

    # 2. 体の反り角度（Arch）
    # ガード条件: 足が90度以上開いている（開脚技中）場合は「反り」の誤判定を防ぐため計算しない
    arch_angle = None
    if split_angle is None or split_angle < 90.0:
        shoulder_center = None
        foot_center = None
        if 11 in kpts and 12 in kpts:
            shoulder_center = [(kpts[11][0] + kpts[12][0])/2, (kpts[11][1] + kpts[12][1])/2]
        if 27 in kpts and 28 in kpts:
            foot_center = [(kpts[27][0] + kpts[28][0])/2, (kpts[27][1] + kpts[28][1])/2]

        if shoulder_center and hip_center and foot_center:
            raw_arch = calculate_angle(shoulder_center, hip_center, foot_center)
            if raw_arch is not None:
                # 直線（180度）からの曲がり具合を「反り度」として算出
                arch_angle = round(180.0 - raw_arch, 1)

    # 全身を綺麗に囲むバウンディングボックス
    if x_coords and y_coords:
        fx1, fy1 = max(0, min(x_coords) - 15), max(0, min(y_coords) - 15)
        fx2, fy2 = min(w_orig, max(x_coords) + 15), min(h_orig, max(y_coords) + 15)
        full_bbox = [fx1, fy1, fx2, fy2]
    else:
        full_bbox = bbox

    return {
        'split_angle': round(split_angle, 1) if split_angle else None,
        'arch_angle': arch_angle,
        'mp_kpts': kpts,
        'full_bbox': full_bbox
    }

def analyze_cheer_flyer_descent(video_path, raw_frames, max_jump_distance=180.0, min_size_ratio=0.01, min_peak_conf=0.20):
    candidates = []
    for frame_info in raw_frames:
        f_idx = frame_info['frame_idx']
        f_height = frame_info.get('frame_height', 1080)
        min_box_h = f_height * min_size_ratio

        for det in frame_info['detections']:
            if det['is_edge']: continue
            if det.get('box_height', 0) < min_box_h: continue
            if det['conf'] < min_peak_conf: continue

            c_data = det.copy()
            c_data['frame_idx'] = f_idx
            candidates.append(c_data)

    if not candidates:
        return []

    candidates.sort(key=lambda x: x['center'][1])

    cap = cv2.VideoCapture(video_path)
    peak_detection = None

    with mp_pose.Pose(static_image_mode=True, model_complexity=1, min_detection_confidence=0.3) as pose:
        for cand in candidates:
            cap.set(cv2.CAP_PROP_POS_FRAMES, cand['frame_idx'])
            ret, frame = cap.read()
            if not ret or frame is None: continue

            mp_res = extract_mediapipe_angles(frame, cand['bbox'], pose)
            
            if len(mp_res.get('mp_kpts', {})) >= 5:
                peak_detection = cand
                peak_detection['split_angle'] = mp_res['split_angle']
                peak_detection['arch_angle'] = mp_res['arch_angle']
                peak_detection['mp_kpts'] = mp_res['mp_kpts']
                peak_detection['full_bbox'] = mp_res['full_bbox']
                break

        if peak_detection is None:
            cap.release()
            return []

        peak_frame_idx = peak_detection['frame_idx']
        trajectory = [peak_detection]
        prev_center = peak_detection['center']

        descent_frames = [f for f in raw_frames if f['frame_idx'] > peak_frame_idx]

        for frame_info in descent_frames:
            f_idx = frame_info['frame_idx']
            valid_dets = [d for d in frame_info['detections'] if not d['is_edge']]
            if not valid_dets: break

            best_cand = min(valid_dets, key=lambda c: np.sqrt((c['center'][0]-prev_center[0])**2 + (c['center'][1]-prev_center[1])**2))
            dist = np.sqrt((best_cand['center'][0]-prev_center[0])**2 + (best_cand['center'][1]-prev_center[1])**2)

            if dist > max_jump_distance:
                break

            cap.set(cv2.CAP_PROP_POS_FRAMES, f_idx)
            ret, frame = cap.read()
            if ret and frame is not None:
                mp_res = extract_mediapipe_angles(frame, best_cand['bbox'], pose)
                tracked_flyer = best_cand.copy()
                tracked_flyer['frame_idx'] = f_idx
                tracked_flyer['split_angle'] = mp_res['split_angle']
                tracked_flyer['arch_angle'] = mp_res['arch_angle']
                tracked_flyer['mp_kpts'] = mp_res['mp_kpts']
                tracked_flyer['full_bbox'] = mp_res['full_bbox']
                trajectory.append(tracked_flyer)
                prev_center = tracked_flyer['center']

    cap.release()
    return trajectory

def render_flyer_capture(video_path, flyer_data):
    if not flyer_data: return None
    cap = cv2.VideoCapture(video_path)
    cap.set(cv2.CAP_PROP_POS_FRAMES, flyer_data['frame_idx'])
    ret, frame = cap.read()
    cap.release()
    if not ret or frame is None: return None

    bbox = flyer_data.get('full_bbox', flyer_data.get('bbox'))
    if bbox:
        x1, y1, x2, y2 = map(int, bbox)
        cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 255), 2)

    kpts = flyer_data.get('mp_kpts', {})
    for idx, pt in kpts.items():
        cv2.circle(frame, (pt[0], pt[1]), 4, (0, 0, 255), -1)

    connections = [
        (11, 12), (11, 23), (12, 24), (23, 24),
        (23, 25), (25, 27), (27, 31),
        (24, 26), (26, 28), (28, 32)
    ]
    for p1, p2 in connections:
        if p1 in kpts and p2 in kpts:
            cv2.line(frame, (kpts[p1][0], kpts[p1][1]), (kpts[p2][0], kpts[p2][1]), (0, 255, 0), 2)

    s_ang = flyer_data.get('split_angle')
    a_ang = flyer_data.get('arch_angle')
    
    cv2.putText(frame, f"Frame: {flyer_data['frame_idx']} | Conf: {int(flyer_data['conf']*100)}%", (20, 40), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)
    cv2.putText(frame, f"Split (Toe): {s_ang if s_ang is not None else 'N/A'} deg", (20, 75), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)
    cv2.putText(frame, f"Arch (Back): {a_ang if a_ang is not None else 'N/A'} deg", (20, 110), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 150, 0), 2)

    return cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

