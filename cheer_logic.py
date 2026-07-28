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

def extract_mediapipe_angles(frame, bbox):
    """
    フライヤー周辺をクロップし、MediaPipe Poseで33箇所のキーポイント（つま先含む）を取得
    """
    h_orig, w_orig, _ = frame.shape
    x1, y1, x2, y2 = map(int, bbox)
    
    pad_w = int((x2 - x1) * 0.3)
    pad_h = int((y2 - y1) * 0.3)
    cx1, cy1 = max(0, x1 - pad_w), max(0, y1 - pad_h)
    cx2, cy2 = min(w_orig, x2 + pad_w), min(h_orig, y2 + pad_h)
    
    crop = frame[cy1:cy2, cx1:cx2]
    if crop.size == 0:
        return {'split_angle': None, 'arch_angle': None, 'mp_kpts': {}}

    crop_rgb = cv2.cvtColor(crop, cv2.COLOR_BGR2RGB)
    
    with mp_pose.Pose(static_image_mode=True, model_complexity=1, min_detection_confidence=0.3) as pose:
        results = pose.process(crop_rgb)
        
        if not results.pose_landmarks:
            return {'split_angle': None, 'arch_angle': None, 'mp_kpts': {}}

        landmarks = results.pose_landmarks.landmark
        
        kpts = {}
        crop_h, crop_w, _ = crop.shape
        for idx, lm in enumerate(landmarks):
            if lm.visibility > 0.2:
                abs_x = int(lm.x * crop_w + cx1)
                abs_y = int(lm.y * crop_h + cy1)
                kpts[idx] = [abs_x, abs_y, lm.visibility]

        # 1. 開脚角度（腰 - つま先/足首）
        split_angle = None
        hip_center = None
        if 23 in kpts and 24 in kpts:
            hip_center = [(kpts[23][0] + kpts[24][0])/2, (kpts[23][1] + kpts[24][1])/2]
        
        left_foot = kpts.get(31, kpts.get(27))
        right_foot = kpts.get(32, kpts.get(28))

        if hip_center and left_foot and right_foot:
            split_angle = calculate_angle(left_foot, hip_center, right_foot)

        # 2. 体の反り角度（肩 - 腰 - 足首）
        arch_angle = None
        shoulder_center = None
        foot_center = None
        if 11 in kpts and 12 in kpts:
            shoulder_center = [(kpts[11][0] + kpts[12][0])/2, (kpts[11][1] + kpts[12][1])/2]
        if 27 in kpts and 28 in kpts:
            foot_center = [(kpts[27][0] + kpts[28][0])/2, (kpts[27][1] + kpts[28][1])/2]

        if shoulder_center and hip_center and foot_center:
            arch_angle = calculate_angle(shoulder_center, hip_center, foot_center)

        return {
            'split_angle': round(split_angle, 1) if split_angle else None,
            'arch_angle': round(arch_angle, 1) if arch_angle else None,
            'mp_kpts': kpts
        }

def analyze_cheer_flyer_descent(video_path, raw_frames, max_jump_distance=180.0, min_size_ratio=0.01, min_peak_conf=0.20):
    """
    1. 人間（骨格）が取れた検出のみをフィルタリング
    2. その中から最も高い位置にあるものを最高到達点として判定
    """
    cap = cv2.VideoCapture(video_path)
    valid_candidates = []

    # 全コマの検出結果から「本当に人間であるもの（骨格が5箇所以上取れたもの）」を抽出
    for frame_info in raw_frames:
        f_idx = frame_info['frame_idx']
        f_height = frame_info.get('frame_height', 1080)
        min_box_h = f_height * min_size_ratio

        for det in frame_info['detections']:
            if det['is_edge']: continue
            if det.get('box_height', 0) < min_box_h: continue
            if det['conf'] < min_peak_conf: continue

            cap.set(cv2.CAP_PROP_POS_FRAMES, f_idx)
            ret, frame = cap.read()
            if not ret or frame is None: continue

            # MediaPipeによる人間チェック
            mp_res = extract_mediapipe_angles(frame, det['bbox'])
            
            # ★ ライトなどの誤検知を防ぐ：キーポイントが5つ以上取れた「本物の人間」だけを許可
            if len(mp_res.get('mp_kpts', {})) >= 5:
                cand = det.copy()
                cand['frame_idx'] = f_idx
                cand['split_angle'] = mp_res['split_angle']
                cand['arch_angle'] = mp_res['arch_angle']
                cand['mp_kpts'] = mp_res['mp_kpts']
                valid_candidates.append(cand)

    cap.release()

    if not valid_candidates:
        return []

    # 「人間」と判定されたものの中で、画面の最も高い位置（Y座標が最小）にあるものをピーク（最高到達点）に設定
    peak_detection = min(valid_candidates, key=lambda x: x['center'][1])
    peak_frame_idx = peak_detection['frame_idx']

    # 最高到達点以降の落下軌道を抽出
    trajectory = [peak_detection]
    prev_center = peak_detection['center']

    # ピーク以降の人間候補のみを追跡
    descent_candidates = [c for c in valid_candidates if c['frame_idx'] > peak_frame_idx]
    
    current_f = peak_frame_idx
    while True:
        # 次のコマの候補を探す
        next_frame_cands = [c for c in descent_candidates if c['frame_idx'] == current_f + 1 or c['frame_idx'] == current_f + 2]
        if not next_frame_cands:
            break
            
        best_cand = min(next_frame_cands, key=lambda c: np.sqrt((c['center'][0]-prev_center[0])**2 + (c['center'][1]-prev_center[1])**2))
        dist = np.sqrt((best_cand['center'][0]-prev_center[0])**2 + (best_cand['center'][1]-prev_center[1])**2)
        
        if dist <= max_jump_distance:
            trajectory.append(best_cand)
            prev_center = best_cand['center']
            current_f = best_cand['frame_idx']
        else:
            break

    return trajectory

def render_flyer_capture(video_path, flyer_data):
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

