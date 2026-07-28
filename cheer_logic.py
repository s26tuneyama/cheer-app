# cheer_logic.py
import cv2
import numpy as np
import mediapipe as mp

mp_pose = mp.solutions.pose

def calculate_angle(a, b, c):
    """3点の座標から角度（度）を計算"""
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
    """MediaPipeによる関節角度の抽出"""
    h_orig, w_orig, _ = frame.shape
    x1, y1, x2, y2 = map(int, bbox)
    
    bw, bh = x2 - x1, y2 - y1
    max_dim = max(bw, bh)
    
    pad_w, pad_h = int(max_dim * 1.2), int(max_dim * 1.0)
    cx1, cy1 = max(0, x1 - pad_w), max(0, y1 - pad_h)
    cx2, cy2 = min(w_orig, x2 + pad_w), min(h_orig, y2 + pad_h)
    
    crop = frame[cy1:cy2, cx1:cx2]
    if crop.size == 0:
        return {'split_angle': None, 'posture_angle': None, 'mp_kpts': {}, 'full_bbox': bbox, 'ankle_y': (y1 + y2) / 2}

    crop_rgb = cv2.cvtColor(crop, cv2.COLOR_BGR2RGB)
    results = pose_estimator.process(crop_rgb)
    
    if not results.pose_landmarks:
        return {'split_angle': None, 'posture_angle': None, 'mp_kpts': {}, 'full_bbox': bbox, 'ankle_y': (y1 + y2) / 2}

    landmarks = results.pose_landmarks.landmark
    kpts = {}
    crop_h, crop_w, _ = crop.shape
    x_coords, y_coords = [], []

    for idx, lm in enumerate(landmarks):
        if lm.visibility > 0.15:
            abs_x = int(lm.x * crop_w + cx1)
            abs_y = int(lm.y * crop_h + cy1)
            kpts[idx] = [abs_x, abs_y, lm.visibility]
            x_coords.append(abs_x)
            y_coords.append(abs_y)

    # 関節座標の取得
    hip_center = [(kpts[23][0] + kpts[24][0])/2, (kpts[23][1] + kpts[24][1])/2] if 23 in kpts and 24 in kpts else None
    shoulder_center = [(kpts[11][0] + kpts[12][0])/2, (kpts[11][1] + kpts[12][1])/2] if 11 in kpts and 12 in kpts else None
    
    left_foot = kpts.get(31, kpts.get(27))
    right_foot = kpts.get(32, kpts.get(28))

    # 足首/つま先高さ (ankle_y)
    if left_foot and right_foot:
        ankle_y = (left_foot[1] + right_foot[1]) / 2.0
    elif left_foot or right_foot:
        ankle_y = (left_foot or right_foot)[1]
    else:
        ankle_y = (y1 + y2) / 2.0

    # 開脚角度 (Split Angle)
    split_angle = calculate_angle(left_foot, hip_center, right_foot) if (hip_center and left_foot and right_foot) else None

    # 体幹姿勢角度 (Posture Angle: 肩-腰-足首)
    foot_center = [(kpts[27][0] + kpts[28][0])/2, (kpts[27][1] + kpts[28][1])/2] if 27 in kpts and 28 in kpts else None
    posture_angle = calculate_angle(shoulder_center, hip_center, foot_center) if (shoulder_center and hip_center and foot_center) else None

    # Bounding Box調整
    full_bbox = [max(0, min(x_coords) - 15), max(0, min(y_coords) - 15), min(w_orig, max(x_coords) + 15), min(h_orig, max(y_coords) + 15)] if (x_coords and y_coords) else bbox

    return {
        'split_angle': round(split_angle, 1) if split_angle else None,
        'posture_angle': round(posture_angle, 1) if posture_angle else None,
        'mp_kpts': kpts,
        'full_bbox': full_bbox,
        'ankle_y': ankle_y
    }

def analyze_cheer_motion(video_path, raw_frames, technique_type="トータッチ・トス（フライヤー）", max_jump_distance=350.0, min_size_ratio=0.01, min_peak_conf=0.15):
    """動体追跡 ＆ 軌道計算メインロジック"""
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
    valid_candidates = []

    with mp_pose.Pose(static_image_mode=True, model_complexity=1, min_detection_confidence=0.2) as pose:
        for cand in candidates[:25]:
            cap.set(cv2.CAP_PROP_POS_FRAMES, cand['frame_idx'])
            ret, frame = cap.read()
            if not ret or frame is None: continue

            mp_res = extract_mediapipe_angles(frame, cand['bbox'], pose)
            if len(mp_res.get('mp_kpts', {})) >= 4:
                cand_data = cand.copy()
                cand_data.update(mp_res)
                valid_candidates.append(cand_data)

        if not valid_candidates:
            cap.release()
            return []

        # 最高到達点（つま先位置が一番高いフレーム）
        peak_detection = min(valid_candidates, key=lambda x: x['ankle_y'])
        trajectory = [peak_detection]
        prev_center = peak_detection['center']

        # 降下追尾
        descent_frames = [f for f in raw_frames if f['frame_idx'] > peak_detection['frame_idx']]
        missing_count = 0

        for frame_info in descent_frames:
            f_idx = frame_info['frame_idx']
            valid_dets = [d for d in frame_info['detections'] if not d['is_edge']]
            
            # 最も近い人物を追う（近接トラッキング）
            close_dets = [d for d in valid_dets if np.sqrt((d['center'][0]-prev_center[0])**2 + (d['center'][1]-prev_center[1])**2) <= max_jump_distance]

            if not close_dets:
                missing_count += 1
                if missing_count > 4: break
                continue

            missing_count = 0
            best_cand = min(close_dets, key=lambda d: np.sqrt((d['center'][0]-prev_center[0])**2 + (d['center'][1]-prev_center[1])**2))

            cap.set(cv2.CAP_PROP_POS_FRAMES, f_idx)
            ret, frame = cap.read()
            if ret and frame is not None:
                mp_res = extract_mediapipe_angles(frame, best_cand['bbox'], pose)
                tracked = best_cand.copy()
                tracked['frame_idx'] = f_idx
                tracked.update(mp_res)
                trajectory.append(tracked)
                prev_center = tracked['center']

    cap.release()
    return trajectory

def generate_diagnosis(peak_data, descent_data, technique_type):
    """高空スナップ評価に対応した骨格診断"""
    diagnoses = []
    
    # 1. 最高到達点（開脚）の診断
    split_angle = peak_data.get('split_angle')
    if split_angle is not None:
        if split_angle >= 150:
            diagnoses.append("✨ **開脚力（ピーク）**: 素晴らしい柔軟性とキープ力です！つま先まで美しくラインが出ています。")
        elif split_angle >= 120:
            diagnoses.append("👍 **開脚力（ピーク）**: 十分な開脚度です。あと少し股関節を引き上げると180度に近づきます！")
        else:
            diagnoses.append("💡 **開脚力（ピーク）**: 跳び出し時の蹴り出しと、空中でのつま先の引き込みを意識してみましょう。")
    
    # 2. 高空スナップ（脚閉じ）の診断
    descent_split = descent_data.get('split_angle')
    posture_angle = descent_data.get('posture_angle')

    if descent_split is not None:
        if descent_split <= 35:
            diagnoses.append("⚡ **高空スナップ**: 頂点直後の脚閉じが非常にスピーディーで綺麗です！")
        elif descent_split > 60:
            diagnoses.append("⚠️ **高空スナップ**: 頂点を過ぎた後、脚を閉じるタイミングが少し遅れている可能性があります。ピーク直後に素早く寄せる意識を持ちましょう。")
        else:
            diagnoses.append("👍 **高空スナップ**: 空中でスピーディーに脚を寄せています。")
    else:
        diagnoses.append("⚡ **高空スナップ**: 左右の脚がぴったり重なるほど綺麗に閉じ切っています！（関節重なり＝完全スナップの証拠です）")

    if posture_angle is not None:
        if posture_angle >= 160:
            diagnoses.append("🎯 **空中姿勢（体幹）**: 軸がブレずまっすぐキープできています。")
        else:
            diagnoses.append("⚠️ **空中姿勢（体幹）**: 空中で腰がやや曲がり気味です。胸と引き上げを意識しましょう。")

    return diagnoses

def render_flyer_capture(video_path, flyer_data):
    """検出フレームの画像レンダリング"""
    if not flyer_data: return None
    cap = cv2.VideoCapture(video_path)
    cap.set(cv2.CAP_PROP_POS_FRAMES, flyer_data['frame_idx'])
    ret, frame = cap.read()
    cap.release()
    if not ret or frame is None: return None

    bbox = flyer_data.get('full_bbox', flyer_data.get('bbox'))
    if bbox:
        cv2.rectangle(frame, (int(bbox[0]), int(bbox[1])), (int(bbox[2]), int(bbox[3])), (0, 255, 255), 2)

    kpts = flyer_data.get('mp_kpts', {})
    for idx, pt in kpts.items():
        cv2.circle(frame, (pt[0], pt[1]), 4, (0, 0, 255), -1)

    connections = [(11, 12), (11, 23), (12, 24), (23, 24), (23, 25), (25, 27), (27, 31), (24, 26), (26, 28), (28, 32)]
    for p1, p2 in connections:
        if p1 in kpts and p2 in kpts:
            cv2.line(frame, (kpts[p1][0], kpts[p1][1]), (kpts[p2][0], kpts[p2][1]), (0, 255, 0), 2)

    s_ang, p_ang = flyer_data.get('split_angle'), flyer_data.get('posture_angle')
    cv2.putText(frame, f"Frame: {flyer_data['frame_idx']} | Conf: {int(flyer_data['conf']*100)}%", (20, 40), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)
    cv2.putText(frame, f"Split Angle: {s_ang if s_ang is not None else 'N/A'} deg", (20, 75), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)
    cv2.putText(frame, f"Body Posture: {p_ang if p_ang is not None else 'N/A'} deg", (20, 110), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 150, 0), 2)

    return cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
