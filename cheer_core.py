# cheer_core.py (一部抜粋：extract_mediapipe_angles 関数の更新)
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
    """MediaPipeによる関節角度・つま先・足閉じの抽出"""
    h_orig, w_orig, _ = frame.shape
    x1, y1, x2, y2 = map(int, bbox)
    
    bw, bh = x2 - x1, y2 - y1
    max_dim = max(bw, bh)
    
    pad_w, pad_h = int(max_dim * 1.2), int(max_dim * 1.0)
    cx1, cy1 = max(0, x1 - pad_w), max(0, y1 - pad_h)
    cx2, cy2 = min(w_orig, x2 + pad_w), min(h_orig, y2 + pad_h)
    
    crop = frame[cy1:cy2, cx1:cx2]
    if crop.size == 0:
        return {'split_angle': None, 'posture_angle': None, 'toe_extended': None, 'feet_closed': None, 'mp_kpts': {}, 'full_bbox': bbox, 'ankle_y': (y1 + y2) / 2}

    crop_rgb = cv2.cvtColor(crop, cv2.COLOR_BGR2RGB)
    results = pose_estimator.process(crop_rgb)
    
    if not results.pose_landmarks:
        return {'split_angle': None, 'posture_angle': None, 'toe_extended': None, 'feet_closed': None, 'mp_kpts': {}, 'full_bbox': bbox, 'ankle_y': (y1 + y2) / 2}

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

    hip_center = [(kpts[23][0] + kpts[24][0])/2, (kpts[23][1] + kpts[24][1])/2] if 23 in kpts and 24 in kpts else None
    shoulder_center = [(kpts[11][0] + kpts[12][0])/2, (kpts[11][1] + kpts[12][1])/2] if 11 in kpts and 12 in kpts else None
    
    left_foot = kpts.get(31, kpts.get(27))
    right_foot = kpts.get(32, kpts.get(28))

    if left_foot and right_foot:
        ankle_y = (left_foot[1] + right_foot[1]) / 2.0
    elif left_foot or right_foot:
        ankle_y = (left_foot or right_foot)[1]
    else:
        ankle_y = (y1 + y2) / 2.0

    # 1. 開脚角度
    split_angle = calculate_angle(left_foot, hip_center, right_foot) if (hip_center and left_foot and right_foot) else None
    
    # 2. 体幹角度（肩-腰-足中央）
    foot_center = [(kpts[27][0] + kpts[28][0])/2, (kpts[27][1] + kpts[28][1])/2] if 27 in kpts and 28 in kpts else None
    posture_angle = calculate_angle(shoulder_center, hip_center, foot_center) if (shoulder_center and hip_center and foot_center) else None

    # 3. つま先の伸ばし判定 (膝-足首-つま先の角度が 145度以上ならポアント状態)
    left_toe_ang = calculate_angle(kpts[25], kpts[27], kpts[31]) if (25 in kpts and 27 in kpts and 31 in kpts) else None
    right_toe_ang = calculate_angle(kpts[26], kpts[28], kpts[32]) if (26 in kpts and 28 in kpts and 32 in kpts) else None
    
    toe_angles = [a for a in [left_toe_ang, right_toe_ang] if a is not None]
    toe_extended = (sum(toe_angles) / len(toe_angles) >= 145) if toe_angles else None

    # 4. 着地の足閉じ判定 (左右のかかと/足首の距離が胴体幅より狭いか)
    feet_closed = None
    if 27 in kpts and 28 in kpts and 23 in kpts and 24 in kpts:
        feet_dist = abs(kpts[27][0] - kpts[28][0])
        hip_dist = abs(kpts[23][0] - kpts[24][0])
        feet_closed = feet_dist <= (hip_dist * 1.5)

    full_bbox = [max(0, min(x_coords) - 15), max(0, min(y_coords) - 15), min(w_orig, max(x_coords) + 15), min(h_orig, max(y_coords) + 15)] if (x_coords and y_coords) else bbox

    return {
        'split_angle': round(split_angle, 1) if split_angle else None,
        'posture_angle': round(posture_angle, 1) if posture_angle else None,
        'toe_extended': toe_extended,
        'feet_closed': feet_closed,
        'mp_kpts': kpts,
        'full_bbox': full_bbox,
        'ankle_y': ankle_y
    }

