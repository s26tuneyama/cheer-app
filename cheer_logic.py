# cheer_logic.py の末尾に追加してください
import cv2

# COCO 17関節の接続関係（骨格線を描くためのペア）
SKELETON_CONNECTIONS = [
    (5, 6),   # 両肩
    (5, 7), (7, 9),   # 左腕
    (6, 8), (8, 10),  # 右腕
    (5, 11), (6, 12), # 体幹（肩〜腰）
    (11, 12), # 両腰
    (11, 13), (13, 15), # 左脚
    (12, 14), (14, 16)  # 右脚
]

def render_flyer_capture(video_path, flyer_data):
    """
    指定されたコマ（flyer_data）のフレーム画像を取り出し、
    BBox・骨格線・角度テキストを描画してRGB画像として返す
    """
    if not flyer_data:
        return None

    cap = cv2.VideoCapture(video_path)
    cap.set(cv2.CAP_PROP_POS_FRAMES, flyer_data['frame_idx'])
    ret, frame = cap.read()
    cap.release()

    if not ret or frame is None:
        return None

    # 1. バウンディングボックス (BBox) の描画（黄色）
    bbox = flyer_data.get('bbox')
    if bbox:
        x1, y1, x2, y2 = map(int, bbox)
        cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 255), 2)

    # 2. 骨格線と関節ポイントの描画
    kpts = flyer_data.get('keypoints', [])
    if kpts and len(kpts) >= 17:
        # 関節点（赤丸）
        for pt in kpts:
            x, y = int(pt[0]), int(pt[1])
            if x > 0 and y > 0:
                cv2.circle(frame, (x, y), 4, (0, 0, 255), -1)

        # 骨格線（緑線）
        for p1_idx, p2_idx in SKELETON_CONNECTIONS:
            pt1, pt2 = kpts[p1_idx], kpts[p2_idx]
            x1, y1 = int(pt1[0]), int(pt1[1])
            x2, y2 = int(pt2[0]), int(pt2[1])
            if x1 > 0 and y1 > 0 and x2 > 0 and y2 > 0:
                cv2.line(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)

    # 3. テキスト情報（Frame, Conf, 角度）の上書き描画
    info_text = f"Frame: {flyer_data['frame_idx']} | Conf: {int(flyer_data['conf']*100)}%"
    cv2.putText(frame, info_text, (20, 40), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2, cv2.LINE_AA)

    if flyer_data.get('body_angle') is not None:
        cv2.putText(frame, f"Body: {flyer_data['body_angle']} deg", (20, 75), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2, cv2.LINE_AA)
        
    if flyer_data.get('split_angle') is not None:
        cv2.putText(frame, f"Split: {flyer_data['split_angle']} deg", (20, 110), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2, cv2.LINE_AA)

    # BGR -> RGB 変換 (Streamlit表示用)
    return cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

