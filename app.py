import streamlit as st
import cv2
import numpy as np
import tempfile
import os
import json
from ultralytics import YOLO

# ==========================================
# 1. 知識ベース（ルールとアドバイス）
# ==========================================
KNOWLEDGE_BASE_JSON = """
{
  "rules": [
    {"metric": "r_knee_angle", "operator": "<", "threshold": 160, "tag": "RIGHT_KNEE_BENT"},
    {"metric": "l_knee_angle", "operator": "<", "threshold": 160, "tag": "LEFT_KNEE_BENT"},
    {"metric": "waist_flexion", "operator": "<", "threshold": 150, "tag": "HIP_FLEXION"}
  ],
  "advices": {
    "RIGHT_KNEE_BENT": [
      {"author": "歴代キャプテン", "advice": "頂点で右膝が曲がっています。つま先をさらに遠くへ引っ張る意識を持とう！"}
    ],
    "LEFT_KNEE_BENT": [
      {"author": "歴代キャプテン", "advice": "頂点で左膝が曲がっています。蹴り上げの瞬間に太ももに力を入れて！"}
    ],
    "HIP_FLEXION": [
      {"author": "コーチ", "advice": "下降時に腰が屈曲しています。上体を一直線にキープするスナップダウンを意識しましょう！"}
    ],
    "PERFECT": [
      {"author": "AI診断", "advice": "膝も伸び、スナップダウン時の腰の屈曲もない完璧なトスです！素晴らしい！"}
    ]
  }
}
"""
knowledge = json.loads(KNOWLEDGE_BASE_JSON)

# ==========================================
# 2. 処理機構（ロジック層）
# ==========================================
def calculate_angle(a, b, c):
    a, b, c = np.array(a), np.array(b), np.array(c)
    ba = a - b
    bc = c - b
    cosine_angle = np.dot(ba, bc) / (np.linalg.norm(ba) * np.linalg.norm(bc))
    return np.degrees(np.arccos(np.clip(cosine_angle, -1.0, 1.0)))

def extract_jump_metrics(video_path, model):
    cap = cv2.VideoCapture(video_path)
    apex_frame, snap_frame = None, None
    apex_kpts, snap_kpts = None, None
    min_y = float('inf') 
    phase = "ASCENT"
    frame_count = 0
    
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret: break
        frame_count += 1
        if frame_count % 2 != 0: continue

        results = model(frame, verbose=False)
        result = results[0]
        
        if len(result.keypoints) > 0:
            keypoints = result.keypoints.xy[0].cpu().numpy()
            l_hip, r_hip = keypoints[11], keypoints[12]
            
            if l_hip[1] > 0 and r_hip[1] > 0:
                current_y = (l_hip[1] + r_hip[1]) / 2
                
                # 頂点の検出
                if current_y < min_y:
                    min_y = current_y
                    apex_frame = frame.copy()
                    apex_kpts = keypoints
                    phase = "ASCENT"
                # 下降の検出
                elif current_y > min_y + 20 and phase == "ASCENT":
                    phase = "DESCENT"
                    
                # スナップダウンの検出
                if phase == "DESCENT" and snap_frame is None:
                    l_knee, r_knee = keypoints[13], keypoints[14]
                    l_ankle, r_ankle = keypoints[15], keypoints[16]
                    if l_ankle[1] > l_knee[1] and r_ankle[1] > r_knee[1]:
                        snap_frame = frame.copy()
                        snap_kpts = keypoints
    cap.release()
    
    metrics = {}
    if apex_kpts is not None and snap_kpts is not None:
        metrics["r_knee_angle"] = calculate_angle(apex_kpts[12], apex_kpts[14], apex_kpts[16])
        metrics["l_knee_angle"] = calculate_angle(apex_kpts[11], apex_kpts[13], apex_kpts[15])
        
        r_waist = calculate_angle(snap_kpts[6], snap_kpts[12], snap_kpts[14])
        l_waist = calculate_angle(snap_kpts[5], snap_kpts[11], snap_kpts[13])
        metrics["waist_flexion"] = (r_waist + l_waist) / 2
        
    # 計測データと一緒に、画像と座標データも画面表示用に返す
    return metrics, apex_frame, apex_kpts, snap_frame, snap_kpts

# ==========================================
# 3. 評価エンジン＆表示層（UI・描画）
# ==========================================
def evaluate_metrics(metrics, rules):
    tags = []
    for rule in rules:
        metric_val = metrics.get(rule["metric"])
        if metric_val is not None:
            if rule["operator"] == "<" and metric_val < rule["threshold"]:
                tags.append(rule["tag"])
    if not tags:
        tags.append("PERFECT")
    return tags

def draw_custom_keypoints(image, keypoints, phase_name):
    """画面表示用に骨格を画像に描き込む関数"""
    img_draw = image.copy()
    targets = {
        5: ("L-Shoulder", (255, 165, 0)), 6: ("R-Shoulder", (255, 165, 0)),
        11: ("L-Hip", (0, 255, 255)), 12: ("R-Hip", (0, 255, 255)),
        13: ("L-Knee", (0, 255, 0)), 14: ("R-Knee", (0, 255, 0)),
        15: ("L-Ankle", (255, 0, 0)), 16: ("R-Ankle", (255, 0, 0))
    }
    
    for idx, (label, color) in targets.items():
        x, y = int(keypoints[idx][0]), int(keypoints[idx][1])
        if x != 0 and y != 0:  
            cv2.circle(img_draw, (x, y), 6, color, -1)
            
    pts_r = [(int(keypoints[i][0]), int(keypoints[i][1])) for i in [6, 12, 14, 16]]
    pts_l = [(int(keypoints[i][0]), int(keypoints[i][1])) for i in [5, 11, 13, 15]]
    
    for pts in [pts_r, pts_l]:
        if all(p[0] != 0 for p in pts):
            cv2.line(img_draw, pts[0], pts[1], (255, 255, 255), 2)
            cv2.line(img_draw, pts[1], pts[2], (255, 255, 255), 2)
            cv2.line(img_draw, pts[2], pts[3], (255, 255, 255), 2)
            
    cv2.putText(img_draw, phase_name, (20, 40), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 255, 0), 3, cv2.LINE_AA)
    return img_draw

st.set_page_config(page_title="チア トータッチ診断AI", layout="centered")
st.title("📣 チア トータッチ診断 AI")

@st.cache_resource
def load_model():
    return YOLO('yolov8s-pose.pt') 

model = load_model()
uploaded_file = st.file_uploader("動画をアップロード", type=["mp4", "mov", "avi"])

if uploaded_file is not None:
    with st.spinner('ジャンプ全体を解析中...（数十秒かかります）'):
        tfile = tempfile.NamedTemporaryFile(delete=False, suffix='.mp4')
        tfile.write(uploaded_file.read())
        tfile.flush()
        
        try:
            metrics, apex_frame, apex_kpts, snap_frame, snap_kpts = extract_jump_metrics(tfile.name, model)
            
            if not metrics:
                st.error("ジャンプの一連の動作が正確に検出できませんでした。")
            else:
                tags = evaluate_metrics(metrics, knowledge["rules"])
                
                # 画像の描画（表示層で行う）
                img_apex = draw_custom_keypoints(apex_frame, apex_kpts, "APEX (Peak)")
                img_snap = draw_custom_keypoints(snap_frame, snap_kpts, "SNAP DOWN")
                
                st.success("一連のフォームを診断しました！")
                
                # 2枚の画像を横に並べて表示
                col_img1, col_img2 = st.columns(2)
                with col_img1:
                    st.image(cv2.cvtColor(img_apex, cv2.COLOR_BGR2RGB), caption="① 開脚の頂点", use_container_width=True)
                with col_img2:
                    st.image(cv2.cvtColor(img_snap, cv2.COLOR_BGR2RGB), caption="② スナップダウン時", use_container_width=True)
                
                # 数値データの表示
                col1, col2, col3 = st.columns(3)
                col1.metric(label="右膝角度 (頂点)", value=f"{metrics['r_knee_angle']:.1f}度")
                col2.metric(label="左膝角度 (頂点)", value=f"{metrics['l_knee_angle']:.1f}度")
                col3.metric(label="腰の角度 (下降時)", value=f"{metrics['waist_flexion']:.1f}度")
                
                # 知識ベースからのアドバイス表示
                st.markdown("### 💡 診断結果とアドバイス")
                for tag in tags:
                    for item in knowledge["advices"].get(tag, []):
                        st.info(f"**{item['author']}より**: {item['advice']}")
        finally:
            tfile.close()
            os.unlink(tfile.name)

