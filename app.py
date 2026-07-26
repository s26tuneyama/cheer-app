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
# 2. 処理機構（ロジック層）★ここを大改修！
# ==========================================
def calculate_angle(a, b, c):
    a, b, c = np.array(a), np.array(b), np.array(c)
    ba = a - b
    bc = c - b
    cosine_angle = np.dot(ba, bc) / (np.linalg.norm(ba) * np.linalg.norm(bc))
    return np.degrees(np.arccos(np.clip(cosine_angle, -1.0, 1.0)))

def extract_jump_metrics(video_path, model):
    cap = cv2.VideoCapture(video_path)
    
    # 全フレームの「一番高い人（ジャンパー）」のデータを記録するリスト
    history = []
    
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret: break

        results = model(frame, verbose=False)
        result = results[0]
        
        # フレーム内に人が検出された場合
        if len(result.keypoints) > 0:
            best_kpts = None
            min_hip_y_in_frame = float('inf')
            
            # 検出された全ての人(kpts)をチェックし、一番腰が高い人（Y座標が小さい人）を探す
            for kpts in result.keypoints.xy:
                kpts_np = kpts.cpu().numpy()
                l_hip, r_hip = kpts_np[11], kpts_np[12]
                
                # 腰の座標が取れている場合
                if l_hip[1] > 0 and r_hip[1] > 0:
                    hip_y = (l_hip[1] + r_hip[1]) / 2
                    if hip_y < min_hip_y_in_frame:
                        min_hip_y_in_frame = hip_y
                        best_kpts = kpts_np
            
            # そのフレーム内で一番高い人を記録
            if best_kpts is not None:
                history.append({
                    "frame": frame.copy(),
                    "kpts": best_kpts,
                    "hip_y": min_hip_y_in_frame
                })
                
    cap.release()
    
    if not history:
        return {}, None, None, None, None

    # --- ① 真のAPEX（頂点）を探す ---
    # 動画全体を通して、腰の位置(hip_y)が最も小さかった（高かった）フレームを特定
    apex_idx = min(range(len(history)), key=lambda i: history[i]["hip_y"])
    apex_data = history[apex_idx]
    apex_frame, apex_kpts = apex_data["frame"], apex_data["kpts"]
    
    # --- ② スナップダウン（下降）を探す ---
    # APEX以降のフレームだけを見て、足首が膝より下になった瞬間を探す
    snap_frame, snap_kpts = None, None
    for i in range(apex_idx + 1, len(history)):
        data = history[i]
        kpts = data["kpts"]
        l_knee, r_knee = kpts[13], kpts[14]
        l_ankle, r_ankle = kpts[15], kpts[16]
        
        # 足首(Y座標)が膝(Y座標)より大きくなった＝脚が下りた（閉じた）
        if l_ankle[1] > l_knee[1] and r_ankle[1] > r_knee[1]:
            snap_frame = data["frame"]
            snap_kpts = kpts
            break
            
    # （保険）もし見つからなかったら、APEXから数フレーム後をスナップダウンとする
    if snap_frame is None and len(history) > apex_idx + 5:
        snap_frame = history[apex_idx + 5]["frame"]
        snap_kpts = history[apex_idx + 5]["kpts"]

    # --- ③ 角度の計算 ---
    metrics = {}
    if apex_kpts is not None and snap_kpts is not None:
        metrics["r_knee_angle"] = calculate_angle(apex_kpts[12], apex_kpts[14], apex_kpts[16])
        metrics["l_knee_angle"] = calculate_angle(apex_kpts[11], apex_kpts[13], apex_kpts[15])
        
        r_waist = calculate_angle(snap_kpts[6], snap_kpts[12], snap_kpts[14])
        l_waist = calculate_angle(snap_kpts[5], snap_kpts[11], snap_kpts[13])
        metrics["waist_flexion"] = (r_waist + l_waist) / 2
        
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
                
                img_apex = draw_custom_keypoints(apex_frame, apex_kpts, "APEX (Peak)")
                img_snap = draw_custom_keypoints(snap_frame, snap_kpts, "SNAP DOWN")
                
                st.success("一連のフォームを診断しました！")
                
                col_img1, col_img2 = st.columns(2)
                with col_img1:
                    st.image(cv2.cvtColor(img_apex, cv2.COLOR_BGR2RGB), caption="① 開脚の頂点", use_container_width=True)
                with col_img2:
                    st.image(cv2.cvtColor(img_snap, cv2.COLOR_BGR2RGB), caption="② スナップダウン時", use_container_width=True)
                
                col1, col2, col3 = st.columns(3)
                col1.metric(label="右膝角度 (頂点)", value=f"{metrics['r_knee_angle']:.1f}度")
                col2.metric(label="左膝角度 (頂点)", value=f"{metrics['l_knee_angle']:.1f}度")
                col3.metric(label="腰の角度 (下降時)", value=f"{metrics['waist_flexion']:.1f}度")
                
                st.markdown("### 💡 診断結果とアドバイス")
                for tag in tags:
                    for item in knowledge["advices"].get(tag, []):
                        st.info(f"**{item['author']}より**: {item['advice']}")
        finally:
            tfile.close()
            os.unlink(tfile.name)

