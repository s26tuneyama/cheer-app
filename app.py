import streamlit as st
import cv2
import numpy as np
import tempfile
import os
import json
from ultralytics import YOLO

# ==========================================
# 1. 知識ベース（ルールとアドバイス）
# ！！！ここは前回から一切変更していません！！！（アーキテクチャの勝利です）
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
# 2. 処理機構（ロジック層）★ここを改良！
# ==========================================
def calculate_angle(a, b, c):
    # (x, y, conf) のうち (x, y) だけを使って計算するように微調整
    a, b, c = np.array(a[:2]), np.array(b[:2]), np.array(c[:2])
    ba = a - b
    bc = c - b
    cosine_angle = np.dot(ba, bc) / (np.linalg.norm(ba) * np.linalg.norm(bc))
    return np.degrees(np.arccos(np.clip(cosine_angle, -1.0, 1.0)))

def extract_jump_metrics(video_path, model):
    cap = cv2.VideoCapture(video_path)
    history = []
    
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret: break

        results = model(frame, verbose=False)
        result = results[0]
        
        if len(result.keypoints) > 0:
            best_kpts = None
            min_hip_y_in_frame = float('inf')
            
            # .xy ではなく .data を使うことで、AIの「確信度(Confidence)」も取得
            for kpts in result.keypoints.data:
                kpts_np = kpts.cpu().numpy()
                l_hip, r_hip = kpts_np[11], kpts_np[12]
                
                # 【改良1】AIの確信度が低い（0.5未満）誤認ノイズは無視する
                if l_hip[2] > 0.5 and r_hip[2] > 0.5:
                    hip_y = (l_hip[1] + r_hip[1]) / 2
                    if hip_y < min_hip_y_in_frame:
                        min_hip_y_in_frame = hip_y
                        best_kpts = kpts_np
            
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
    apex_idx = min(range(len(history)), key=lambda i: history[i]["hip_y"])
    apex_data = history[apex_idx]
    apex_frame, apex_kpts = apex_data["frame"], apex_data["kpts"]
    
    # --- ② スナップダウン（下降）を探す ---
    snap_frame, snap_kpts = None, None
    for i in range(apex_idx + 1, len(history)):
        data = history[i]
        kpts = data["kpts"]
        l_hip, r_hip = kpts[11], kpts[12]
        l_ankle, r_ankle = kpts[15], kpts[16]
        
        # 【改良2】左右の足首の距離と、左右の腰幅を計算
        ankle_dist = np.linalg.norm(l_ankle[:2] - r_ankle[:2])
        hip_width = np.linalg.norm(l_hip[:2] - r_hip[:2])
        
        # 足首が腰より下にあり、かつ「足首の距離が腰幅の1.5倍以内に近づいた（＝脚が閉じた）」瞬間をスナップダウンとする
        if l_ankle[1] > l_hip[1] and r_ankle[1] > r_hip[1] and ankle_dist < (hip_width * 1.5):
            snap_frame = data["frame"]
            snap_kpts = kpts
            break
            
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
        x, y, conf = keypoints[idx][0], keypoints[idx][1], keypoints[idx][2]
        # 確信度が極端に低い部位（0.3未満）は描画しない
        if x != 0 and y != 0 and conf > 0.3:  
            cv2.circle(img_draw, (int(x), int(y)), 6, color, -1)
            
    pts_r = [(int(keypoints[i][0]), int(keypoints[i][1])) for i in [6, 12, 14, 16] if keypoints[i][2] > 0.3]
    pts_l = [(int(keypoints[i][0]), int(keypoints[i][1])) for i in [5, 11, 13, 15] if keypoints[i][2] > 0.3]
    
    if len(pts_r) == 4:
        cv2.line(img_draw, pts_r[0], pts_r[1], (255, 255, 255), 2)
        cv2.line(img_draw, pts_r[1], pts_r[2], (255, 255, 255), 2)
        cv2.line(img_draw, pts_r[2], pts_r[3], (255, 255, 255), 2)
    if len(pts_l) == 4:
        cv2.line(img_draw, pts_l[0], pts_l[1], (255, 255, 255), 2)
        cv2.line(img_draw, pts_l[1], pts_l[2], (255, 255, 255), 2)
        cv2.line(img_draw, pts_l[2], pts_l[3], (255, 255, 255), 2)
            
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

