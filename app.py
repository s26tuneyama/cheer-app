import streamlit as st
import cv2
import numpy as np
import tempfile
import os
import json
from ultralytics import YOLO

# ==========================================
# 1. 知識ベース（将来的に完全に別ファイル・DBになる部分）
# 「診断基準（ルール）」と「アドバイス」の両方を定義します
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
# 評価は一切行わず、「物理量（角度）」を計算して返すだけの汎用エンジン
# ==========================================
def calculate_angle(a, b, c):
    a, b, c = np.array(a), np.array(b), np.array(c)
    ba = a - b
    bc = c - b
    cosine_angle = np.dot(ba, bc) / (np.linalg.norm(ba) * np.linalg.norm(bc))
    return np.degrees(np.arccos(np.clip(cosine_angle, -1.0, 1.0)))

def extract_jump_metrics(video_path, model):
    """動画から頂点と下降時を見つけ、純粋な『数値データ』だけを抽出する"""
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
                if current_y < min_y:
                    min_y = current_y
                    apex_frame = frame.copy()
                    apex_kpts = keypoints
                    phase = "ASCENT"
                elif current_y > min_y + 20 and phase == "ASCENT":
                    phase = "DESCENT"
                    
                if phase == "DESCENT" and snap_frame is None:
                    l_knee, r_knee = keypoints[13], keypoints[14]
                    l_ankle, r_ankle = keypoints[15], keypoints[16]
                    if l_ankle[1] > l_knee[1] and r_ankle[1] > r_knee[1]:
                        snap_frame = frame.copy()
                        snap_kpts = keypoints
    cap.release()
    
    # 取得したポイントから物理量（角度）を計算して辞書で返す
    metrics = {}
    if apex_kpts is not None and snap_kpts is not None:
        metrics["r_knee_angle"] = calculate_angle(apex_kpts[12], apex_kpts[14], apex_kpts[16])
        metrics["l_knee_angle"] = calculate_angle(apex_kpts[11], apex_kpts[13], apex_kpts[15])
        
        r_waist = calculate_angle(snap_kpts[6], snap_kpts[12], snap_kpts[14])
        l_waist = calculate_angle(snap_kpts[5], snap_kpts[11], snap_kpts[13])
        metrics["waist_flexion"] = (r_waist + l_waist) / 2
        
    return metrics, apex_frame, snap_frame

# ==========================================
# 3. 汎用評価エンジン＆表示層
# 知識ベースのルールに従って、数値を自動評価する
# ==========================================
def evaluate_metrics(metrics, rules):
    """計測された数値を、知識ベースのルールと照らし合わせてタグを発行する"""
    tags = []
    for rule in rules:
        metric_val = metrics.get(rule["metric"])
        if metric_val is not None:
            # 知識ベースで「< (未満)」と定義されていれば、その計算を行う
            if rule["operator"] == "<" and metric_val < rule["threshold"]:
                tags.append(rule["tag"])
    
    if not tags:
        tags.append("PERFECT")
    return tags

st.set_page_config(page_title="汎用フォーム診断AI", layout="centered")
st.title("📣 チア トータッチ診断 AI")

@st.cache_resource
def load_model():
    return YOLO('yolov8s-pose.pt') 

model = load_model()
uploaded_file = st.file_uploader("動画をアップロード", type=["mp4", "mov", "avi"])

if uploaded_file is not None:
    with st.spinner('解析中...'):
        tfile = tempfile.NamedTemporaryFile(delete=False, suffix='.mp4')
        tfile.write(uploaded_file.read())
        tfile.flush()
        
        try:
            # 1. ロジック層に計算だけさせる
            metrics, apex_frame, snap_frame = extract_jump_metrics(tfile.name, model)
            
            if not metrics:
                st.error("解析に失敗しました。")
            else:
                # 2. 汎用エンジンでルール（知識）と照らし合わせる
                tags = evaluate_metrics(metrics, knowledge["rules"])
                
                # 3. 結果の表示
                st.success("解析完了！")
                st.write(f"📊 **計測データ**: 右膝 {metrics['r_knee_angle']:.1f}度 / 左膝 {metrics['l_knee_angle']:.1f}度 / 下降時腰角度 {metrics['waist_flexion']:.1f}度")
                
                st.markdown("### 💡 診断結果とアドバイス")
                for tag in tags:
                    for item in knowledge["advices"].get(tag, []):
                        st.info(f"**{item['author']}より**: {item['advice']}")
        finally:
            tfile.close()
            os.unlink(tfile.name)

