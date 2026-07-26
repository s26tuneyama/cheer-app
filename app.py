import streamlit as st
import cv2
import numpy as np
import tempfile
import os
import json
from ultralytics import YOLO

# ==========================================
# 1. 知識ベース（ルールとアドバイス）
# ※ここは一切変更なし（アーキテクチャの完全分離）
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
# 2. 処理機構（ロジック層）★オントロジー（制約）を追加！
# ==========================================
def calculate_angle(a, b, c):
    a, b, c = np.array(a), np.array(b), np.array(c)
    ba = a - b
    bc = c - b
    cosine_angle = np.dot(ba, bc) / (np.linalg.norm(ba) * np.linalg.norm(bc))
    return np.degrees(np.arccos(np.clip(cosine_angle, -1.0, 1.0)))

def get_body_axis(kpts, l_idx, r_idx, min_conf=0.5, expected_y_below=None):
    """
    【オントロジー関数】
    左右の関節データから、信頼できる「体の中心点」を推測する。
    物理的にあり得ない点（expected_y_below）はYOLOが確信していても除外する。
    """
    valid_pts = []
    for idx in (l_idx, r_idx):
        pt = kpts[idx]
        if pt[2] > min_conf: # 確信度が一定以上
            # 物理制約：特定のY座標（例：腰）より下にあるべきパーツか？
            if expected_y_below is None or pt[1] > expected_y_below:
                valid_pts.append(pt[:2])
                
    if len(valid_pts) == 2:
        return (valid_pts[0] + valid_pts[1]) / 2 # 両方見えれば中点
    elif len(valid_pts) == 1:
        return valid_pts[0] # 片方だけならそれを中心軸とみなす
    return None # 両方見えない、または物理法則違反

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
            
            for kpts in result.keypoints.data:
                kpts_np = kpts.cpu().numpy()
                l_hip, r_hip = kpts_np[11], kpts_np[12]
                
                if l_hip[2] > 0.5 and r_hip[2] > 0.5:
                    hip_y = (l_hip[1] + r_hip[1]) / 2
                    if hip_y < min_hip_y_in_frame:
                        min_hip_y_in_frame = hip_y
                        best_kpts = kpts_np
            
            if best_kpts is not None:
                history.append({"frame": frame.copy(), "kpts": best_kpts, "hip_y": min_hip_y_in_frame})
                
    cap.release()
    
    if not history:
        return {}, None, None, None, None

    # --- ① 真のAPEX（頂点） ---
    apex_idx = min(range(len(history)), key=lambda i: history[i]["hip_y"])
    apex_data = history[apex_idx]
    apex_frame, apex_kpts = apex_data["frame"], apex_data["kpts"]
    
    # --- ② スナップダウン（下降） ---
    snap_frame, snap_kpts = None, None
    for i in range(apex_idx + 1, len(history)):
        data = history[i]
        kpts = data["kpts"]
        
        c_hip = get_body_axis(kpts, 11, 12, min_conf=0.5)
        c_ankle = get_body_axis(kpts, 15, 16, min_conf=0.4, expected_y_below=c_hip[1] if c_hip is not None else None)
        
        # 足首が腰より下で確認できた瞬間をスナップダウンとする
        if c_hip is not None and c_ankle is not None and c_ankle[1] > c_hip[1] + 20:
            snap_frame = data["frame"]
            snap_kpts = kpts
            break
            
    if snap_frame is None and len(history) > apex_idx + 5:
        snap_frame = history[apex_idx + 5]["frame"]
        snap_kpts = history[apex_idx + 5]["kpts"]

    # --- ③ 角度の計算（オントロジー適用） ---
    metrics = {}
    if apex_kpts is not None and snap_kpts is not None:
        metrics["r_knee_angle"] = calculate_angle(apex_kpts[12][:2], apex_kpts[14][:2], apex_kpts[16][:2])
        metrics["l_knee_angle"] = calculate_angle(apex_kpts[11][:2], apex_kpts[13][:2], apex_kpts[15][:2])
        
        # スナップダウンの腰角度は「体の中心軸」で計算する
        c_shoulder = get_body_axis(snap_kpts, 5, 6)
        c_hip = get_body_axis(snap_kpts, 11, 12)
        # 膝は「絶対に腰より下にある」という制約をかける
        c_knee = get_body_axis(snap_kpts, 13, 14, expected_y_below=c_hip[1] if c_hip is not None else None)
        
        if c_shoulder is not None and c_hip is not None and c_knee is not None:
            metrics["waist_flexion"] = calculate_angle(c_shoulder, c_hip, c_knee)
        else:
            # ベースに完全に隠れるなどして推測不能な場合は、一直線(180度)とみなす
            metrics["waist_flexion"] = 180.0 
        
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
    
    # 描画時も「あり得ないノイズ」を描かないように軽くフィルタリング
    hip_y = (keypoints[11][1] + keypoints[12][1]) / 2 if (keypoints[11][2]>0.5 and keypoints[12][2]>0.5) else 0
    
    def is_valid_draw(idx, is_lower_body=False):
        if keypoints[idx][2] < 0.4: return False
        if is_lower_body and hip_y > 0 and phase_name == "SNAP DOWN":
            if keypoints[idx][1] < hip_y: return False # スナップダウン中に腰より上にある下半身は描かない
        return True

    # 線の描画（有効な点だけ繋ぐ）
    pts_r = [(int(keypoints[i][0]), int(keypoints[i][1])) for i in [6, 12, 14, 16] if is_valid_draw(i, i in [14, 16])]
    pts_l = [(int(keypoints[i][0]), int(keypoints[i][1])) for i in [5, 11, 13, 15] if is_valid_draw(i, i in [13, 15])]
    
    for pts in [pts_r, pts_l]:
        for i in range(len(pts) - 1):
            cv2.line(img_draw, pts[i], pts[i+1], (255, 255, 255), 2)
            cv2.circle(img_draw, pts[i], 6, (0, 255, 255), -1)
        if pts:
            cv2.circle(img_draw, pts[-1], 6, (0, 255, 255), -1)
            
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

