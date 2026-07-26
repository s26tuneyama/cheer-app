import streamlit as st
import cv2
import numpy as np
import tempfile
import os
import json
from ultralytics import YOLO

# ==========================================
# 0. フィードバック集計用の設定
# ==========================================
FEEDBACK_FILE = "feedback_counts.json"

def load_feedback():
    if os.path.exists(FEEDBACK_FILE):
        with open(FEEDBACK_FILE, "r") as f:
            return json.load(f)
    return {"helpful": 0, "not_helpful": 0}

def save_feedback(helpful_count, not_helpful_count):
    with open(FEEDBACK_FILE, "w") as f:
        json.dump({"helpful": helpful_count, "not_helpful": not_helpful_count}, f)

if "feedback_given" not in st.session_state:
    st.session_state.feedback_given = False

# ==========================================
# 1. 知識ベース（ルール・制約・アドバイス）
# ★オントロジー（物理的制約）もここに定義！
# ==========================================
KNOWLEDGE_BASE_JSON = """
{
  "ontology": {
    "phases": {
      "SNAP_DOWN": {
        "constraints": {
          "knee_must_be_below": "hip",
          "ankle_must_be_below_hip_margin": 20
        }
      }
    }
  },
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
# 2. 処理機構（汎用的なロジック層）
# ==========================================
def calculate_angle(a, b, c):
    a, b, c = np.array(a), np.array(b), np.array(c)
    ba = a - b
    bc = c - b
    cosine_angle = np.dot(ba, bc) / (np.linalg.norm(ba) * np.linalg.norm(bc))
    return np.degrees(np.arccos(np.clip(cosine_angle, -1.0, 1.0)))

def get_body_axis(kpts, l_idx, r_idx, min_conf=0.5, expected_y_below=None):
    """オントロジー制約（expected_y_below）に従って体の中心点を推測する"""
    valid_pts = []
    for idx in (l_idx, r_idx):
        pt = kpts[idx]
        if pt[2] > min_conf:
            if expected_y_below is None or pt[1] > expected_y_below:
                valid_pts.append(pt[:2])
                
    if len(valid_pts) == 2:
        return (valid_pts[0] + valid_pts[1]) / 2
    elif len(valid_pts) == 1:
        return valid_pts[0]
    return None

def extract_jump_metrics(video_path, model):
    cap = cv2.VideoCapture(video_path)
    history = []
    
    # --- 動画から「一番高い人」を全フレーム追跡 ---
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

    # --- ① 真のAPEX（頂点）の特定 ---
    apex_idx = min(range(len(history)), key=lambda i: history[i]["hip_y"])
    apex_data = history[apex_idx]
    apex_frame, apex_kpts = apex_data["frame"], apex_data["kpts"]
    
    # 知識ベース（JSON）からスナップダウン時の制約をロード
    snap_ontology = knowledge["ontology"]["phases"]["SNAP_DOWN"]["constraints"]
    ankle_margin = snap_ontology.get("ankle_must_be_below_hip_margin", 20)
    knee_below_target = snap_ontology.get("knee_must_be_below")
    
    # --- ② スナップダウン（下降）の特定 ---
    snap_frame, snap_kpts = None, None
    for i in range(apex_idx + 1, len(history)):
        data = history[i]
        kpts = data["kpts"]
        
        c_hip = get_body_axis(kpts, 11, 12, min_conf=0.5)
        
        # 制約：足首は腰より[マージン]分下にあるべき
        c_ankle = get_body_axis(kpts, 15, 16, min_conf=0.4)
        if c_hip is not None and c_ankle is not None and c_ankle[1] > c_hip[1] + ankle_margin:
            snap_frame = data["frame"]
            snap_kpts = kpts
            break
            
    if snap_frame is None and len(history) > apex_idx + 5:
        snap_frame = history[apex_idx + 5]["frame"]
        snap_kpts = history[apex_idx + 5]["kpts"]

    # --- ③ 角度の計算（オントロジー制約の適用） ---
    metrics = {}
    if apex_kpts is not None and snap_kpts is not None:
        metrics["r_knee_angle"] = calculate_angle(apex_kpts[12][:2], apex_kpts[14][:2], apex_kpts[16][:2])
        metrics["l_knee_angle"] = calculate_angle(apex_kpts[11][:2], apex_kpts[13][:2], apex_kpts[15][:2])
        
        c_shoulder = get_body_axis(snap_kpts, 5, 6)
        c_hip = get_body_axis(snap_kpts, 11, 12)
        
        # 制約：膝が腰(hip)より下にあるべきなら、それを条件にフィルター
        knee_constraint_y = c_hip[1] if (knee_below_target == "hip" and c_hip is not None) else None
        c_knee = get_body_axis(snap_kpts, 13, 14, expected_y_below=knee_constraint_y)
        
        if c_shoulder is not None and c_hip is not None and c_knee is not None:
            metrics["waist_flexion"] = calculate_angle(c_shoulder, c_hip, c_knee)
        else:
            # 物理的制約に反してパーツが見えない場合は推測不能（一直線=180度）とする
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

def draw_custom_keypoints(image, keypoints, phase_name, ontology_constraints=None):
    """描画時にもオントロジー制約を適用してAIの幻覚（無駄な線）を描かない"""
    img_draw = image.copy()
    
    hip_y = (keypoints[11][1] + keypoints[12][1]) / 2 if (keypoints[11][2]>0.5 and keypoints[12][2]>0.5) else 0
    
    def is_valid_draw(idx, is_lower_body=False):
        if keypoints[idx][2] < 0.4: return False
        
        # 知識ベースからの制約があれば適用
        if is_lower_body and ontology_constraints and hip_y > 0:
            if ontology_constraints.get("knee_must_be_below") == "hip":
                if keypoints[idx][1] < hip_y: return False 
        return True

    pts_r = [(int(keypoints[i][0]), int(keypoints[i][1])) for i in [6, 12, 14, 16] if is_valid_draw(i, i in [13, 14, 15, 16])]
    pts_l = [(int(keypoints[i][0]), int(keypoints[i][1])) for i in [5, 11, 13, 15] if is_valid_draw(i, i in [13, 14, 15, 16])]
    
    for pts in [pts_r, pts_l]:
        for i in range(len(pts) - 1):
            cv2.line(img_draw, pts[i], pts[i+1], (255, 255, 255), 2)
            cv2.circle(img_draw, pts[i], 6, (0, 255, 255), -1)
        if pts:
            cv2.circle(img_draw, pts[-1], 6, (0, 255, 255), -1)
            
    cv2.putText(img_draw, phase_name, (20, 40), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 255, 0), 3, cv2.LINE_AA)
    return img_draw

# ==========================================
# 4. Streamlit UI 構成
# ==========================================
st.set_page_config(page_title="チア トータッチ診断AI", layout="centered")
st.title("📣 チア トータッチ診断 AI")

@st.cache_resource
def load_model():
    return YOLO('yolov8s-pose.pt') 

model = load_model()
uploaded_file = st.file_uploader("動画をアップロード", type=["mp4", "mov", "avi"])

if uploaded_file is not None:
    # 新しい動画がアップロードされたらフィードバック状態をリセット
    if "current_file" not in st.session_state or st.session_state.current_file != uploaded_file.name:
        st.session_state.feedback_given = False
        st.session_state.current_file = uploaded_file.name

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
                
                # 描画時にもJSONのオントロジー制約を渡す
                snap_constraints = knowledge["ontology"]["phases"]["SNAP_DOWN"]["constraints"]
                img_apex = draw_custom_keypoints(apex_frame, apex_kpts, "APEX (Peak)")
                img_snap = draw_custom_keypoints(snap_frame, snap_kpts, "SNAP DOWN", snap_constraints)
                
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
                
                # --- フィードバック UI ---
                st.markdown("---")
                st.markdown("#### 📝 フィードバック")
                
                if not st.session_state.feedback_given:
                    st.write("このアドバイスは役に立ちましたか？")
                    col_fb1, col_fb2 = st.columns(2)
                    
                    with col_fb1:
                        if st.button("👍 役に立った！", use_container_width=True):
                            counts = load_feedback()
                            counts["helpful"] += 1
                            save_feedback(counts["helpful"], counts["not_helpful"])
                            st.session_state.feedback_given = True
                            st.rerun()
                            
                    with col_fb2:
                        if st.button("👎 いまいち", use_container_width=True):
                            counts = load_feedback()
                            counts["not_helpful"] += 1
                            save_feedback(counts["helpful"], counts["not_helpful"])
                            st.session_state.feedback_given = True
                            st.rerun()
                
                else:
                    st.success("フィードバックありがとうございます！今後の精度向上に役立てます。")
                    counts = load_feedback()
                    st.write(f"📊 **現在の評価集計**: 👍 {counts['helpful']}件 / 👎 {counts['not_helpful']}件")

        finally:
            tfile.close()
            os.unlink(tfile.name)

