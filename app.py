import streamlit as st
import cv2
import numpy as np
import tempfile
import os
from ultralytics import YOLO

# ==========================================
# 1. 知識ベース（アドバイス層）
# 将来的にデータベースに移行する部分です。
# ==========================================
knowledge_base = {
    "KNEE_BENT": [
        {"author": "歴代キャプテン", "advice": "空中に出た瞬間、つま先を遠くに引っ張る意識で膝が伸びます！"},
        {"author": "コーチ", "advice": "ジャンプ前の沈み込みを深くして、踏み込みを強くしましょう。"}
    ],
    "PERFECT": [
        {"author": "AI診断", "advice": "素晴らしいトスです！今のフォームを身体に覚えさせましょう。"}
    ]
}


# ==========================================
# 2. 処理機構（ロジック層）
# 物理的な計算やAIの処理のみを行い、テキストは持ちません。
# ==========================================
def calculate_angle(a, b, c):
    """3点から角度を計算する関数"""
    a, b, c = np.array(a), np.array(b), np.array(c)
    ba = a - b
    bc = c - b
    cosine_angle = np.dot(ba, bc) / (np.linalg.norm(ba) * np.linalg.norm(bc))
    return np.degrees(np.arccos(np.clip(cosine_angle, -1.0, 1.0)))

def get_highest_jump_frame(video_path, model):
    """動画から腰の位置が一番高い（Y座標が最小の）フレームを抽出する関数"""
    cap = cv2.VideoCapture(video_path)
    highest_frame = None
    min_y = float('inf') # 画面の上端がY=0なので、数値が小さいほど高い位置
    best_result = None
    
    frame_count = 0
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
            
        frame_count += 1
        # サーバーの負担を減らすため、2コマに1回だけAI判定を行う
        if frame_count % 2 != 0:
            continue

        # AIで骨格検出
        results = model(frame, verbose=False)
        result = results[0]
        
        if len(result.keypoints) > 0:
            keypoints = result.keypoints.xy[0].cpu().numpy()
            # 11:左腰, 12:右腰
            l_hip, r_hip = keypoints[11], keypoints[12]
            
            # 腰の高さ（Y座標）の平均を計算
            if l_hip[1] > 0 and r_hip[1] > 0:
                current_y = (l_hip[1] + r_hip[1]) / 2
                # これまでで一番高い位置なら記録を更新
                if current_y < min_y:
                    min_y = current_y
                    highest_frame = frame.copy()
                    best_result = result
                    
    cap.release()
    return highest_frame, best_result


# ==========================================
# 3. アプリの画面（表示層）
# ユーザーとのやり取りと、各層の橋渡しをします。
# ==========================================
st.set_page_config(page_title="チア トータッチ診断AI", layout="centered")

st.title("📣 チアリーディング トータッチ診断 AI")
st.write("ジャンプの動画（10秒以内）をアップロードしてね！自動で一番高い位置を見つけて診断します。")

# AIの準備（1回だけ読み込む）
@st.cache_resource
def load_model():
    return YOLO('yolov8n-pose.pt')

model = load_model()

# 動画アップロード機能
uploaded_file = st.file_uploader("動画をアップロード", type=["mp4", "mov", "avi"])

if uploaded_file is not None:
    with st.spinner('動画からジャンプの瞬間を探しています...（数十秒かかります）'):
        # アップロードされた動画をシステム内に一時保存
        tfile = tempfile.NamedTemporaryFile(delete=False, suffix='.mp4')
        tfile.write(uploaded_file.read())
        tfile.flush()
        
        try:
            # 処理機構を呼び出して最高到達点を計算
            best_frame, best_result = get_highest_jump_frame(tfile.name, model)
            
            if best_frame is None or len(best_result.keypoints) == 0:
                st.error("人が検出されませんでした。全身が写っている動画を選んでね。")
            else:
                # 関節の座標を取得（12:右腰, 14:右膝, 16:右足首）
                keypoints = best_result.keypoints.xy[0].cpu().numpy()
                r_hip, r_knee, r_ankle = keypoints[12], keypoints[14], keypoints[16]
                
                if r_hip[0] == 0 or r_knee[0] == 0 or r_ankle[0] == 0:
                    st.warning("足が見切れているため、正確に診断できませんでした。全身が収まるように撮影してください。")
                else:
                    # 処理機構で角度を計算
                    knee_angle = calculate_angle(r_hip, r_knee, r_ankle)
                    
                    # 状態のタグ付け
                    tag = "KNEE_BENT" if knee_angle < 160 else "PERFECT"
                    
                    # 画像に骨格を描画して表示用に変換
                    annotated_img = best_result.plot()
                    annotated_img_rgb = cv2.cvtColor(annotated_img, cv2.COLOR_BGR2RGB)
                    
                    # 結果の表示
                    st.success("一番高い位置を自動で抽出しました！")
                    st.image(annotated_img_rgb, caption="ジャンプの最高到達点", use_container_width=True)
                    st.metric(label="右膝の角度", value=f"{knee_angle:.1f}度")
                    
                    # 知識ベースからアドバイスを引き出して表示
                    st.markdown("### 💡 アドバイス")
                    for item in knowledge_base[tag]:
                        st.info(f"**{item['author']}より**: {item['advice']}")
        
        finally:
            # スマホやサーバーの容量を圧迫しないよう、一時保存した動画を削除
            tfile.close()
            os.unlink(tfile.name)

