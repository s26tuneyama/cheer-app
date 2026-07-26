import streamlit as st
import cv2
import numpy as np
from PIL import Image
from ultralytics import YOLO

# 画面の基本設定（スマホ表示対応）
st.set_page_config(page_title="チア トータッチ診断AI", layout="centered")

st.title("📣 チアリーディング トータッチ診断 AI")
st.write("ジャンプの一番高い位置の写真をアップロードしてね！")

# AIモデルの読み込み（初回のみ実行して効率化）
@st.cache_resource
def load_model():
    return YOLO('yolov8n-pose.pt')

model = load_model()

# 1. 知識ベース（アドバイスの一覧）
knowledge_base = {
    "KNEE_BENT": [
        {"author": "歴代キャプテン", "advice": "空中に出た瞬間、つま先を遠くに引っ張る意識で膝が伸びます！"},
        {"author": "コーチ", "advice": "ジャンプ前の沈み込みを深くして、踏み込みを強くしましょう。"}
    ],
    "PERFECT": [
        {"author": "AI診断", "advice": "素晴らしいトスです！今のフォームを身体に覚えさせましょう。"}
    ]
}

# 2. 処理機構（膝の角度計算）
def calculate_angle(a, b, c):
    a, b, c = np.array(a), np.array(b), np.array(c)
    ba = a - b
    bc = c - b
    cosine_angle = np.dot(ba, bc) / (np.linalg.norm(ba) * np.linalg.norm(bc))
    angle = np.degrees(np.arccos(np.clip(cosine_angle, -1.0, 1.0)))
    return angle

# 写真のアップロード機能
uploaded_file = st.file_uploader("写真をアップロードしてください", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    # 画像の読み込み処理
    image = Image.open(uploaded_file)
    img_array = np.array(image)
    
    with st.spinner('AIが解析しています...'):
        results = model(img_array)
        result = results[0]

        if len(result.keypoints) == 0:
            st.error("人が検出されませんでした。全身が写っている写真を選んでね。")
        else:
            # 右腰(12), 右膝(14), 右足首(16) の座標を取得
            keypoints = result.keypoints.xy[0].cpu().numpy()
            r_hip, r_knee, r_ankle = keypoints[12], keypoints[14], keypoints[16]
            
            # 角度計算
            knee_angle = calculate_angle(r_hip, r_knee, r_ankle)
            
            # 判定結果
            tag = "KNEE_BENT" if knee_angle < 160 else "PERFECT"
            
            # 画像に骨格の線を引く
            annotated_img = result.plot()
            annotated_img_rgb = cv2.cvtColor(annotated_img, cv2.COLOR_BGR2RGB)
            
            # 診断結果の表示
            st.image(annotated_img_rgb, caption="骨格検出結果", use_container_width=True)
            st.metric(label="右膝の角度", value=f"{knee_angle:.1f}度")
            
            st.markdown("### 💡 アドバイス")
            for item in knowledge_base[tag]:
                st.info(f"**{item['author']}より**: {item['advice']}")

