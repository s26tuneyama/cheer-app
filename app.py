# app.py
import os
import tempfile
import cv2
import streamlit as st
from ultralytics import YOLO

from cheer_core import analyze_cheer_motion, render_flyer_capture
from techniques import toe_touch_jump

# 1. ページ基本設定
st.set_page_config(page_title="Cheer Form Analyzer", layout="wide")
st.title("📣 Cheer AI Form Analyzer")
st.caption("トータッチ・ジャンプの最高到達点と着地フォームをAIが判定します")

# 2. YOLOモデルの読み込み（キャッシュ化して高速化）
@st.cache_resource
def load_model():
    # 人体検出用のYOLOモデル（軽量なyolov8nを使用）
    return YOLO("yolov8n.pt")

model = load_model()

# 3. サイドバー設定
st.sidebar.header("⚙️ 解析設定")
selected_technique = st.sidebar.selectbox(
    "解析する技を選択",
    ["トータッチ・ジャンプ"]
)

# モジュールマッピング
tech_module = toe_touch_jump

# 4. メイン画面：動画アップロード
uploaded_file = st.file_uploader("解析したい演技動画（MP4 / MOV）をアップロードしてください", type=["mp4", "mov", "avi"])

if uploaded_file is not None:
    # 一時ファイルとして動画を保存
    tfile = tempfile.NamedTemporaryFile(delete=False, suffix='.mp4')
    tfile.write(uploaded_file.read())
    video_path = tfile.name

    st.video(video_path)
    
    if st.button("🚀 AIフォーム解析を実行", type="primary"):
        with st.spinner("動画をフレーム解析中...（YOLO & MediaPipe処理）"):
            
            # --- A. 動画全フレームからYOLOで人体バウンディングボックスを検出 ---
            cap = cv2.VideoCapture(video_path)
            fps = cap.get(cv2.CAP_PROP_FPS)
            f_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
            f_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
            
            raw_frames = []
            frame_idx = 0

            while cap.isOpened():
                ret, frame = cap.read()
                if not ret:
                    break

                # YOLO推論 (class 0: person)
                results = model(frame, verbose=False, classes=[0])
                detections = []

                for r in results:
                    for box in r.boxes:
                        x1, y1, x2, y2 = box.xyxy[0].tolist()
                        conf = float(box.conf[0])
                        bw, bh = x2 - x1, y2 - y1
                        
                        # 画面端の誤検出フラグ
                        is_edge = (x1 < 5 or y1 < 5 or x2 > (f_width - 5) or y2 > (f_height - 5))

                        detections.append({
                            'bbox': [x1, y1, x2, y2],
                            'conf': conf,
                            'center': ((x1 + x2) / 2.0, (y1 + y2) / 2.0),
                            'box_height': bh,
                            'is_edge': is_edge
                        })

                raw_frames.append({
                    'frame_idx': frame_idx,
                    'frame_height': f_height,
                    'detections': detections
                })
                frame_idx += 1

            cap.release()

            # --- B. 共通トラッキング＆MediaPipe解析 ---
            trajectory = analyze_cheer_motion(video_path, raw_frames)

            if not trajectory:
                st.error("⚠️ ジャンプ動作または選手を正常に検出できませんでした。別の動画でお試しください。")
            else:
                # --- C. 技ごとのベストフレーム選出 ---
                peak_data, descent_data = tech_module.select_best_frames(trajectory)

                # --- D. 画像のレンダリング（骨格描画） ---
                peak_img = render_flyer_capture(video_path, peak_data) if peak_data else None
                descent_img = render_flyer_capture(video_path, descent_data) if descent_data else None

                # --- E. 結果表示 ---
                st.subheader("📸 キャプチャ・骨格判定")
                col1, col2 = st.columns(2)

                with col1:
                    st.markdown("### 🔝 【最高到達点】")
                    if peak_img is not None:
                        st.image(peak_img, use_column_width=True)
                    else:
                        st.write("最高到達点の画像を取得できませんでした")

                with col2:
                    st.markdown("### 🛬 【着地】")
                    if descent_img is not None:
                        st.image(descent_img, use_column_width=True)
                    else:
                        st.write("着地画像の画像を取得できませんでした")

                # --- F. AIフォーム診断レポート表示 ---
                st.markdown("---")
                st.subheader("📋 AIフォーム診断レポート")
                
                diagnoses = tech_module.generate_diagnosis(peak_data, descent_data)
                
                for item in diagnoses:
                    if item.startswith("###") or item == "---":
                        st.markdown(item)
                    elif "💡" in item or "🚨" in item:
                        st.warning(item)
                    else:
                        st.success(item)

    # 一時ファイルの削除（クリーンアップ）
    try:
        os.remove(video_path)
    except Exception:
        pass

