# app.py
import os
import tempfile
import cv2
import streamlit as st
from ultralytics import YOLO

from cheer_core import analyze_cheer_motion, render_flyer_capture
from techniques import toe_touch_jump, toe_touch_toss

st.set_page_config(page_title="Cheer Form Analyzer", layout="wide")
st.title("📣 Cheer AI Form Analyzer")

@st.cache_resource
def load_model():
    return YOLO("yolov8n.pt")

model = load_model()

# サイドバー技選択
st.sidebar.header("⚙️ 解析設定")
selected_technique = st.sidebar.selectbox(
    "解析する技を選択",
    ["トータッチ・トス", "トータッチ・ジャンプ"]
)

if selected_technique == "トータッチ・ジャンプ":
    tech_module = toe_touch_jump
else:
    tech_module = toe_touch_toss

st.sidebar.markdown("---")
st.sidebar.subheader("🎛️ 各種パラメータ調整")
frame_step = st.sidebar.slider("解析フレーム間隔", 1, 5, 1)
min_peak_conf = st.sidebar.slider("YOLO 検出信頼度閾値", 0.05, 0.50, 0.15, 0.05)
max_jump_distance = st.sidebar.slider("フレーム間 移動許容距離 (px)", 100, 800, 350, 50)
min_size_ratio = st.sidebar.slider("最小検出サイズ比率", 0.005, 0.05, 0.01, 0.005)

uploaded_file = st.file_uploader("解析したい演技動画（MP4 / MOV）をアップロードしてください", type=["mp4", "mov", "avi"])

if uploaded_file is not None:
    tfile = tempfile.NamedTemporaryFile(delete=False, suffix='.mp4')
    tfile.write(uploaded_file.read())
    video_path = tfile.name
    st.video(video_path)
    
    if st.button("🚀 AIフォーム解析を実行", type="primary"):
        progress_bar = st.progress(0)
        status_text = st.empty()
        
        cap = cv2.VideoCapture(video_path)
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT)) or 1
        f_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        f_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        
        raw_frames = []
        frame_idx = 0

        while cap.isOpened():
            ret, frame = cap.read()
            if not ret: break

            if frame_idx % frame_step == 0:
                results = model(frame, verbose=False, classes=[0])
                detections = []
                for r in results:
                    for box in r.boxes:
                        x1, y1, x2, y2 = box.xyxy[0].tolist()
                        conf = float(box.conf[0])
                        bw, bh = x2 - x1, y2 - y1
                        is_edge = (x1 < 5 or y1 < 5 or x2 > (f_width - 5) or y2 > (f_height - 5))
                        detections.append({
                            'bbox': [x1, y1, x2, y2],
                            'conf': conf,
                            'center': ((x1 + x2) / 2.0, (y1 + y2) / 2.0),
                            'box_height': bh,
                            'is_edge': is_edge
                        })

                raw_frames.append({'frame_idx': frame_idx, 'frame_height': f_height, 'detections': detections})

            frame_idx += 1
            progress_bar.progress(min(1.0, frame_idx / total_frames))
            status_text.text(f"⏳ YOLO人体検出中... ({frame_idx} / {total_frames})")

        cap.release()
        status_text.text("🦴 骨格・軌跡解析中...")
        
        trajectory = analyze_cheer_motion(
            video_path, raw_frames, 
            max_jump_distance=max_jump_distance, 
            min_size_ratio=min_size_ratio, 
            min_peak_conf=min_peak_conf
        )

        progress_bar.progress(1.0)
        status_text.success("✅ 解析完了")

        if not trajectory:
            st.error("⚠️ 選手を検出できませんでした。")
        else:
            st.subheader("📸 動作プロセスのコマ撮り")

            # --- 技ごとにコマ数とレイアウトを自動制御 ---
            if selected_technique == "トータッチ・トス":
                f1, f2, f3, f4 = tech_module.select_best_frames(trajectory)
                imgs = [render_flyer_capture(video_path, f) if f else None for f in (f1, f2, f3, f4)]
                titles = ["① 空中初期", "② ピーク直前", "③ 最高到達点", "④ 反り・着地"]
                cols = st.columns(4)
                for col, title, img in zip(cols, titles, imgs):
                    with col:
                        st.markdown(f"##### {title}")
                        if img is not None: st.image(img, use_column_width=True)
                diagnoses = tech_module.generate_diagnosis(f1, f2, f3, f4, trajectory)

            else:  # トータッチ・ジャンプ
                peak_f, landing_f = tech_module.select_best_frames(trajectory)
                imgs = [render_flyer_capture(video_path, f) if f else None for f in (peak_f, landing_f)]
                titles = ["① 最高到達点", "② 着地"]
                cols = st.columns(2)
                for col, title, img in zip(cols, titles, imgs):
                    with col:
                        st.markdown(f"##### {title}")
                        if img is not None: st.image(img, use_column_width=True)
                diagnoses = tech_module.generate_diagnosis(peak_f, landing_f, trajectory)

            st.markdown("---")
            st.subheader("📋 AIフォーム診断レポート")
            for item in diagnoses:
                if item.startswith("###") or item == "---":
                    st.markdown(item)
                elif "💡" in item or "🚨" in item:
                    st.warning(item)
                else:
                    st.success(item)

    try:
        os.remove(video_path)
    except Exception:
        pass

