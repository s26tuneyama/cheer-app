# app.py
import os
import tempfile
import cv2
import streamlit as st
from ultralytics import YOLO

from cheer_core import analyze_cheer_motion, render_flyer_capture
from techniques import toe_touch_jump, toe_touch_toss

# 1. ページ基本設定
st.set_page_config(page_title="Cheer Form Analyzer", layout="wide")
st.title("📣 Cheer AI Form Analyzer")
st.caption("AIがチアリーディングのフォーム（最高到達点・動作プロセスのコマ撮り）を自動解析・診断します")

# 2. YOLOモデルの読み込み
@st.cache_resource
def load_model():
    return YOLO("yolov8n.pt")

model = load_model()

# 3. サイドバー設定
st.sidebar.header("⚙️ 解析設定")

selected_technique = st.sidebar.selectbox(
    "解析する技を選択",
    ["トータッチ・ジャンプ", "トータッチ・トス"]
)

if selected_technique == "トータッチ・ジャンプ":
    tech_module = toe_touch_jump
else:
    tech_module = toe_touch_toss

st.sidebar.markdown("---")
st.sidebar.subheader("🎛️ 各種パラメータ調整")

frame_step = st.sidebar.slider(
    "解析フレーム間隔 (コマ数)", 
    min_value=1, max_value=5, value=1, step=1,
    help="1＝全コマ解析（高精度）、2＝1コマ飛ばし（処理スピード優先）"
)

min_peak_conf = st.sidebar.slider(
    "YOLO 検出信頼度閾値", 
    min_value=0.05, max_value=0.50, value=0.15, step=0.05
)

max_jump_distance = st.sidebar.slider(
    "フレーム間 移動許容距離 (px)", 
    min_value=100, max_value=800, value=350, step=50
)

min_size_ratio = st.sidebar.slider(
    "最小検出サイズ比率", 
    min_value=0.005, max_value=0.05, value=0.01, step=0.005
)

# 4. メイン画面
uploaded_file = st.file_uploader("解析したい演技動画（MP4 / MOV）をアップロードしてください", type=["mp4", "mov", "avi"])

if uploaded_file is not None:
    tfile = tempfile.NamedTemporaryFile(delete=False, suffix='.mp4')
    tfile.write(uploaded_file.read())
    video_path = tfile.name

    st.video(video_path)
    
    if st.button("🚀 AIフォーム解析を実行", type="primary"):
        
        progress_bar = st.progress(0)
        status_text = st.empty()
        status_text.text("動画を読み込み中...")

        cap = cv2.VideoCapture(video_path)
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        if total_frames <= 0: total_frames = 1

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

                raw_frames.append({
                    'frame_idx': frame_idx,
                    'frame_height': f_height,
                    'detections': detections
                })

            frame_idx += 1
            progress = min(1.0, frame_idx / total_frames)
            progress_bar.progress(progress)
            status_text.text(f"⏳ YOLO人体検出処理中... ({frame_idx} / {total_frames} フレーム)")

        cap.release()

        status_text.text("🦴 骨格検出 ＆ 軌跡トラッキング解析中 (MediaPipe)...")
        
        trajectory = analyze_cheer_motion(
            video_path, 
            raw_frames, 
            max_jump_distance=max_jump_distance, 
            min_size_ratio=min_size_ratio, 
            min_peak_conf=min_peak_conf
        )

        progress_bar.progress(1.0)
        status_text.success("✅ 解析が完了しました！結果を表示します。")

        if not trajectory:
            st.error("⚠️ 選手を正常に検出できませんでした。パラメータを調整するか、別の動画でお試しください。")
        else:
            # --- 4コマ（コマ撮り）主要コマ選出 ---
            f1, f2, f3, f4 = tech_module.select_best_frames(trajectory)

            img1 = render_flyer_capture(video_path, f1) if f1 else None
            img2 = render_flyer_capture(video_path, f2) if f2 else None
            img3 = render_flyer_capture(video_path, f3) if f3 else None
            img4 = render_flyer_capture(video_path, f4) if f4 else None

            # --- コマ撮りシーケンス表示 ---
            st.subheader("📸 動作プロセスのコマ撮り（根拠キャプチャ）")
            c1, c2, c3, c4 = st.columns(4)

            with c1:
                st.markdown("##### ① 空中初期")
                if img1 is not None: st.image(img1, use_column_width=True)

            with c2:
                st.markdown("##### ② ピーク直前")
                if img2 is not None: st.image(img2, use_column_width=True)

            with c3:
                st.markdown("##### ③ 最高到達点")
                if img3 is not None: st.image(img3, use_column_width=True)

            with c4:
                st.markdown("##### ④ アーチ・着地")
                if img4 is not None: st.image(img4, use_column_width=True)

            # --- AIフォーム診断レポート表示 ---
            st.markdown("---")
            st.subheader("📋 AIフォーム診断レポート")
            
            diagnoses = tech_module.generate_diagnosis(f1, f2, f3, f4, trajectory)
            
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

