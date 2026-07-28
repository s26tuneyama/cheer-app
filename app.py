import streamlit as st
import cv2
import tempfile
import os
from cheer_logic import analyze_cheer_motion, render_flyer_capture, generate_diagnosis
from ultralytics import YOLO

st.set_page_config(page_title="Cheer Form Analyzer", layout="wide")

st.title("📣 チアリーディング フォーム＆技診断 AI")
st.caption("AIが技の最高到達点と降下姿勢を検出し、骨格に基づいたアドバイスを自動生成します。")

# --- サイドバー設定 ---
st.sidebar.markdown("### 🎯 技の選択")
technique_type = st.sidebar.selectbox(
    "解析する技の種類を選択",
    ["トータッチ・トス（フライヤー）", "トータッチ・ジャンプ（ソロ）"]
)

st.sidebar.markdown("---")
st.sidebar.markdown("### ⚡ 処理スピード & 精度設定")
step_option = st.sidebar.select_slider(
    "解析スピード",
    options=["爆速 (4コマごと)", "高速 (3コマごと)", "標準 (2コマごと)", "最高精度 (全コマ)"],
    value="標準 (2コマごと)"
)

frame_step = 2
if "爆速" in step_option: frame_step = 4
elif "高速" in step_option: frame_step = 3
elif "標準" in step_option: frame_step = 2
elif "最高精度" in step_option: frame_step = 1

st.sidebar.markdown("### ⚙️ 検出エリア調整")
min_conf = st.sidebar.slider("AI全体感度 (Conf)", 0.01, 0.50, 0.15, 0.01)
min_peak_conf = st.sidebar.slider("ピーク検知の最小確信度", 0.05, 0.80, 0.35, 0.05)
side_crop_pct = st.sidebar.slider("左右端カット率 (%)", 0, 30, 8, 1)
top_crop_pct = st.sidebar.slider("天井ノイズカット率 (%)", 0, 20, 2, 1)

@st.cache_resource
def load_yolo_model():
    return YOLO("yolov8n.pt")

yolo_model = load_yolo_model()

uploaded_file = st.file_uploader("解析する動画ファイルをアップロードしてください (.mp4, .mov)", type=["mp4", "mov"])

if uploaded_file is not None:
    tfile = tempfile.NamedTemporaryFile(delete=False, suffix='.mp4')
    tfile.write(uploaded_file.read())
    video_path = tfile.name

    st.video(video_path)

    if st.button("🚀 フォーム診断を実行", type="primary"):
        progress_bar = st.progress(0)
        status_text = st.empty()

        cap = cv2.VideoCapture(video_path)
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        raw_frames = []
        f_idx = 0

        while cap.isOpened():
            ret, frame = cap.read()
            if not ret: break

            if f_idx % frame_step == 0:
                h, w, _ = frame.shape
                results = yolo_model(frame, verbose=False, conf=min_conf)
                
                detections = []
                for box in results[0].boxes:
                    if int(box.cls[0]) == 0:  # 人間
                        b = box.xyxy[0].cpu().numpy()
                        conf = float(box.conf[0])
                        cx, cy = (b[0] + b[2]) / 2, (b[1] + b[3]) / 2
                        
                        side_margin = w * (side_crop_pct / 100.0)
                        top_margin = h * (top_crop_pct / 100.0)
                        is_edge = (b[0] < side_margin) or (b[2] > (w - side_margin)) or (b[1] < top_margin)

                        detections.append({'bbox': b.tolist(), 'center': (cx, cy), 'conf': conf, 'box_height': b[3] - b[1], 'is_edge': is_edge})

                raw_frames.append({'frame_idx': f_idx, 'frame_height': h, 'frame_width': w, 'detections': detections})

            f_idx += 1
            if total_frames > 0:
                pct = min(1.0, f_idx / total_frames)
                progress_bar.progress(pct)
                status_text.text(f"⏳ 動画解析中... {f_idx}/{total_frames} フレーム ({int(pct * 100)}%)")

        cap.release()

        status_text.text("🦴 骨格診断 ＆ 角度評価を実行中...")
        
        trajectory = analyze_cheer_motion(
            video_path, 
            raw_frames, 
            technique_type=technique_type,
            min_peak_conf=min_peak_conf
        )

        progress_bar.progress(1.0)
        status_text.empty()

        if not trajectory:
            st.error("人物または技の動作が検出できませんでした。感度パラメータを調整してみてください。")
        else:
            st.success("解析が完了しました！")
            st.markdown("---")

            # カード選出
            peak_data = trajectory[0]
            
            # 降下カードの選出（ピークから一定以上離れたフレーム）
            peak_f_idx = peak_data['frame_idx']
            offset = 5 if "ジャンプ" in technique_type else 8
            descent_candidates = [d for d in trajectory if d['frame_idx'] >= peak_f_idx + offset]

            if descent_candidates:
                valid_splits = [d for d in descent_candidates if d.get('split_angle') is not None]
                descent_data = min(valid_splits, key=lambda x: x['split_angle']) if valid_splits else descent_candidates[-1]
            else:
                descent_data = trajectory[-1] if len(trajectory) > 1 else peak_data

            # 2列表示
            col1, col2 = st.columns(2)

            img_peak = render_flyer_capture(video_path, peak_data)
            with col1:
                st.subheader("⭐ 1. 最高到達点（最大開脚）")
                if img_peak is not None:
                    st.image(img_peak, use_column_width=True)
                    st.metric("開脚角度 (Split Angle)", f"{peak_data.get('split_angle', 'N/A')} deg")
                    st.caption(f"検出フレーム: {peak_data['frame_idx']} | 信頼度: {int(peak_data['conf']*100)}%")

            img_descent = render_flyer_capture(video_path, descent_data)
            with col2:
                st.subheader("📉 2. 降下時の姿勢（スナップ ＆ 体幹）")
                if img_descent is not None:
                    st.image(img_descent, use_column_width=True)
                    st.metric("体幹角度 (Body Posture)", f"{descent_data.get('posture_angle', 'N/A')} deg")
                    st.metric("降下時開脚度 (Snap Angle)", f"{descent_data.get('split_angle', 'N/A')} deg")
                    st.caption(f"検出フレーム: {descent_data['frame_idx']} | 信頼度: {int(descent_data['conf']*100)}%")

            # 📋 AI骨格診断フィードバックの表示
            st.markdown("---")
            st.subheader("📋 AIフォーム診断レポート")
            diagnoses = generate_diagnosis(peak_data, descent_data, technique_type)
            for diag in diagnoses:
                st.info(diag)

    if os.path.exists(video_path):
        os.remove(video_path)

