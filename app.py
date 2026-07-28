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
st.caption("AIがチアリーディングのフォーム（最高到達点・着地）を自動解析・診断します")

# 2. YOLOモデルの読み込み（キャッシュ化）
@st.cache_resource
def load_model():
    return YOLO("yolov8n.pt")

model = load_model()

# 3. サイドバー設定（技の選択 ＆ 各種調整バー）
st.sidebar.header("⚙️ 解析設定")

# 技の切り替え機能
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

# ① 解析コマ数（フレーム間隔）スライダー
frame_step = st.sidebar.slider(
    "解析フレーム間隔 (コマ数)", 
    min_value=1, max_value=5, value=1, step=1,
    help="1＝全コマ解析（高精度）、2＝1コマ飛ばし（処理スピード優先）のように調整できます。"
)

# ② YOLO 検出信頼度閾値
min_peak_conf = st.sidebar.slider(
    "YOLO 検出信頼度閾値", 
    min_value=0.05, max_value=0.50, value=0.15, step=0.05,
    help="値を下げると小さく映った選手も検出できますが、誤検出が増える場合があります。"
)

# ③ 移動許容距離
max_jump_distance = st.sidebar.slider(
    "フレーム間 移動許容距離 (px)", 
    min_value=100, max_value=800, value=350, step=50,
    help="フレーム間で選手が移動できる最大ピクセル距離です。速い動きには大きめの値を設定します。"
)

# ④ 最小検出サイズ比率
min_size_ratio = st.sidebar.slider(
    "最小検出サイズ比率", 
    min_value=0.005, max_value=0.05, value=0.01, step=0.005,
    help="画面全体に対して小さすぎる誤検出オブジェクトを除外します。"
)

# 4. メイン画面：動画アップロード
uploaded_file = st.file_uploader("解析したい演技動画（MP4 / MOV）をアップロードしてください", type=["mp4", "mov", "avi"])

if uploaded_file is not None:
    tfile = tempfile.NamedTemporaryFile(delete=False, suffix='.mp4')
    tfile.write(uploaded_file.read())
    video_path = tfile.name

    st.video(video_path)
    
    if st.button("🚀 AIフォーム解析を実行", type="primary"):
        
        # --- プログレスバー ＆ ステータス表示 ---
        progress_bar = st.progress(0)
        status_text = st.empty()
        status_text.text("動画を読み込み中...")

        cap = cv2.VideoCapture(video_path)
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        if total_frames <= 0:
            total_frames = 1  # ゼロ除算防止

        f_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        f_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        
        raw_frames = []
        frame_idx = 0

        # --- A. フレーム解析（コマ数間引き＆プログレスバー更新付き） ---
        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break

            # 設定したコマ数間隔（frame_step）ごとにYOLO解析を実行
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

            # プログレスバーの更新
            frame_idx += 1
            progress = min(1.0, frame_idx / total_frames)
            progress_bar.progress(progress)
            status_text.text(f"⏳ YOLO人体検出処理中... ({frame_idx} / {total_frames} フレーム)")

        cap.release()

        # --- B. MediaPipe & 軌跡トラッキング ---
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
            st.error("⚠️ ジャンプ動作または選手を正常に検出できませんでした。サイドバーの調整バーで閾値を下げるか、別の動画でお試しください。")
        else:
            # --- C. ベストフレーム選出 ---
            peak_data, descent_data = tech_module.select_best_frames(trajectory)

            # --- D. 画像の描画 ---
            peak_img = render_flyer_capture(video_path, peak_data) if peak_data else None
            descent_img = render_flyer_capture(video_path, descent_data) if descent_data else None

            # --- E. 結果キャプチャ表示 ---
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

    # クリーンアップ
    try:
        os.remove(video_path)
    except Exception:
        pass

