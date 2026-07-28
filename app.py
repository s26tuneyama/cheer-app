import streamlit as st
import cv2
import tempfile
import os
from cheer_logic import analyze_cheer_flyer_descent, render_flyer_capture
from ultralytics import YOLO

st.set_page_config(page_title="Cheer Flyer Analyzer", layout="wide")

st.title("📣 チアリーディング フライヤーフォーム解析")
st.caption("ジャンプの最高到達点（最大開脚）と、降下フェーズでの姿勢・スナップを自動分析します。")

# --- サイドバー設定（以前の設定を完全復元） ---
st.sidebar.markdown("### ⚡ 処理スピード & 精度設定")
step_option = st.sidebar.select_slider(
    "解析スピード",
    options=["爆速 (4コマごと)", "高速 (3コマごと)", "標準 (2コマごと)", "最高精度 (全コマ)"],
    value="標準 (2コマごと)"
)

frame_step = 2
if "爆速" in step_option:
    frame_step = 4
elif "高速" in step_option:
    frame_step = 3
elif "標準" in step_option:
    frame_step = 2
elif "最高精度" in step_option:
    frame_step = 1

st.sidebar.markdown("### ⚙️ 検出エリア調整")
min_conf = st.sidebar.slider("AI全体感度 (Conf)", 0.01, 0.50, 0.15, 0.01)
min_peak_conf = st.sidebar.slider("ピーク検知の最小確信度", 0.05, 0.80, 0.35, 0.05)
side_crop_pct = st.sidebar.slider("左右端カット率 (%)", 0, 30, 8, 1)
top_crop_pct = st.sidebar.slider("天井ノイズカット率 (%)", 0, 20, 2, 1)

# モデルロード（キャッシュ化）
@st.cache_resource
def load_yolo_model():
    return YOLO("yolov8n.pt")

yolo_model = load_yolo_model()

# 動画アップロード
uploaded_file = st.file_uploader("解析する動画ファイルをアップロードしてください (.mp4, .mov)", type=["mp4", "mov"])

if uploaded_file is not None:
    # 一時ファイルに保存
    tfile = tempfile.NamedTemporaryFile(delete=False, suffix='.mp4')
    tfile.write(uploaded_file.read())
    video_path = tfile.name

    st.video(video_path)

    if st.button("🚀 フォーム解析を実行", type="primary"):
        # プログレスバー & ステータステキストの初期化
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
                    cls_id = int(box.cls[0])
                    if cls_id == 0:  # 人間 (Person)
                        b = box.xyxy[0].cpu().numpy()
                        conf = float(box.conf[0])
                        cx, cy = (b[0] + b[2]) / 2, (b[1] + b[3]) / 2
                        
                        # 端・天井カット判定
                        side_margin = w * (side_crop_pct / 100.0)
                        top_margin = h * (top_crop_pct / 100.0)
                        is_edge = (b[0] < side_margin) or (b[2] > (w - side_margin)) or (b[1] < top_margin)

                        detections.append({
                            'bbox': b.tolist(),
                            'center': (cx, cy),
                            'conf': conf,
                            'box_height': b[3] - b[1],
                            'is_edge': is_edge
                        })

                raw_frames.append({
                    'frame_idx': f_idx,
                    'frame_height': h,
                    'frame_width': w,
                    'detections': detections
                })

            f_idx += 1
            
            # プログレスバーをリアルタイム更新
            if total_frames > 0:
                pct = min(1.0, f_idx / total_frames)
                progress_bar.progress(pct)
                status_text.text(f"⏳ 動画解析中... {f_idx}/{total_frames} フレーム ({int(pct * 100)}%)")

        cap.release()

        status_text.text("🦴 骨格・姿勢の精密分析を実行中...")
        
        # フライヤー軌道＆姿勢の解析
        trajectory = analyze_cheer_flyer_descent(
            video_path, 
            raw_frames, 
            min_peak_conf=min_peak_conf
        )

        progress_bar.progress(1.0)
        status_text.empty()  # 解析完了後にテキストをクリア

        if not trajectory:
            st.error("フライヤーのジャンプ技がうまく検出できませんでした。設定の感度を調整してみてください。")
        else:
            st.success("解析が完了しました！")
            st.markdown("---")

            # 2列レイアウトで結果を表示
            col1, col2 = st.columns(2)

            # 1. 最高到達点（最大開脚）
            peak_data = trajectory[0]
            img_peak = render_flyer_capture(video_path, peak_data)

            with col1:
                st.subheader("⭐ 1. 最高到達点（最大開脚）")
                if img_peak is not None:
                    st.image(img_peak, use_column_width=True)
                    st.metric("開脚角度 (Split Angle)", f"{peak_data.get('split_angle', 'N/A')} deg")
                    st.caption(f"検出フレーム: {peak_data['frame_idx']} | 信頼度: {int(peak_data['conf']*100)}%")

            # 2. 降下時の姿勢（スナップ・腰の伸び）
            descent_data = None
            if len(trajectory) > 1:
                descent_candidates = trajectory[1:]
                valid_descent = [d for d in descent_candidates if d.get('split_angle') is not None]
                if valid_descent:
                    descent_data = min(valid_descent, key=lambda x: x['split_angle'])
                else:
                    descent_data = descent_candidates[len(descent_candidates) // 2]
            else:
                descent_data = peak_data

            img_descent = render_flyer_capture(video_path, descent_data)

            with col2:
                st.subheader("📉 2. 降下時の姿勢（着地前の伸び）")
                if img_descent is not None:
                    st.image(img_descent, use_column_width=True)
                    st.metric("体幹角度 (Body Posture)", f"{descent_data.get('arch_angle', 'N/A')} deg")
                    st.metric("着地前開脚度", f"{descent_data.get('split_angle', 'N/A')} deg")
                    st.caption(f"検出フレーム: {descent_data['frame_idx']} | 信頼度: {int(descent_data['conf']*100)}%")

    # 一時ファイルの削除
    if os.path.exists(video_path):
        os.remove(video_path)

