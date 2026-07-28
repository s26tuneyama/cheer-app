import streamlit as st
import cv2
import tempfile
import os
from cheer_logic import analyze_cheer_flyer_descent, render_flyer_capture
from ultralytics import YOLO

st.set_page_config(page_title="Cheer Flyer Analyzer", layout="wide")

st.title("📣 チアリーディング フライヤーフォーム解析")
st.caption("ジャンプの最高到達点（最大開脚）と、降下フェーズでの姿勢・スナップを自動分析します。")

# サイドバー設定
st.sidebar.header("⚡ 検出パラメーター設定")
min_conf = st.sidebar.slider("AI全体感度 (Conf)", 0.05, 0.50, 0.15, 0.05)
min_peak_conf = st.sidebar.slider("ピーク検知の最小確信度", 0.10, 0.50, 0.35, 0.05)

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
        with st.spinner("YOLO検出 ＆ MediaPipe骨格解析を実行中..."):
            cap = cv2.VideoCapture(video_path)
            raw_frames = []
            f_idx = 0

            while cap.isOpened():
                ret, frame = cap.read()
                if not ret: break

                h, w, _ = frame.shape
                results = yolo_model(frame, verbose=False, conf=min_conf)
                
                detections = []
                for box in results[0].boxes:
                    cls_id = int(box.cls[0])
                    if cls_id == 0:  # 人間 (Person)
                        b = box.xyxy[0].cpu().numpy()
                        conf = float(box.conf[0])
                        cx, cy = (b[0] + b[2]) / 2, (b[1] + b[3]) / 2
                        
                        # 画面端チェック
                        is_edge = (b[0] < w * 0.05) or (b[2] > w * 0.95)
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

            cap.release()

            # フライヤー軌道＆姿勢の解析
            trajectory = analyze_cheer_flyer_descent(
                video_path, 
                raw_frames, 
                min_peak_conf=min_peak_conf
            )

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

            # 2. 降下時の姿勢（スナップ・腰の伸ばし）
            # 降下中のフレームから、脚を閉じ始めている・あるいは降下中盤のコマを選択
            descent_data = None
            if len(trajectory) > 1:
                # 降下中の後半または開脚角度が狭まってきたコマを取得
                descent_candidates = trajectory[1:]
                # 足が閉じ始めているコマ（split_angleが最小）をピックアップ
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

