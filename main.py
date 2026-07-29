# main.py
import os
import cv2
import tempfile
import streamlit as st
from ultralytics import YOLO

# 自作モジュールのインポート
from common.motion_trimmer import detect_active_motion_range
from techniques.toe_touch_toss import select_best_frames, generate_diagnosis

# ---------------------------------------------------------
# Page Configuration
# ---------------------------------------------------------
st.set_page_config(
    page_title="Cheer AI Form Analyzer",
    page_icon="📣",
    layout="wide"
)

st.title("📣 チアリーディング AI フォーム解析")
st.write("動画をアップロードすると、AIが技の主要コマを自動選出＆フォーム診断を行います。")

# YOLOモデルのロード（キャッシュ化して2回目以降を高速化）
@st.cache_resource
def load_yolo_model():
    return YOLO("yolov8n.pt")

model = load_yolo_model()

# ---------------------------------------------------------
# Sidebar / File Uploader
# ---------------------------------------------------------
uploaded_file = st.sidebar.file_uploader("解析する動画を選択してください", type=["mp4", "mov", "avi"])

if uploaded_file is not None:
    # 一時ファイルとして動画を保存
    tfile = tempfile.NamedTemporaryFile(delete=False, suffix='.mp4')
    tfile.write(uploaded_file.read())
    video_path = tfile.name

    st.sidebar.success("動画のアップロードが完了しました！")

    if st.sidebar.button("🚀 AI解析を開始する", type="primary"):
        status_text = st.empty()
        progress_bar = st.progress(0)

        # ---------------------------------------------------------
        # 【第1段階】共通動体検知トリミング（爆速粗削り）
        # ---------------------------------------------------------
        status_text.text("⚡ [1/3] OpenCVで動画の技発生エリア（アクティブ区間）を高速スキャン中...")
        start_frame, end_frame, fps, total_frames = detect_active_motion_range(video_path)

        if start_frame is None:
            st.error("動画の読み込みに失敗しました。")
            st.stop()

        target_frame_count = end_frame - start_frame + 1

        # デバッグ用データの計算
        orig_sec = total_frames / fps
        trimmed_sec = target_frame_count / fps
        cut_frames = total_frames - target_frame_count
        reduction_rate = (cut_frames / total_frames) * 100 if total_frames > 0 else 0

        # ---------------------------------------------------------
        # 【YOLO推論】特定区間のみトラッキング（高速解析）
        # ---------------------------------------------------------
        status_text.text("🔍 [2/3] YOLOによるフライヤーの軌跡解析を実行中...")
        
        cap = cv2.VideoCapture(video_path)
        cap.set(cv2.CAP_PROP_POS_FRAMES, start_frame)

        trajectory = []
        captured_images = {}  # frame_idx -> OpenCV image
        
        processed_count = 0
        current_frame = start_frame

        while cap.isOpened() and current_frame <= end_frame:
            ret, frame = cap.read()
            if not ret:
                break

            # YOLOで人物検出
            results = model.track(frame, persist=True, verbose=False)
            
            if results and len(results[0].boxes) > 0:
                boxes = results[0].boxes
                person_boxes = [b for b in boxes if int(b.cls[0]) == 0]
                
                if person_boxes:
                    best_box = max(person_boxes, key=lambda b: float(b.conf[0]))
                    xyxy = best_box.xyxy[0].cpu().numpy()
                    conf = float(best_box.conf[0])
                    
                    x1, y1, x2, y2 = xyxy
                    center_x = (x1 + x2) / 2.0
                    center_y = (y1 + y2) / 2.0

                    data_point = {
                        'frame_idx': current_frame,
                        'bbox': [float(x1), float(y1), float(x2), float(y2)],
                        'center': [center_x, center_y],
                        'conf': conf
                    }
                    trajectory.append(data_point)
                    captured_images[current_frame] = frame.copy()

            processed_count += 1
            progress_bar.progress(min(1.0, processed_count / max(1, target_frame_count)))
            current_frame += 1

        cap.release()

        # ---------------------------------------------------------
        # 【判定ロジック】主要4コマの自動選出 ＆ AI診断レポート生成
        # ---------------------------------------------------------
        status_text.text("🧠 [3/3] AIフォーム診断およびベストコマ選出中...")
        
        f1, f2, f3, f4 = select_best_frames(trajectory)
        diagnoses = generate_diagnosis(f1, f2, f3, f4, trajectory)

        status_text.text("✅ 解析が完了しました！")
        progress_bar.progress(1.0)
        st.balloons()

        # =========================================================
        # 📊 結果表示エリア（一括出力）
        # =========================================================

        # 1. 自動トリミング・パフォーマンス結果
        st.subheader("⚡ 解析パフォーマンス（自動トリミング結果）")
        
        m1, m2, m3 = st.columns(3)
        with m1:
            st.metric(
                label="解析フレーム数", 
                value=f"{target_frame_count} コマ", 
                delta=f"-{cut_frames} コマカット",
                delta_color="normal"
            )
        with m2:
            st.metric(
                label="解析動画サイズ", 
                value=f"{trimmed_sec:.1f} 秒", 
                delta=f"-{orig_sec - trimmed_sec:.1f} 秒短縮",
                delta_color="normal"
            )
        with m3:
            st.metric(
                label="無駄時間の削減率", 
                value=f"{reduction_rate:.1f} %", 
                delta="処理速度UP!",
                delta_color="normal"
            )

        with st.expander("🛠️ トリミング詳細デバッグデータ"):
            st.write(f"- **元動画全体**: Frame 0 〜 {total_frames - 1} ({orig_sec:.2f}秒 / {total_frames}コマ)")
            st.write(f"- **抽出された範囲**: **Frame {start_frame} 〜 {end_frame}** ({trimmed_sec:.2f}秒 / {target_frame_count}コマ)")
            st.write(f"- **動画FPS**: {fps:.2f} fps")

        st.divider()

        # 2. 動作プロセスのコマ撮り
        st.subheader("📷 動作プロセスのコマ撮り")
        col1, col2, col3, col4 = st.columns(4)

        phase_info = [
            ("① 空中初期", f1, col1),
            ("② ピーク直前", f2, col2),
            ("③ 最高到達点", f3, col3),
            ("④ 反り・着地", f4, col4)
        ]

        for title, frame_data, col in phase_info:
            with col:
                st.markdown(f"#### {title}")
                if frame_data and frame_data['frame_idx'] in captured_images:
                    idx = frame_data['frame_idx']
                    img = captured_images[idx].copy()
                    
                    b = frame_data['bbox']
                    cv2.rectangle(img, (int(b[0]), int(b[1])), (int(b[2]), int(b[3])), (0, 255, 255), 3)
                    
                    label = f"Frame: {idx} | Conf: {int(frame_data['conf']*100)}%"
                    cv2.putText(img, label, (int(b[0]), max(20, int(b[1])-10)),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2)

                    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
                    st.image(img_rgb, use_container_width=True)
                else:
                    st.warning("検出不可")

        st.divider()

        # 3. AIフォーム診断レポート
        st.subheader("📋 AIフォーム診断レポート")
        for diag in diagnoses:
            st.markdown(diag)

        # 一時ファイルの削除
        try:
            os.remove(video_path)
        except Exception:
            pass

