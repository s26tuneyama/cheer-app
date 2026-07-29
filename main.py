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
    # ※軽量化のため 'yolov8n.pt' またはお手持ちの学習済みモデルを指定
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
        st.info(f"🎬 全 {total_frames} コマ中、動きのある 【Frame {start_frame} 〜 {end_frame}】 ({target_frame_count} コマ) を解析対象に絞り込みました！")

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
            
            # 最も信頼度の高い（あるいは一番上の）人物BBoxを取得
            if results and len(results[0].boxes) > 0:
                boxes = results[0].boxes
                # 人物クラス(cls == 0)の検出結果を抽出
                person_boxes = [b for b in boxes if int(b.cls[0]) == 0]
                
                if person_boxes:
                    # 画面内で最も高い位置（Y最小）にいる人物、または自信度最高をフライヤーと判定
                    best_box = max(person_boxes, key=lambda b: float(b.conf[0]))
                    xyxy = best_box.xyxy[0].cpu().numpy()  # [x1, y1, x2, y2]
                    conf = float(best_box.conf[0])
                    
                    x1, y1, x2, y2 = xyxy
                    center_x = (x1 + x2) / 2.0
                    center_y = (y1 + y2) / 2.0

                    # 軌跡データへの格納
                    data_point = {
                        'frame_idx': current_frame,
                        'bbox': [float(x1), float(y1), float(x2), float(y2)],
                        'center': [center_x, center_y],
                        'conf': conf
                    }
                    trajectory.append(data_point)
                    captured_images[current_frame] = frame.copy()

            # プログレスバーの更新
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

        # ---------------------------------------------------------
        # 画面表示1：動作プロセスのコマ撮り
        # ---------------------------------------------------------
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
                    
                    # BBoxの描画（黄色い枠）
                    b = frame_data['bbox']
                    cv2.rectangle(img, (int(b[0]), int(b[1])), (int(b[2]), int(b[3])), (0, 255, 255), 3)
                    
                    # フレーム番号と信頼度の描画
                    label = f"Frame: {idx} | Conf: {int(frame_data['conf']*100)}%"
                    cv2.putText(img, label, (int(b[0]), max(20, int(b[1])-10)),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2)

                    # BGRからRGBに変換してStreamlit表示
                    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
                    st.image(img_rgb, use_container_width=True)
                else:
                    st.warning("検出不可")

        st.divider()

        # ---------------------------------------------------------
        # 画面表示2：AIフォーム診断レポート
        # ---------------------------------------------------------
        st.subheader("📋 AIフォーム診断レポート")
        for diag in diagnoses:
            st.markdown(diag)

        # 一時ファイルの削除
        try:
            os.remove(video_path)
        except Exception:
            pass
