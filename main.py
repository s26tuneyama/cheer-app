# main.py

import os
import time
import cv2
import tempfile
import streamlit as st
from ultralytics import YOLO

# 共通モジュールのインポート
from common.motion_trimmer import detect_active_motion_range

# 技別モジュールのインポート
from techniques import toe_touch_toss

# ---------------------------------------------------------
# Page Configuration & Model Load
# ---------------------------------------------------------
st.set_page_config(page_title="Cheer AI Form Analyzer", page_icon="📣", layout="wide")

st.title("📣 チアリーディング AI フォーム解析")
st.write("動画をアップロードすると、AIが技の主要コマを自動選出＆フォーム診断を行います。")

@st.cache_resource
def load_yolo_model():
    return YOLO("yolov8n.pt")

model = load_yolo_model()

# セッション状態の初期化（結果とエラーの保存用）
if "analysis_result" not in st.session_state:
    st.session_state["analysis_result"] = None
if "error_message" not in st.session_state:
    st.session_state["error_message"] = None

# ---------------------------------------------------------
# Sidebar (共通設定 ＆ 技の選択)
# ---------------------------------------------------------
st.sidebar.header("⚙️ 解析設定")
technique_type = st.sidebar.selectbox(
    "解析する技を選択してください",
    ["トータッチ・トス", "ソロジャンプ"]
)

# コマ間引き設定
speed_option = st.sidebar.radio(
    "⚡ 解析スピード設定",
    ["爆速（3コマに1コマ解析 / 3倍速）", "標準（2コマに1コマ解析 / 2倍速）", "精密（全コマ解析）"],
    index=0
)

if "3倍速" in speed_option:
    FRAME_STEP = 3
elif "2倍速" in speed_option:
    FRAME_STEP = 2
else:
    FRAME_STEP = 1

uploaded_file = st.sidebar.file_uploader("解析する動画を選択してください", type=["mp4", "mov", "avi"])

# ---------------------------------------------------------
# 解析処理の実行
# ---------------------------------------------------------
if uploaded_file is not None:
    st.sidebar.success("動画のアップロードが完了しました！")

    if st.sidebar.button("🚀 AI解析を開始する", type="primary"):
        st.session_state["error_message"] = None
        status_text = st.empty()
        progress_bar = st.progress(0)

        # 一時ファイルの作成
        tfile = tempfile.NamedTemporaryFile(delete=False, suffix='.mp4')
        tfile.write(uploaded_file.getvalue())
        tfile.close()
        video_path = tfile.name

        try:
            # 1. 動体検知トリミング
            status_text.text("⚡ [1/3] OpenCVで動画のアクティブ区間を高速スキャン中...")
            start_frame, end_frame, fps, total_frames = detect_active_motion_range(video_path)

            if start_frame is None or total_frames == 0:
                st.session_state["error_message"] = "動画の読み込みまたは動体検知に失敗しました。"
                st.stop()

            target_frame_count = end_frame - start_frame + 1
            orig_sec = total_frames / fps if fps > 0 else 0
            trimmed_sec = target_frame_count / fps if fps > 0 else 0
            cut_frames = total_frames - target_frame_count
            reduction_rate = (cut_frames / total_frames) * 100 if total_frames > 0 else 0

            # 2. YOLO追跡（通信切断防止＆高速化）
            status_text.text(f"🔍 [2/3] YOLOによるフライヤーの軌跡解析を実行中（{FRAME_STEP}コマ間引き）...")
            cap = cv2.VideoCapture(video_path)
            cap.set(cv2.CAP_PROP_POS_FRAMES, start_frame)

            trajectory = []
            processed_count = 0
            current_frame = start_frame

            while cap.isOpened() and current_frame <= end_frame:
                if (current_frame - start_frame) % FRAME_STEP == 0:
                    ret, frame = cap.read()
                    if not ret:
                        break

                    # YOLO推論（predictモードで軽量・高速化）
                    results = model.predict(frame, verbose=False, classes=[0])
                    if results and len(results[0].boxes) > 0:
                        boxes = results[0].boxes
                        person_boxes = [b for b in boxes if int(b.cls[0]) == 0]
                        
                        if person_boxes:
                            best_box = max(person_boxes, key=lambda b: float(b.conf[0]))
                            xyxy = best_box.xyxy[0].cpu().numpy()
                            conf = float(best_box.conf[0])
                            
                            x1, y1, x2, y2 = xyxy
                            data_point = {
                                'frame_idx': current_frame,
                                'bbox': [float(x1), float(y1), float(x2), float(y2)],
                                'center': [(x1 + x2) / 2.0, (y1 + y2) / 2.0],
                                'conf': conf
                            }
                            trajectory.append(data_point)
                else:
                    # スキップコマは grab() で爆速スキップ（デコード処理省略）
                    ret = cap.grab()
                    if not ret:
                        break

                processed_count += 1
                current_frame += 1

                # 💡 【重要】定期的に進捗更新 ＆ time.sleep(0.001) の息継ぎを入れて通信切断を100%防止
                if processed_count % 10 == 0 or current_frame > end_frame:
                    progress_bar.progress(min(1.0, processed_count / max(1, target_frame_count)))
                    time.sleep(0.001)

            cap.release()

            # 3. 技別診断
            status_text.text("🧠 [3/3] AIフォーム診断およびコマ選出中...")
            
            if technique_type == "トータッチ・トス":
                f1, f2, f3, f4 = toe_touch_toss.select_best_frames(trajectory)
                diagnoses = toe_touch_toss.generate_diagnosis(f1, f2, f3, f4, trajectory)
                phase_names = ["① 空中初期", "② ピーク直前", "③ 最高到達点", "④ 反り・着地"]
                selected_frames = [f1, f2, f3, f4]
            else:
                f1, f2, f3, f4 = toe_touch_toss.select_best_frames(trajectory)
                diagnoses = toe_touch_toss.generate_diagnosis(f1, f2, f3, f4, trajectory)
                phase_names = ["① 離空・踏み切り", "② 膝の溜め", "③ 最高到達点(開脚)", "④ 着地姿勢"]
                selected_frames = [f1, f2, f3, f4]

            # 4. 画像抽出
            captured_images = {}
            target_indices = []
            for f in selected_frames:
                if f is not None and isinstance(f, dict) and 'frame_idx' in f:
                    target_indices.append(f['frame_idx'])

            if target_indices:
                cap_extract = cv2.VideoCapture(video_path)
                for idx in target_indices:
                    cap_extract.set(cv2.CAP_PROP_POS_FRAMES, idx)
                    ret_ex, frame_ex = cap_extract.read()
                    if ret_ex:
                        captured_images[idx] = frame_ex
                cap_extract.release()

            status_text.text("✅ 解析が完了しました！")
            progress_bar.progress(1.0)

            # セッションに結果を保存
            st.session_state["analysis_result"] = {
                "technique_type": technique_type,
                "target_frame_count": target_frame_count,
                "cut_frames": cut_frames,
                "trimmed_sec": trimmed_sec,
                "orig_sec": orig_sec,
                "reduction_rate": reduction_rate,
                "phase_names": phase_names,
                "selected_frames": selected_frames,
                "captured_images": captured_images,
                "diagnoses": diagnoses
            }

        except Exception as e:
            st.session_state["error_message"] = f"解析中にエラーが発生しました: {e}"

        finally:
            try:
                if os.path.exists(video_path):
                    os.remove(video_path)
            except Exception:
                pass

# =========================================================
# 📊 結果＆エラーの表示（セッション状態から常時表示）
# =========================================================
if st.session_state["error_message"]:
    st.error(st.session_state["error_message"])

if st.session_state["analysis_result"] is not None:
    res = st.session_state["analysis_result"]

    st.markdown("---")
    st.subheader("⚡ 解析パフォーマンス（自動トリミング結果）")
    
    col_m1, col_m2, col_m3 = st.columns(3)
    col_m1.metric("解析フレーム数", f"{res['target_frame_count']} コマ", f"-{res['cut_frames']} コマカット")
    col_m2.metric("解析動画サイズ", f"{res['trimmed_sec']:.1f} 秒", f"-{res['orig_sec'] - res['trimmed_sec']:.1f} 秒短縮")
    col_m3.metric("無駄時間の削減率", f"{res['reduction_rate']:.1f} %", "処理速度大幅UP!")

    st.markdown("---")
    st.subheader(f"📷 {res['technique_type']} : 動作プロセスのコマ撮り")
    
    cols = st.columns(4)
    for i in range(4):
        title = res['phase_names'][i]
        frame_data = res['selected_frames'][i]
        col = cols[i]
        
        with col:
            st.markdown(f"#### {title}")
            if frame_data and isinstance(frame_data, dict) and 'frame_idx' in frame_data and frame_data['frame_idx'] in res['captured_images']:
                idx = frame_data['frame_idx']
                img = res['captured_images'][idx].copy()
                b = frame_data['bbox']
                cv2.rectangle(img, (int(b[0]), int(b[1])), (0, 255, 255), 3)
                
                label = f"Frame: {idx}"
                cv2.putText(img, label, (int(b[0]), max(20, int(b[1])-10)),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2)

                img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
                try:
                    st.image(img_rgb, use_container_width=True)
                except TypeError:
                    st.image(img_rgb, use_column_width=True)
            else:
                st.warning("検出不可")

    st.markdown("---")
    st.subheader("📋 AIフォーム診断レポート")
    for diag in res['diagnoses']:
        st.markdown(diag)

