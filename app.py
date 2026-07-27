import gc
import os
import tempfile
import urllib.request
import cv2
from huggingface_hub import hf_hub_download
import numpy as np
import streamlit as st
import torch
from easy_ViTPose import VitInference

# --------------------------------------------------
# アプリのタイトルと説明
# --------------------------------------------------
st.title("🔍 骨格推定データ欠損デバッグアプリ (軽量版)")
st.write(
    "1フレームごとに『データ不在（見失い）』か『低確信度（低Score）』かをメモリ負荷を抑えて計測します。"
)

# --------------------------------------------------
# サイドバー設定
# --------------------------------------------------
st.sidebar.title("⚙️ 解析パラメータ設定")

conf_threshold = st.sidebar.slider(
    "検出感度 (YOLO Confidence)",
    min_value=0.10,
    max_value=0.80,
    value=0.20,
    step=0.05,
)

yolo_size = st.sidebar.selectbox(
    "YOLO入力サイズ (解像度)",
    options=[320, 640],
    index=1,
)

use_tracking = st.sidebar.checkbox(
    "動画トラッキング (ByteTrack) を有効化", value=False
)


# --------------------------------------------------
# ViTPoseモデルのロード（YOLO Medium版：省メモリ設定）
# --------------------------------------------------
@st.cache_resource
def load_vitpose_model(yolo_size_val: int, is_video_val: bool):
  device = "cuda" if torch.cuda.is_available() else "cpu"

  model_file = hf_hub_download(
      repo_id="JunkyByte/easy_ViTPose", filename="torch/coco/vitpose-s-coco.pth"
  )

  # メモリ節約のため yolov8m (Medium) を使用
  yolo_file = "yolov8m.pt"
  if not os.path.exists(yolo_file):
    url = "https://github.com/ultralytics/assets/releases/download/v8.2.0/yolov8m.pt"
    with st.spinner("🧠 省メモリ型AIモデル (YOLOv8-Medium) を準備中..."):
      urllib.request.urlretrieve(url, yolo_file)

  model = VitInference(
      model=model_file,
      yolo=yolo_file,
      model_name="s",
      dataset="coco",
      yolo_size=yolo_size_val,
      is_video=is_video_val,
      device=device,
  )
  return model


with st.spinner("AIモデルを読み込み中..."):
  model = load_vitpose_model(yolo_size, use_tracking)

try:
  model.yolo.conf = conf_threshold
except AttributeError:
  pass

# --------------------------------------------------
# 動画アップロード & デバッグ解析処理
# --------------------------------------------------
st.subheader("📁 メディアのアップロード")
uploaded_file = st.file_uploader(
    "デバッグしたい動画を選択してください", type=["mp4", "mov", "avi"]
)

if uploaded_file is not None:
  st.video(uploaded_file)

  if st.button("🚨 デバッグ解析を開始"):
    tfile = tempfile.NamedTemporaryFile(delete=False, suffix=".mp4")
    tfile.write(uploaded_file.read())

    cap = cv2.VideoCapture(tfile.name)

    # レイアウトの構築
    col1, col2 = st.columns([3, 2])
    with col1:
      st_frame = st.empty()
    with col2:
      st.markdown("### 📊 デバッグ状態")
      st_status = st.empty()
      st_metrics = st.empty()

    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    progress_bar = st.progress(0)

    missing_frames = 0
    frame_count = 0

    while cap.isOpened():
      ret, frame = cap.read()
      if not ret:
        break

      pts = model.inference(frame)
      raw_out_frame = model.draw()

      if raw_out_frame is None:
        out_frame = frame.copy()
      else:
        out_frame = np.ascontiguousarray(raw_out_frame, dtype=np.uint8)

      frame_count += 1
      is_missing = len(pts) == 0

      if is_missing:
        missing_frames += 1
        st_status.error(
            f"❌ **Frame {frame_count}**: 人物未検出 (データ空っぽ)"
        )

        cv2.putText(
            out_frame,
            "MISSING (No Detection)",
            (30, 50),
            cv2.FONT_HERSHEY_SIMPLEX,
            1.0,
            (0, 0, 255),
            3,
        )
      else:
        st_status.success(
            f"✅ **Frame {frame_count}**: 人物検出完了 ({len(pts)}人)"
        )

        try:
          kpts_data = list(pts.values())[0]
          if (
              isinstance(kpts_data, dict)
              and "keypoints" in kpts_data
          ):
            kpts = np.array(kpts_data["keypoints"])
          else:
            kpts = np.array(kpts_data)

          if kpts.ndim == 2 and kpts.shape[1] >= 3:
            confs = kpts[:, 2]
          else:
            confs = np.ones(17)
        except Exception:
          confs = np.ones(17)

        avg_conf = float(np.mean(confs))
        min_conf = float(np.min(confs))

        cv2.putText(
            out_frame,
            f"Avg Conf: {avg_conf:.2f}",
            (30, 50),
            cv2.FONT_HERSHEY_SIMPLEX,
            1.0,
            (0, 255, 0),
            2,
        )

        st_metrics.metric(
            label="現在のキーポイント平均確信度",
            value=f"{avg_conf * 100:.1f}%",
            delta=f"最低関節: {min_conf * 100:.1f}%",
        )

      out_frame_rgb = cv2.cvtColor(out_frame, cv2.COLOR_BGR2RGB)
      st_frame.image(out_frame_rgb, channels="RGB", use_container_width=True)

      if total_frames > 0:
        progress_bar.progress(min(frame_count / total_frames, 1.0))

      # メモリ解放処理
      del raw_out_frame, out_frame, out_frame_rgb, pts
      if frame_count % 10 == 0:
        gc.collect()

    cap.release()

    # --------------------------------------------------
    # 最終分析サマリーレポート
    # --------------------------------------------------
    st.markdown("---")
    st.subheader("📋 デバッグ分析結果サマリー")

    missing_rate = (
        (missing_frames / frame_count * 100) if frame_count > 0 else 0
    )

    col_a, col_b, col_c = st.columns(3)
    col_a.metric("総フレーム数", f"{frame_count} frames")
    col_b.metric("データ消失（未検出）コマ数", f"{missing_frames} frames")
    col_c.metric("データ消失率", f"{missing_rate:.1f}%")

    if missing_rate > 15:
      st.warning(
          "⚠️ **主な原因: 人物検出（YOLO）の途切れ**\n"
          f"全体の {missing_rate:.1f}% のコマでAIが人を完全に見失っています。"
          "前後のデータ補間（Interpolation）が有効です。"
      )
    else:
      st.info(
          "💡 **主な原因: 関節確信度（Confidence）の低下**\n"
          "人物自体は見失っていませんが、関節の確信度が低いため描画ルールにより消えています。"
          "補間処理でキレイに接続可能です。"
      )

