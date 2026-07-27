import gc
import os
import subprocess
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
st.title("🤸‍♀️ チアリーディング骨格検出アプリ (決定版)")
st.write(
    "ViTPose + YOLO-Large"
    " による高精度骨格検出。解析後は画面上のプレイヤーで何度も見返せます。"
)

# --------------------------------------------------
# サイドバー設定
# --------------------------------------------------
st.sidebar.title("⚙️ 解析パラメータ設定")

# 1. 検出感度スライダー
conf_threshold = st.sidebar.slider(
    "検出感度 (YOLO Confidence)",
    min_value=0.10,
    max_value=0.80,
    value=0.20,
    step=0.05,
    help="値を下げると(例: 0.15〜0.25)、高速移動やブレで不鮮明になったフライヤーも見失わずに検出し続けます。",
)

# 2. 解像度設定
yolo_size = st.sidebar.selectbox(
    "YOLO入力サイズ (解像度)",
    options=[320, 640],
    index=1,
    help="640にすると遠くのフライヤーや複雑な体勢の検出精度が上がります。",
)

# 3. トラッキング設定
use_tracking = st.sidebar.checkbox(
    "動画トラッキング (ByteTrack) を有効化",
    value=False,
    help="※何も拾えなくなる場合はチェックを外してください (感度優先モードになります)。",
)


# --------------------------------------------------
# ViTPoseモデルのロード（YOLO Large版）
# --------------------------------------------------
@st.cache_resource
def load_vitpose_model(yolo_size_val: int, is_video_val: bool):
  device = "cuda" if torch.cuda.is_available() else "cpu"

  model_file = hf_hub_download(
      repo_id="JunkyByte/easy_ViTPose", filename="torch/coco/vitpose-s-coco.pth"
  )

  yolo_file = "yolov8l.pt"
  if not os.path.exists(yolo_file):
    url = "https://github.com/ultralytics/assets/releases/download/v8.2.0/yolov8l.pt"
    with st.spinner("🧠 大型AIモデル (YOLOv8-Large) を準備中..."):
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
# メディア処理
# --------------------------------------------------
st.subheader("📁 メディアのアップロード")
uploaded_file = st.file_uploader(
    "解析したい動画（MP4/MOV）を選択してください", type=["mp4", "mov", "avi"]
)

if uploaded_file is not None:
  st.video(uploaded_file)

  if st.button("🚀 骨格検出解析を開始"):
    tfile = tempfile.NamedTemporaryFile(delete=False, suffix=".mp4")
    tfile.write(uploaded_file.read())
    input_path = tfile.name

    raw_output_path = tempfile.NamedTemporaryFile(
        delete=False, suffix=".mp4"
    ).name
    final_output_path = tempfile.NamedTemporaryFile(
        delete=False, suffix=".mp4"
    ).name

    cap = cv2.VideoCapture(input_path)
    fps = int(cap.get(cv2.CAP_PROP_FPS)) or 30
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    st_frame = st.empty()
    progress_bar = st.progress(0)

    # トラッカーが存在する場合は初期化
    if hasattr(model, "tracker") and model.tracker is not None:
      try:
        model.tracker.reset()
      except AttributeError:
        pass

    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    out = cv2.VideoWriter(raw_output_path, fourcc, fps, (width, height))

    frame_count = 0
    st.info("解析を実行中...")

    while cap.isOpened():
      ret, frame = cap.read()
      if not ret:
        break

      pts = model.inference(frame)
      raw_out = model.draw()

      if raw_out is None:
        out_frame = frame.copy()
      else:
        out_frame = np.ascontiguousarray(raw_out, dtype=np.uint8)

      out.write(out_frame)

      out_frame_rgb = cv2.cvtColor(out_frame, cv2.COLOR_BGR2RGB)
      st_frame.image(out_frame_rgb, channels="RGB", use_container_width=True)

      frame_count += 1
      if total_frames > 0:
        progress_bar.progress(min(frame_count / total_frames, 0.9))

    cap.release()
    out.release()

    # --- ブラウザ再生用にエンコード (ffmpeg) ---
    try:
      subprocess.run(
          [
              "ffmpeg",
              "-y",
              "-i",
              raw_output_path,
              "-vcodec",
              "libx264",
              "-f",
              "mp4",
              final_output_path,
          ],
          check=True,
          stdout=subprocess.DEVNULL,
          stderr=subprocess.DEVNULL,
      )
      display_path = final_output_path
    except Exception:
      display_path = raw_output_path

    progress_bar.progress(1.0)
    st.success("🎉 全フレームの解析完了！")

    # --------------------------------------------------
    # 動画プレイヤー表示 & 保存機能
    # --------------------------------------------------
    st.markdown("---")
    st.subheader("🎬 解析結果動画 (巻き戻し・スロー再生が可能)")

    with open(display_path, "rb") as video_file:
      video_bytes = video_file.read()

      st.video(video_bytes)

      st.download_button(
          label="📥 解析動画を保存（ダウンロード）",
          data=video_bytes,
          file_name="cheer_skeleton_result.mp4",
          mime="video/mp4",
      )

    gc.collect()
