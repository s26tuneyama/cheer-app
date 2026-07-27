import tempfile
import cv2
from huggingface_hub import hf_hub_download
import numpy as np
import streamlit as st
import torch
from easy_ViTPose import VitInference

# アプリのタイトルと説明
st.title("🤸‍♀️ チアリーディング骨格検出アプリ (ViTPose版)")
st.write(
    "Vision Transformer"
    " と動画トラッキングを使用し、空中トスやブレに強く骨格を追従します。"
)

# --------------------------------------------------
# サイドバー設定 (感度 & トラッキング)
# --------------------------------------------------
st.sidebar.title("⚙️ 解析パラメータ設定")

# 1. 検出感度スライダー (低いほど空中ブレでも見失わない)
conf_threshold = st.sidebar.slider(
    "検出感度 (YOLO Confidence)",
    min_value=0.10,
    max_value=0.80,
    value=0.20,
    step=0.05,
    help=(
        "値を下げると(例:"
        " 0.15〜0.25)、高速移動やブレで不鮮明になったフライヤーも見失わずに検出し続けます。"
    ),
)

# 2. 解像度設定
yolo_size = st.sidebar.selectbox(
    "YOLO入力サイズ (解像度)",
    options=[320, 640],
    index=1,
    help="640にすると遠くのフライヤーや複雑な体勢の検出精度が上がります。",
)

# 3. トラッキング有効化スイッチ
use_tracking = st.sidebar.checkbox(
    "動画トラッキング (ByteTrack) を有効化",
    value=True,
    help="フレーム間で同一人物を識別・追跡し、一瞬の骨格スキップを強力に補正します。",
)

# --------------------------------------------------
# ViTPoseモデルのロード（キャッシュ機能つき）
# --------------------------------------------------



@st.cache_resource
def load_vitpose_model(yolo_size_val: int, is_video_val: bool):
  device = "cuda" if torch.cuda.is_available() else "cpu"

  model_file = hf_hub_download(
      repo_id="JunkyByte/easy_ViTPose", filename="torch/coco/vitpose-s-coco.pth"
  )

  # YOLOモデルを 's' (Small) から 'l' (Large) に変更して変形姿勢への追従力を強化
  yolo_file = hf_hub_download(
      repo_id="JunkyByte/easy_ViTPose", filename="yolov8/yolov8l.pt"
  )

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


# モデル読み込み
with st.spinner("AIモデルを読み込み中..."):
  model = load_vitpose_model(yolo_size, use_tracking)

# 検出感度（Confidence）の動的反映
try:
  model.yolo.conf = conf_threshold
except AttributeError:
  pass

# --------------------------------------------------
# ファイルのアップロード & 解析処理
# --------------------------------------------------
st.subheader("📁 メディアのアップロード")
uploaded_file = st.file_uploader(
    "解析したい動画（MP4/MOV）または画像（JPG/PNG）を選択してください",
    type=["mp4", "mov", "avi", "jpg", "png", "jpeg"],
)

if uploaded_file is not None:
  file_type = uploaded_file.name.split(".")[-1].lower()

  # 【1】 画像ファイルの場合
  if file_type in ["jpg", "png", "jpeg"]:
    st.image(
        uploaded_file, caption="アップロード画像", use_container_width=True
    )

    if st.button("✨ 骨格検出を実行 (画像)"):
      with st.spinner("AI骨格検出を実行中..."):
        file_bytes = np.asarray(
            bytearray(uploaded_file.read()), dtype=np.uint8
        )
        img = cv2.imdecode(file_bytes, cv2.IMREAD_COLOR)

        pts = model.inference(img)
        res_img = model.draw()

        res_img_rgb = cv2.cvtColor(res_img, cv2.COLOR_BGR2RGB)
        st.image(
            res_img_rgb, caption="解析完了（結果）", use_container_width=True
        )
        st.success("骨格検出が完了しました！")

  # 【2】 動画ファイルの場合
  else:
    st.video(uploaded_file)

    if st.button("🚀 骨格検出を開始 (動画解析)"):
      tfile = tempfile.NamedTemporaryFile(delete=False, suffix=".mp4")
      tfile.write(uploaded_file.read())

      cap = cv2.VideoCapture(tfile.name)
      st_frame = st.empty()

      total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
      progress_bar = st.progress(0)

      # トラッカーが存在する場合は初期化
      if hasattr(model, "tracker") and model.tracker is not None:
        try:
          model.tracker.reset()
        except AttributeError:
          pass

      frame_count = 0
      while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
          break

        pts = model.inference(frame)
        out_frame = model.draw()

        out_frame_rgb = cv2.cvtColor(out_frame, cv2.COLOR_BGR2RGB)
        st_frame.image(out_frame_rgb, channels="RGB", use_container_width=True)

        frame_count += 1
        if total_frames > 0:
          progress_bar.progress(min(frame_count / total_frames, 1.0))

      cap.release()
      st.success("🎉 動画の全フレーム解析が完了しました！")
