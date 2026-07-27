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
    " を使用し、空中トスや重なり（オクルージョン）に強い姿勢推定を行います。"
)


# --------------------------------------------------
# ViTPoseモデルのロード（キャッシュして高速化）
# --------------------------------------------------
@st.cache_resource
def load_vitpose_model():
  device = "cuda" if torch.cuda.is_available() else "cpu"

  # Hugging Face からモデルファイルをダウンロード
  model_file = hf_hub_download(
      repo_id="JunkyByte/easy_ViTPose", filename="torch/coco/vitpose-s-coco.pth"
  )
  yolo_file = hf_hub_download(
      repo_id="JunkyByte/easy_ViTPose", filename="yolov8/yolov8s.pt"
  )

  # 正しいパラメータでモデル初期化
  model = VitInference(
      model=model_file,
      yolo=yolo_file,
      model_name="s",
      dataset="coco",
      yolo_size=320,
      device=device,
  )
  return model


# モデルの呼び出し
with st.spinner("AIモデルを読み込み中..."):
  model = load_vitpose_model()

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

  # 【1】 画像ファイルの場合の処理
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

        # ViTPose推論・描画
        pts = model.inference(img)
        res_img = model.draw(img)

        # BGRからRGBに変換して表示
        res_img_rgb = cv2.cvtColor(res_img, cv2.COLOR_BGR2RGB)
        st.image(
            res_img_rgb, caption="解析完了（結果）", use_container_width=True
        )
        st.success("骨格検出が完了しました！")

  # 【2】 動画ファイルの場合の処理
  else:
    st.video(uploaded_file)

    if st.button("🚀 骨格検出を開始 (動画解析)"):
      # 一時ファイルとして動画を保存
      tfile = tempfile.NamedTemporaryFile(delete=False, suffix=".mp4")
      tfile.write(uploaded_file.read())

      cap = cv2.VideoCapture(tfile.name)
      st_frame = st.empty()

      total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
      progress_bar = st.progress(0)

      frame_count = 0
      while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
          break

        # フレームごとに姿勢推定
        pts = model.inference(frame)
        out_frame = model.draw(frame)

        # 表示更新
        out_frame_rgb = cv2.cvtColor(out_frame, cv2.COLOR_BGR2RGB)
        st_frame.image(out_frame_rgb, channels="RGB", use_container_width=True)

        frame_count += 1
        if total_frames > 0:
          progress_bar.progress(min(frame_count / total_frames, 1.0))

      cap.release()
      st.success("🎉 動画の全フレーム解析が完了しました！")
