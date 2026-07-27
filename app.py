from huggingface_hub import hf_hub_download
import streamlit as st
import torch
from easy_ViTPose import VitInference

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

  # 1. Hugging Face からモデルファイルをダウンロード
  model_file = hf_hub_download(
      repo_id="JunkyByte/easy_ViTPose", filename="torch/coco/vitpose-s-coco.pth"
  )
  yolo_file = hf_hub_download(
      repo_id="JunkyByte/easy_ViTPose", filename="yolov8/yolov8s.pt"
  )

  # 2. 正解の引数名 (model / yolo) で初期化
  model = VitInference(
      model=model_file,  # ViTPoseモデルファイル
      yolo=yolo_file,  # YOLOモデルファイル
      model_name="s",
      dataset="coco",
      yolo_size=320,
      device=device,
  )
  return model


# モデルの呼び出し
model = load_vitpose_model()
