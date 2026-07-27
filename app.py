import inspect
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

  # 1. Hugging Face からモデルファイルを確実に取得
  model_path = hf_hub_download(
      repo_id="JunkyByte/easy_ViTPose", filename="torch/coco/vitpose-s-coco.pth"
  )
  yolo_path = hf_hub_download(
      repo_id="JunkyByte/easy_ViTPose", filename="yolov8/yolov8s.pt"
  )

  # 2. 初期化を試行し、失敗した場合は画面上に生の理由を表示
  try:
    model = VitInference(
        model_path=model_path,
        yolo_path=yolo_path,
        model_name="s",
        yolo_size=320,
        device=device,
    )
    return model
  except Exception as e:
    # Streamlit Cloudの伏せ字を回避し、画面上に直接原因を出力
    st.error(f"❌ モデル読み込みエラーの詳細: {e}")
    st.info(
        "💡 easy_ViTPose が受け取れる正しい引数一覧:"
        f" {inspect.signature(VitInference.__init__)}"
    )
    raise e


# モデルの呼び出し
model = load_vitpose_model()
