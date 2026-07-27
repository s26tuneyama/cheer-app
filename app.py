import gc
import os
import tempfile
import urllib.request
import cv2
from huggingface_hub import hf_hub_download
import numpy as np
import pandas as pd
import streamlit as st
import torch
from easy_ViTPose import VitInference

# COCO 17関節の接続関係 (骨格ラインを描く対)
SKELETON_EDGES = [
    (0, 1),
    (0, 2),
    (1, 3),
    (2, 4),  # 顔
    (5, 6),
    (5, 7),
    (7, 9),
    (6, 8),
    (8, 10),  # 腕
    (5, 11),
    (6, 12),  # 胴体
    (11, 12),
    (11, 13),
    (13, 15),
    (12, 14),
    (14, 16),  # 脚
]

# --------------------------------------------------
# アプリのタイトル
# --------------------------------------------------
st.title("🤸‍♀️ 骨格自動補間 & 解析アプリ")
st.write(
    "高速トスでAIが骨格を見失っても、前後のフレームから『途切れない骨格ライン』を全自動で推測・復元します。"
)

# --------------------------------------------------
# サイドバー設定
# --------------------------------------------------
st.sidebar.title("⚙️ 設定")
conf_threshold = st.sidebar.slider(
    "関節採用の最小確信度", 0.05, 0.50, 0.15, 0.05
)


# --------------------------------------------------
# モデルのロード (軽量かつ高精度な Medium 版)
# --------------------------------------------------
@st.cache_resource
def load_vitpose_model():
  device = "cuda" if torch.cuda.is_available() else "cpu"
  model_file = hf_hub_download(
      repo_id="JunkyByte/easy_ViTPose", filename="torch/coco/vitpose-s-coco.pth"
  )
  yolo_file = "yolov8m.pt"
  if not os.path.exists(yolo_file):
    url = "https://github.com/ultralytics/assets/releases/download/v8.2.0/yolov8m.pt"
    urllib.request.urlretrieve(url, yolo_file)

  model = VitInference(
      model=model_file,
      yolo=yolo_file,
      model_name="s",
      dataset="coco",
      yolo_size=640,
      is_video=False,
      device=device,
  )
  return model


with st.spinner("AIモデル準備中..."):
  model = load_vitpose_model()


# --------------------------------------------------
# 補間後の骨格を綺麗に動画へ上書き描画する関数
# --------------------------------------------------
def draw_interpolated_skeleton(frame, kpts_17x2):
  """補間された座標 (17, 2) を動画フレームに描画"""
  # 関節点（円）の描画
  for i, (x, y) in enumerate(kpts_17x2):
    if not np.isnan(x) and not np.isnan(y):
      cv2.circle(frame, (int(x), int(y)), 5, (0, 255, 0), -1)

  # 骨格線（ライン）の描画
  for p1_idx, p2_idx in SKELETON_EDGES:
    x1, y1 = kpts_17x2[p1_idx]
    x2, y2 = kpts_17x2[p2_idx]

    if (
        not np.isnan(x1)
        and not np.isnan(y1)
        and not np.isnan(x2)
        and not np.isnan(y2)
    ):
      cv2.line(
          frame,
          (int(x1), int(y1)),
          (int(x2), int(y2)),
          (0, 255, 255),
          2,
      )

  return frame


# --------------------------------------------------
# 動画の処理メインルーチン
# --------------------------------------------------
uploaded_file = st.file_uploader(
    "解析したい動画を選択してください", type=["mp4", "mov", "avi"]
)

if uploaded_file is not None:
  st.video(uploaded_file)

  if st.button("🚀 欠損補間 & 骨格解析を実行"):
    tfile = tempfile.NamedTemporaryFile(delete=False, suffix=".mp4")
    tfile.write(uploaded_file.read())

    # --- Pass 1: 全フレームの骨格座標を抽出 ---
    cap = cv2.VideoCapture(tfile.name)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    st.info("ステップ 1/2: 動画から骨格座標データを抽出中...")
    progress_bar = st.progress(0)

    raw_data = []  # 各フレームの座標保持用リスト
    frame_list = []

    frame_idx = 0
    while cap.isOpened():
      ret, frame = cap.read()
      if not ret:
        break

      frame_list.append(frame)
      pts = model.inference(frame)

      # 17関節分 (x, y) の初期値を NaN（未検出）でセット
      frame_kpts = np.full((17, 2), np.nan)

      if len(pts) > 0:
        # 先頭の人物データ（フライヤー）を取得
        kpts_data = list(pts.values())[0]
        if isinstance(kpts_data, dict) and "keypoints" in kpts_data:
          kpts = np.array(kpts_data["keypoints"])
        else:
          kpts = np.array(kpts_data)

        if kpts.ndim == 2:
          for j in range(min(17, len(kpts))):
            conf = kpts[j, 2] if kpts.shape[1] >= 3 else 1.0
            if conf >= conf_threshold:
              frame_kpts[j] = [kpts[j, 0], kpts[j, 1]]

      raw_data.append(frame_kpts.flatten())  # 17x2 = 34次元

      frame_idx += 1
      if total_frames > 0:
        progress_bar.progress(min(frame_idx / total_frames, 0.5))

    cap.release()

    # --- ステップ 2: 前後フレームからの補間（Interpolation） ---
    st.info("ステップ 2/2: 見失ったフレームの骨格を全自動補間中...")

    # 34列（17関節×XY座標）のDataFrameに変換
    df_coords = pd.DataFrame(raw_data)

    # ★核心部分: 線形補間（前後データから欠損コマを自動計算）
    df_interpolated = df_coords.interpolate(
        method="linear", limit_direction="both"
    )

    # --- ステップ 3: 補正された骨格で動画をレンダリング表示 ---
    st_frame = st.empty()

    for idx, frame in enumerate(frame_list):
      kpts_smoothed = df_interpolated.iloc[idx].values.reshape(17, 2)

      # 補正済みのラインを描画
      out_frame = draw_interpolated_skeleton(frame.copy(), kpts_smoothed)

      out_frame_rgb = cv2.cvtColor(out_frame, cv2.COLOR_BGR2RGB)
      st_frame.image(out_frame_rgb, channels="RGB", use_container_width=True)

      if total_frames > 0:
        progress_bar.progress(min(0.5 + (idx / total_frames) * 0.5, 1.0))

    st.success(
        "🎉 骨格の欠損補間と追従描画が完了しました！連続した綺麗な骨格データが保持されています。"
    )

    del frame_list, raw_data, df_coords, df_interpolated
    gc.collect()

