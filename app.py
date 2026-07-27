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

# COCO 17関節の接続関係
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

# IDごとの識別用カラーパレット (BGR)
COLOR_PALETTE = [
    (255, 51, 51),  # 赤/青系
    (51, 255, 51),  # 黄緑
    (51, 51, 255),  # 青
    (255, 255, 51),  # 黄
    (255, 51, 255),  # マゼンタ
    (51, 255, 255),  # シアン
    (255, 153, 51),  # オレンジ
    (153, 51, 255),  # 紫
    (51, 255, 153),  # エメラルド
    (255, 102, 178),  # ピンク
]


def get_id_color(track_id: int):
  """ID番号に応じた固有の色を取得"""
  idx = int(track_id) % len(COLOR_PALETTE)
  return COLOR_PALETTE[idx]


def draw_person_skeleton(frame, kpts, track_id, conf_thresh=0.15):
  """個別の人物骨格とIDラベルを描画"""
  color = get_id_color(track_id)

  # 関節（点）の描画
  for pt in kpts:
    x, y = pt[0], pt[1]
    conf = pt[2] if len(pt) >= 3 else 1.0
    if conf >= conf_thresh and not np.isnan(x) and not np.isnan(y):
      cv2.circle(frame, (int(x), int(y)), 4, color, -1)

  # 骨格（線）の描画
  for p1, p2 in SKELETON_EDGES:
    pt1, pt2 = kpts[p1], kpts[p2]
    c1 = pt1[2] if len(pt1) >= 3 else 1.0
    c2 = pt2[2] if len(pt2) >= 3 else 1.0
    if c1 >= conf_thresh and c2 >= conf_thresh:
      if not (
          np.isnan(pt1[0])
          or np.isnan(pt1[1])
          or np.isnan(pt2[0])
          or np.isnan(pt2[1])
      ):
        cv2.line(
            frame,
            (int(pt1[0]), int(pt1[1])),
            (int(pt2[0]), int(pt2[1])),
            color,
            2,
        )

  # 頭上に ID ラベルを描画
  valid_y = [
      pt[1] for pt in kpts if (len(pt) < 3 or pt[2] >= conf_thresh)
  ]
  valid_x = [
      pt[0] for pt in kpts if (len(pt) < 3 or pt[2] >= conf_thresh)
  ]
  if valid_y and valid_x:
    top_x = int(np.mean(valid_x))
    top_y = int(min(valid_y)) - 15
    cv2.putText(
        frame,
        f"ID: {track_id}",
        (max(10, top_x - 20), max(20, top_y)),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.6,
        color,
        2,
    )

  return frame


# --------------------------------------------------
# アプリのタイトルと説明
# --------------------------------------------------
st.title("🤸‍♀️ チーム全員マルチIDトラッキング骨格解析アプリ")
st.write(
    "全員の骨格を同時に検知し、IDごとに色分けして自動追跡。"
    "グループ全体の連携や交差時の挙動を幅広く分析できます。"
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
    "YOLO入力サイズ (解像度)", options=[320, 640], index=1
)


# --------------------------------------------------
# ViTPoseモデルのロード（YOLO Large + 動画トラッキング有効）
# --------------------------------------------------
@st.cache_resource
def load_vitpose_model(yolo_size_val: int):
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
      is_video=True,  # ByteTrack有効化
      device=device,
  )
  return model


with st.spinner("AIモデルを読み込み中..."):
  model = load_vitpose_model(yolo_size)

try:
  model.yolo.conf = conf_threshold
except AttributeError:
  pass

# --------------------------------------------------
# 動画アップロード & 解析処理
# --------------------------------------------------
st.subheader("📁 メディアのアップロード")
uploaded_file = st.file_uploader(
    "解析したい動画（MP4/MOV）を選択してください", type=["mp4", "mov", "avi"]
)

if uploaded_file is not None:
  st.video(uploaded_file)

  if st.button("🚀 全員マルチID追跡解析を開始"):
    tfile = tempfile.NamedTemporaryFile(delete=False, suffix=".mp4")
    tfile.write(uploaded_file.read())

    cap = cv2.VideoCapture(tfile.name)
    st_frame = st.empty()

    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    progress_bar = st.progress(0)

    # トラッカーの初期化
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
      out_frame = frame.copy()

      frame_count += 1

      # 全員の骨格を描画 (pts は ID をキーとする辞書)
      if len(pts) > 0:
        for track_id, kpts_data in pts.items():
          kpts = (
              np.array(kpts_data["keypoints"])
              if isinstance(kpts_data, dict)
              else np.array(kpts_data)
          )

          if kpts.ndim == 2:
            out_frame = draw_person_skeleton(
                out_frame, kpts, track_id, conf_thresh=conf_threshold
            )

      out_frame_rgb = cv2.cvtColor(out_frame, cv2.COLOR_BGR2RGB)
      st_frame.image(out_frame_rgb, channels="RGB", use_container_width=True)

      if total_frames > 0:
        progress_bar.progress(min(frame_count / total_frames, 1.0))

    cap.release()
    st.success(
        "🎉 動画の解析が完了しました！チーム全員のIDと骨格が可視化されました。"
    )
