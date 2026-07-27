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

# --------------------------------------------------
# タイトルと説明
# --------------------------------------------------
st.title("🤸‍♀️ フライヤー見た目追跡 (ReID) 骨格解析アプリ")
st.write(
    "地上での密着時にIDが入れ替わっても、ユニフォームの『見た目（色特徴）』でフライヤーを自動識別・ロックオンします。"
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

color_similarity_thresh = st.sidebar.slider(
    "見た目類似度しきい値",
    min_value=0.10,
    max_value=0.80,
    value=0.30,
    step=0.05,
    help="地上で重なった際、衣装の色がこの値以上一致する人をフライヤーとみなします。",
)


# --------------------------------------------------
# 見た目（ReID）用特徴量抽出関数
# --------------------------------------------------
def extract_color_histogram(frame, kpts):
  """キーポイントから人物領域を切り出し、HSVカラーヒストグラム（見た目特徴）を計算"""
  valid_pts = kpts[kpts[:, 2] > 0.1] if kpts.shape[1] >= 3 else kpts
  if len(valid_pts) < 3:
    return None

  x1, y1 = np.min(valid_pts[:, 0]), np.min(valid_pts[:, 1])
  x2, y2 = np.max(valid_pts[:, 0]), np.max(valid_pts[:, 1])

  h, w, _ = frame.shape
  x1, y1 = max(0, int(x1)), max(0, int(y1))
  x2, y2 = min(w, int(x2)), min(h, int(y2))

  if x2 <= x1 or y2 <= y1:
    return None

  crop = frame[y1:y2, x1:x2]
  hsv = cv2.cvtColor(crop, cv2.COLOR_BGR2HSV)

  # H(色相)とS(彩度)のヒストグラムを抽出
  hist = cv2.calcHist([hsv], [0, 1], None, [18, 25], [0, 180, 0, 256])
  cv2.normalize(hist, hist, 0, 1, cv2.NORM_MINMAX)
  return hist


def compare_histograms(hist1, hist2):
  """2つの見た目特徴量の相関度 (0.0 ~ 1.0) を算出"""
  if hist1 is None or hist2 is None:
    return 0.0
  return float(cv2.compareHist(hist1, hist2, cv2.HISTCMP_CORREL))


def draw_skeleton(frame, kpts_17x3):
  """特定された人物の骨格を描画"""
  for x, y, conf in kpts_17x3:
    if conf >= 0.15:
      cv2.circle(frame, (int(x), int(y)), 5, (0, 255, 0), -1)

  for p1_idx, p2_idx in SKELETON_EDGES:
    x1, y1, c1 = kpts_17x3[p1_idx]
    x2, y2, c2 = kpts_17x3[p2_idx]
    if c1 >= 0.15 and c2 >= 0.15:
      cv2.line(
          frame,
          (int(x1), int(y1)),
          (int(x2), int(y2)),
          (0, 255, 255),
          2,
      )
  return frame


# --------------------------------------------------
# モデルのロード (YOLO Large)
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
      is_video=False,
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
# 動画解析処理
# --------------------------------------------------
st.subheader("📁 メディアのアップロード")
uploaded_file = st.file_uploader(
    "解析したい動画（MP4/MOV）を選択してください", type=["mp4", "mov", "avi"]
)

if uploaded_file is not None:
  st.video(uploaded_file)

  if st.button("🚀 見た目ロックオン解析を開始"):
    tfile = tempfile.NamedTemporaryFile(delete=False, suffix=".mp4")
    tfile.write(uploaded_file.read())

    cap = cv2.VideoCapture(tfile.name)
    st_frame = st.empty()

    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    progress_bar = st.progress(0)

    flyer_histogram = None  # フライヤーの登録済み見た目データ
    frame_count = 0

    while cap.isOpened():
      ret, frame = cap.read()
      if not ret:
        break

      pts = model.inference(frame)
      out_frame = frame.copy()

      frame_count += 1

      if len(pts) > 0:
        candidates = []
        for person_id, kpts_data in pts.items():
          kpts = (
              np.array(kpts_data["keypoints"])
              if isinstance(kpts_data, dict)
              else np.array(kpts_data)
          )

          if kpts.ndim == 2:
            hist = extract_color_histogram(frame, kpts)
            # 画面内での平均高さ (Y座標の小ささ = 高さ)
            avg_y = np.mean(kpts[:, 1]) if len(kpts) > 0 else 9999
            candidates.append({
                "kpts": kpts,
                "hist": hist,
                "avg_y": avg_y,
            })

        # --- フライヤーの初回ロックオン（一番高い位置にいる人物を登録） ---
        if flyer_histogram is None and len(candidates) > 0:
          # Y座標が最も上（最小）の人物を初期フライヤーとして登録
          candidates_sorted = sorted(candidates, key=lambda x: x["avg_y"])
          flyer_candidate = candidates_sorted[0]
          if flyer_candidate["hist"] is not None:
            flyer_histogram = flyer_candidate["hist"]
            out_frame = draw_skeleton(out_frame, flyer_candidate["kpts"])
            cv2.putText(
                out_frame,
                "FLYER LOCKED (Registered)",
                (30, 50),
                cv2.FONT_HERSHEY_SIMPLEX,
                1.0,
                (0, 255, 0),
                3,
            )

        # --- 2フレーム目以降：見た目の類似度でフライヤーを特定 ---
        elif flyer_histogram is not None and len(candidates) > 0:
          best_match = None
          best_score = -1.0

          for cand in candidates:
            score = compare_histograms(flyer_histogram, cand["hist"])
            if score > best_score:
              best_score = score
              best_match = cand

          if best_match is not None and best_score >= color_similarity_thresh:
            out_frame = draw_skeleton(out_frame, best_match["kpts"])

            # 見た目特徴量を少しずつ更新（照明変化や体勢変化に対応）
            if best_match["hist"] is not None:
              flyer_histogram = (
                  0.85 * flyer_histogram + 0.15 * best_match["hist"]
              )

            cv2.putText(
                out_frame,
                f"TRACKING FLYER (Match: {best_score * 100:.0f}%)",
                (30, 50),
                cv2.FONT_HERSHEY_SIMPLEX,
                1.0,
                (0, 255, 0),
                2,
            )
          else:
            cv2.putText(
                out_frame,
                "SEARCHING FLYER...",
                (30, 50),
                cv2.FONT_HERSHEY_SIMPLEX,
                1.0,
                (0, 0, 255),
                2,
            )

      out_frame_rgb = cv2.cvtColor(out_frame, cv2.COLOR_BGR2RGB)
      st_frame.image(out_frame_rgb, channels="RGB", use_container_width=True)

      if total_frames > 0:
        progress_bar.progress(min(frame_count / total_frames, 1.0))

    cap.release()
    st.success(
        "🎉 動画の解析が完了しました！フライヤーの見た目を保持して追跡できました。"
    )
