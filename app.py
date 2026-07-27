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

# COCO 17キーポイントの名称リスト
KEYPOINT_NAMES = [
    "Nose",
    "L_Eye",
    "R_Eye",
    "L_Ear",
    "R_Ear",
    "L_Shoulder",
    "R_Shoulder",
    "L_Elbow",
    "R_Elbow",
    "L_Wrist",
    "R_Wrist",
    "L_Hip",
    "R_Hip",
    "L_Knee",
    "R_Knee",
    "L_Ankle",
    "R_Ankle",
]

# --------------------------------------------------
# アプリのタイトルと説明
# --------------------------------------------------
st.title("🔍 骨格推定データ欠損デバッグアプリ")
st.write(
    "1フレームごとに『データ不在（見失い）』か『低確信度（低Score）』かをリアルタイムで計測・ログ出力します。"
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

use_tracking = st.sidebar.checkbox(
    "動画トラッキング (ByteTrack) を有効化", value=False
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
    with st.spinner("🧠 大型AIモデル (YOLOv8-Large) を初回ダウンロード中..."):
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
      st.markdown("### 📊 リアルタイム・デバッグログ")
      st_status = st.empty()
      st_metrics = st.empty()
      st_kpt_table = st.empty()

    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    progress_bar = st.progress(0)

    # 統計用データ格納
    frame_logs = []
    missing_frames = 0
    frame_count = 0

    while cap.isOpened():
      ret, frame = cap.read()
      if not ret:
        break

      pts = model.inference(frame)
      out_frame = model.draw()

      frame_count += 1
      is_missing = len(pts) == 0

      if is_missing:
        missing_frames += 1
        st_status.error(
            f"❌ **Frame {frame_count}**: 人物未検出 (データ空っぽ)"
        )

        # 画面にエラー文字を描画
        cv2.putText(
            out_frame,
            "MISSING (No Detection)",
            (30, 50),
            cv2.FONT_HERSHEY_SIMPLEX,
            1.0,
            (0, 0, 255),
            3,
        )

        frame_logs.append({
            "frame": frame_count,
            "detected": False,
            "avg_conf": 0.0,
            "min_conf": 0.0,
        })
      else:
        st_status.success(
            f"✅ **Frame {frame_count}**: 人物検出完了 ({len(pts)}人)"
        )

        # 先頭の人物（フライヤー想定）の確信度を抽出
        kpts = list(pts.values())[0]  # shape: (17, 3) など

        # 確信度カラムの取得 (3列目があればそれがconfidence)
        if kpts.shape[1] >= 3:
          confs = kpts[:, 2]
        else:
          confs = np.ones(17)  # 確信度情報がない場合のフォールバック

        avg_conf = float(np.mean(confs))
        min_conf = float(np.min(confs))

        # 画面に平均確信度を描画
        cv2.putText(
            out_frame,
            f"Avg Conf: {avg_conf:.2f}",
            (30, 50),
            cv2.FONT_HERSHEY_SIMPLEX,
            1.0,
            (0, 255, 0),
            2,
        )

        # メトリクス表示
        st_metrics.metric(
            label="現在のキーポイント平均確信度",
            value=f"{avg_conf * 100:.1f}%",
            delta=f"最低関節: {min_conf * 100:.1f}%",
        )

        # 各関節の確信度テーブル作成
        df_kpts = pd.DataFrame(
            {"関節名": KEYPOINT_NAMES, "確信度 (%)": (confs * 100).round(1)}
        )
        st_kpt_table.dataframe(df_kpts.style.highlight_between(left=0, right=30, color="#ffcdd2"), height=250)

        frame_logs.append({
            "frame": frame_count,
            "detected": True,
            "avg_conf": avg_conf,
            "min_conf": min_conf,
        })

      # 映像更新
      out_frame_rgb = cv2.cvtColor(out_frame, cv2.COLOR_BGR2RGB)
      st_frame.image(out_frame_rgb, channels="RGB", use_container_width=True)

      if total_frames > 0:
        progress_bar.progress(min(frame_count / total_frames, 1.0))

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
          "YOLOの閾値を下げるか、前後のデータ補間（Interpolation）が必要です。"
      )
    else:
      st.info(
          "💡 **主な原因: 関節確信度（Confidence）の低下**\n"
          "人物自体は見失っていませんが、特定の関節の確信度が低いため描画ルールにより画面上で消えています。"
          "描画閾値の解除、または欠損キーポイントの補間処理でキレイに接続可能です。"
      )

