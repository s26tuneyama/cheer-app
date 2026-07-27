import streamlit as st
import cv2
import tempfile
import os
import torch
from easy_ViTPose import VitInference
import urllib.request

# --------------------------------------------------
# ページ設定
# --------------------------------------------------
st.set_page_config(page_title="チア動画姿勢推定 (ViTPose)", layout="wide")
st.title("🤸‍♀️ チアリーディング骨格検出アプリ (ViTPose版)")
st.write("Vision Transformer を使用し、空中トスや重なり（オクルージョン）に強い姿勢推定を行います。")

# --------------------------------------------------
# ViTPoseモデルのロード（キャッシュして高速化）
# --------------------------------------------------
@st.cache_resource
def load_vitpose_model():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    
    # 1. ViTPoseの重みファイルが無い場合は自動でダウンロード
    model_path = 'vitpose-s-coco.pth'
    if not os.path.exists(model_path):
        url = "https://huggingface.co/JunkyByte/easy_ViTPose/resolve/main/vitpose-s-coco.pth"
        urllib.request.urlretrieve(url, model_path)
    
    yolo_path = 'yolov8s.pt'

    # 2. 正しい引数でモデルを初期化
    model = VitInference(
        model_path=model_path,
        yolo_path=yolo_path,
        model_name='s',
        device=device
    )
    return model




# モデル読み込み
with st.spinner("AIモデルを読み込んでいます..."):
    model = load_vitpose_model()

# --------------------------------------------------
# 動画アップロード & 処理UI (YOLO版と同様の構成)
# --------------------------------------------------
uploaded_file = st.file_uploader("解析したい動画を選択してください", type=["mp4", "mov", "avi"])

if uploaded_file is not None:
    # 一時ファイルとして動画を保存
    tfile = tempfile.NamedTemporaryFile(delete=False, suffix='.mp4')
    tfile.write(uploaded_file.read())
    input_path = tfile.name

    st.video(input_path) # アップロード動画のプレビュー

    if st.button("骨格検出を開始する"):
        st.info("動画の解析を開始しました。しばらくお待ちください...")
        
        # 出力先パスの設定
        output_path = tempfile.NamedTemporaryFile(delete=False, suffix='.mp4').name
        
        # 動画読み込み
        cap = cv2.VideoCapture(input_path)
        fps = int(cap.get(cv2.CAP_PROP_FPS))
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

        # 動画書き出し設定
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))

        # 進捗バーの初期化
        progress_bar = st.progress(0)
        frame_count = 0

        # フレーム単位の処理ループ
        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break
            
            # --- ViTPoseでの推論と描画 ---
            # frame (BGR) をモデルに入力
            img_dict = model.inference(frame)
            
            # 骨格が描画された画像を取得
            frame_drawn = model.draw(img_dict)
            
            # 動画ファイルへ書き込み
            out.write(frame_drawn)

            # 進捗更新
            frame_count += 1
            if total_frames > 0:
                progress_bar.progress(min(frame_count / total_frames, 1.0))

        cap.release()
        out.release()

        st.success("解析が完了しました！")
        
        # 完成動画の保存・ダウンロードボタン
        with open(output_path, 'rb') as f:
            st.download_button(
                label="解析済み動画をダウンロード",
                data=f,
                file_name="vitpose_result.mp4",
                mime="video/mp4"
            )

