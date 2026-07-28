import streamlit as st
import tempfile
import os
import pandas as pd

from detector import detect_and_filter_frames
from cheer_logic import analyze_cheer_flyer_descent, render_flyer_capture

st.set_page_config(layout="wide")
st.title("🤸‍♀️ チアリーディング 技・姿勢診断AI（MediaPipe ハイブリッド版）")

# ⚙️ サイドバー
st.sidebar.header("⚙️ 検出パラメータ調整")
conf_thresh = st.sidebar.slider("AI全体感度 (Conf)", 0.05, 0.50, 0.15, 0.01)
min_peak_conf = st.sidebar.slider("ピーク検知の最小確信度", 0.20, 0.60, 0.35, 0.05)
margin_pct = st.sidebar.slider("左右端カット率 (%)", 0, 30, 8, 1) / 100.0
top_margin_pct = st.sidebar.slider("天井ノイズカット率 (%)", 0, 15, 5, 1) / 100.0
min_size_pct = st.sidebar.slider("最小人物サイズ (%)", 0.0, 5.0, 1.0, 0.5) / 100.0

uploaded_file = st.file_uploader("演技動画を選択してください", type=["mp4", "mov"])

if uploaded_file is not None:
    with tempfile.NamedTemporaryFile(delete=False, suffix='.mp4') as tmp_file:
        tmp_file.write(uploaded_file.read())
        video_path = tmp_file.name

    st.video(video_path)

    if st.button("🚀 AI解析を開始（YOLO ＋ MediaPipe精密追跡）"):
        with st.spinner("YOLOv8で高速物体追跡中..."):
            raw_frames = detect_and_filter_frames(
                video_path, 
                conf_threshold=conf_thresh, 
                margin_ratio=margin_pct, 
                top_margin_ratio=top_margin_pct
            )

        with st.spinner("MediaPipeでつま先＆反りの精密角度計算中..."):
            flyer_trajectory = analyze_cheer_flyer_descent(
                video_path,
                raw_frames, 
                min_size_ratio=min_size_pct,
                min_peak_conf=min_peak_conf
            )

        if flyer_trajectory:
            st.success(f"解析完了！ {len(flyer_trajectory)} コマの精密骨格データを抽出しました。")

            # 3大シャッターチャンスの特定
            peak_data = flyer_trajectory[0] # ⭐ 最高到達点
            
            # 🤸 最大開脚（つま先含む角度が最大）
            valid_splits = [p for p in flyer_trajectory if p['split_angle'] is not None]
            max_split_data = max(valid_splits, key=lambda x: x['split_angle']) if valid_splits else peak_data

            # 🏹 最大反り（アーチ角度が最小 ＝ 一番深いくの字/反りになっている瞬間）
            valid_arches = [p for p in flyer_trajectory if p['arch_angle'] is not None]
            max_arch_data = min(valid_arches, key=lambda x: x['arch_angle']) if valid_arches else peak_data

            st.subheader("🖼️ チア判定 3大シャッターチャンス")
            col1, col2, col3 = st.columns(3)

            with col1:
                st.markdown("**⭐ 1. 最高到達点（ピーク）**")
                img1 = render_flyer_capture(video_path, peak_data)
                if img1 is not None: st.image(img1, use_container_width=True)

            with col2:
                st.markdown("**🤸 2. 最大開脚の瞬間（つま先考慮）**")
                img2 = render_flyer_capture(video_path, max_split_data)
                if img2 is not None: st.image(img2, use_container_width=True)

            with col3:
                st.markdown("**🏹 3. 最大反りの瞬間（アーチ）**")
                img3 = render_flyer_capture(video_path, max_arch_data)
                if img3 is not None: st.image(img3, use_container_width=True)

            # データテーブル
            formatted_data = []
            for pt in flyer_trajectory:
                split_ang = f"{pt['split_angle']}°" if pt['split_angle'] is not None else "---"
                arch_ang = f"{pt['arch_angle']}°" if pt['arch_angle'] is not None else "---"

                status = []
                if pt['frame_idx'] == peak_data['frame_idx']: status.append("⭐最高点")
                if pt['frame_idx'] == max_split_data['frame_idx']: status.append("🤸最大開脚")
                if pt['frame_idx'] == max_arch_data['frame_idx']: status.append("🏹最大反り")

                formatted_data.append({
                    "コマ (Frame)": pt['frame_idx'],
                    "高さ (Y座標)": f"{pt['center'][1]:.1f} px",
                    "AI確信度": f"{pt['conf'] * 100:.0f}%",
                    "開脚角度 (つま先)": split_ang,
                    "反り角度 (アーチ)": arch_ang,
                    "ハイライト": " / ".join(status) if status else "---"
                })

            df = pd.DataFrame(formatted_data)
            st.subheader("📊 追跡データ ＆ 精密角度一覧")
            st.dataframe(df, use_container_width=True)

        else:
            st.warning("フライヤーの最高到達点が検知できませんでした。")

        if os.path.exists(video_path):
            os.remove(video_path)

