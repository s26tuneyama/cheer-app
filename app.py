import streamlit as st
import tempfile
import os
import pandas as pd

from detector import detect_and_filter_frames
from cheer_logic import analyze_cheer_flyer_descent, render_flyer_capture

st.title("🤸‍♀️ チアリーディング 技・姿勢診断AI")

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

    if st.button("🚀 AI解析を開始（高精度モデルで実行）"):
        with st.spinner("YOLOv8 Mediumで高精度スキャン中...（初回はモデルの自動ダウンロードに数秒かかります）"):
            raw_frames = detect_and_filter_frames(
                video_path, 
                conf_threshold=conf_thresh, 
                margin_ratio=margin_pct, 
                top_margin_ratio=top_margin_pct
            )

        with st.spinner("ピーク検知 ＆ 角度解析中..."):
            flyer_trajectory = analyze_cheer_flyer_descent(
                raw_frames, 
                min_size_ratio=min_size_pct,
                min_peak_conf=min_peak_conf
            )

        if flyer_trajectory:
            st.success(f"解析完了！ {len(flyer_trajectory)} コマ抽出しました。")

            st.subheader("🖼️ AI判断チェック")
            peak_data = flyer_trajectory[0]
            valid_pts = [p for p in flyer_trajectory if p['split_angle'] is not None]
            max_split_data = max(valid_pts, key=lambda x: x['split_angle']) if valid_pts else None

            col_img1, col_img2 = st.columns(2)
            with col_img1:
                st.markdown("**⭐ 最高到達点（ピーク）**")
                peak_img = render_flyer_capture(video_path, peak_data)
                if peak_img is not None:
                    st.image(peak_img, use_container_width=True)

            with col_img2:
                st.markdown("**🤸 最大開脚の瞬間**")
                if max_split_data:
                    split_img = render_flyer_capture(video_path, max_split_data)
                    if split_img is not None:
                        st.image(split_img, use_container_width=True)
                else:
                    st.info("開脚が正確に計測できたコマがありませんでした（脚の関節信頼度が低いため除外されました）。")

            formatted_data = []
            for pt in flyer_trajectory:
                body_ang = f"{pt['body_angle']}°" if pt['body_angle'] is not None else "---"
                split_ang = f"{pt['split_angle']}°" if pt['split_angle'] is not None else "---"

                formatted_data.append({
                    "コマ (Frame)": pt['frame_idx'],
                    "高さ (Y座標)": f"{pt['center'][1]:.1f} px",
                    "AI確信度": f"{pt['conf'] * 100:.0f}%",
                    "体幹角度": body_ang,
                    "開脚角度": split_ang,
                    "判定": "⭕ 計測成功" if (pt['split_angle'] is not None or pt['body_angle'] is not None) else "⚠️ 関節見失い"
                })

            df = pd.DataFrame(formatted_data)
            st.subheader("📊 追跡データ ＆ 骨格角度一覧")
            st.dataframe(df, use_container_width=True)

        else:
            st.warning("フライヤーの最高到達点が検知できませんでした。サイドバーの「ピーク検知の最小確信度」を下げて試してみてください。")

        if os.path.exists(video_path):
            os.remove(video_path)

