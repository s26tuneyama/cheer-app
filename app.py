import streamlit as st
import tempfile
import os
import pandas as pd

from detector import detect_and_filter_frames
from cheer_logic import analyze_cheer_flyer_descent, render_flyer_capture

st.title("🤸‍♀️ チアリーディング 技・姿勢診断AI")

# 1. 動画ファイルのアップロード
uploaded_file = st.file_uploader("演技動画を選択してください", type=["mp4", "mov"])

if uploaded_file is not None:
    # 一時ファイルの作成
    with tempfile.NamedTemporaryFile(delete=False, suffix='.mp4') as tmp_file:
        tmp_file.write(uploaded_file.read())
        video_path = tmp_file.name

    st.video(video_path)

    # 2. ボタンが押された時だけ解析を実行
    if st.button("🚀 AI解析を開始（骨格推定 & 姿勢診断）"):
        with st.spinner("YOLO Poseスキャン中（骨格・関節ポイントを抽出中）..."):
            raw_frames = detect_and_filter_frames(video_path, conf_threshold=0.10)

        with st.spinner("最高到達点検知 ＆ 角度データ計算中..."):
            flyer_trajectory = analyze_cheer_flyer_descent(raw_frames)

        # 3. 診断結果の表示（必ずボタンが押されたブロックの中で実行する）
        if flyer_trajectory:
            st.success(f"解析完了！ フライヤーの下降軌道と骨格角度を {len(flyer_trajectory)} コマ抽出しました。")

            # 🖼️ 主要コマの画像キャプチャ表示
            st.subheader("🖼️ AI判断チェック（主要コマのキャプチャ画像）")
            
            peak_data = flyer_trajectory[0]
            valid_pts = [p for p in flyer_trajectory if p['valid_for_scoring'] and p['split_angle'] is not None]
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
                    st.info("開脚計測可能なコマがありませんでした。")

            # 📊 データテーブルの表示
            formatted_data = []
            for pt in flyer_trajectory:
                body_ang = f"{pt['body_angle']}°" if (pt['valid_for_scoring'] and pt['body_angle'] is not None) else "---"
                split_ang = f"{pt['split_angle']}°" if (pt['valid_for_scoring'] and pt['split_angle'] is not None) else "---"

                formatted_data.append({
                    "コマ (Frame)": pt['frame_idx'],
                    "高さ (Y座標)": f"{pt['center'][1]:.1f} px",
                    "AI確信度": f"{pt['conf'] * 100:.0f}%",
                    "体幹角度 (肩-腰-膝)": body_ang,
                    "開脚角度 (足-腰-足)": split_ang,
                    "判定": "⭕ 姿勢計測可" if pt['valid_for_scoring'] else "⚠️ 追跡のみ"
                })

            df = pd.DataFrame(formatted_data)
            st.subheader("📊 追跡データ ＆ 骨格角度一覧")
            st.dataframe(df, use_container_width=True)

        else:
            st.warning("フライヤーの最高到達点が検知できませんでした。")

        # 一時ファイルの削除
        if os.path.exists(video_path):
            os.remove(video_path)

