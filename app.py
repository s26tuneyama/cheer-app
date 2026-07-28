import streamlit as st
import tempfile
import os
import pandas as pd

from detector import detect_and_filter_frames
from cheer_logic import analyze_cheer_flyer_descent

st.title("🤸‍♀️ チアリーディング 技・姿勢診断AI")

uploaded_file = st.file_uploader("演技動画を選択してください", type=["mp4", "mov"])

if uploaded_file is not None:
    with tempfile.NamedTemporaryFile(delete=False, suffix='.mp4') as tmp_file:
        tmp_file.write(uploaded_file.read())
        video_path = tmp_file.name

    st.video(video_path)

    if st.button("🚀 AI解析を開始（骨格推定 & 姿勢診断）"):
        with st.spinner("YOLO Poseスキャン中（骨格・関節ポイントを抽出中）..."):
            raw_frames = detect_and_filter_frames(video_path, conf_threshold=0.10)

        with st.spinner("最高到達点検知 ＆ 角度データ計算中..."):
            flyer_trajectory = analyze_cheer_flyer_descent(raw_frames)

        if os.path.exists(video_path):
            os.remove(video_path)

        if flyer_trajectory:
            st.success(f"解析完了！ フライヤーの下降軌道と骨格角度を {len(flyer_trajectory)} コマ抽出しました。")

            formatted_data = []
            for pt in flyer_trajectory:
                # 採点不可コマは角度表示を「---」にする
                body_ang = f"{pt['body_angle']}°" if (pt['valid_for_scoring'] and pt['body_angle']) else "---"
                split_ang = f"{pt['split_angle']}°" if (pt['valid_for_scoring'] and pt['split_angle']) else "---"

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

            # 主要メトリクスの表示
            valid_pts = [p for p in flyer_trajectory if p['valid_for_scoring']]
            if valid_pts:
                max_split = max([p['split_angle'] for p in valid_pts if p['split_angle'] is not None], default=0)
                peak_body = valid_pts[0]['body_angle']
                
                col1, col2 = st.columns(2)
                with col1:
                    st.metric(label="ピーク時の体幹角度", value=f"{peak_body}°" if peak_body else "計測不能")
                with col2:
                    st.metric(label="最大開脚角度", value=f"{max_split}°" if max_split else "計測不能")

        else:
            st.warning("フライヤーの最高到達点が検知できませんでした。")

