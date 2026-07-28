import streamlit as st
import tempfile
import os
import pandas as pd

# 別ファイルにした自作モジュールを読み込み
from detector import detect_and_filter_frames
from cheer_logic import analyze_cheer_flyer_descent

st.title("🤸‍♀️ チアリーディング 技・採点診断AI")

# 1. 動画ファイルのアップロード
uploaded_file = st.file_uploader("演技動画を選択してください", type=["mp4", "mov"])

if uploaded_file is not None:
    # Streamlitで動画を扱うための一時ファイル作成
    with tempfile.NamedTemporaryFile(delete=False, suffix='.mp4') as tmp_file:
        tmp_file.write(uploaded_file.read())
        video_path = tmp_file.name

    st.video(video_path) # 動画のプレビュー

    # 2. ボタンを押した時だけ解析を実行
    if st.button("🚀 AI解析を開始（conf=0.10 & 下降追跡）"):
        with st.spinner("YOLOで超低閾値スキャン中...（端の観客は自動排除しています）"):
            # 汎用AIで全コマ抽出 ＆ 端の観客ノイズを排除
            raw_frames = detect_and_filter_frames(video_path, conf_threshold=0.10)

        with st.spinner("最高到達点を検知 ＆ 落下軌道を近接追跡中..."):
            # チア専門ロジックでピーク検知 ＆ 落下バトンタッチ
            flyer_trajectory = analyze_cheer_flyer_descent(raw_frames)

        # 一時ファイルの削除
        if os.path.exists(video_path):
            os.remove(video_path)

        # 3. 診断結果の表示（必ずボタンの処理の内側に書くことでNameErrorを回避！）
        if flyer_trajectory:
            st.success(f"解析完了！ フライヤーの下降軌道を {len(flyer_trajectory)} フレーム捕捉しました。")

            # 人間が見やすい日本語の表データを作成
            formatted_data = []
            for pt in flyer_trajectory:
                formatted_data.append({
                    "コマ (Frame)": pt['frame_idx'],
                    "高さ (Y座標)": f"{pt['center'][1]:.1f} px",
                    "AIの確信度": f"{pt['conf'] * 100:.0f}%",
                    "診断での使い道": "⭕ 採点対象（角度・姿勢）" if pt['valid_for_scoring'] else "⚠️ 軌道追跡のみ（キャッチ・速度）"
                })

            df = pd.DataFrame(formatted_data)
            
            st.subheader("📊 追跡データ一覧")
            st.dataframe(df, use_container_width=True)

            # サマリー（まとめ数値）の表示
            peak_frame = flyer_trajectory[0]['frame_idx']
            catch_frame = flyer_trajectory[-1]['frame_idx']
            
            col1, col2 = st.columns(2)
            with col1:
                st.metric(label="最高到達点（ピーク）", value=f"Frame {peak_frame}")
            with col2:
                st.metric(label="キャッチ完了位置", value=f"Frame {catch_frame}")
                
        else:
            st.warning("フライヤーの最高到達点が検知できませんでした。")

