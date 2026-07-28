import streamlit as st
import tempfile
import os

# 昨日作った/今回新しく分けたモジュールを呼び出し！
from detector import detect_and_filter_frames
from cheer_logic import analyze_cheer_flyer_descent

st.title("🤸‍♀️ チアリーディング 技・採点診断AI")

# 動画アップローダー
uploaded_file = st.file_uploader("演技動画を選択してください", type=["mp4", "mov"])

if uploaded_file is not None:
    # Streamlitで動画を扱うための一時ファイル作成（昨日戦ったやつです！）
    with tempfile.NamedTemporaryFile(delete=False, suffix='.mp4') as tmp_file:
        tmp_file.write(uploaded_file.read())
        video_path = tmp_file.name

    st.video(video_path) # アップロード動画のプレビュー

    if st.button("🚀 AI解析を開始（conf=0.10 & 下降追跡）"):
        with st.spinner("YOLOで超低閾値スキャン中...（端の観客は自動排除しています）"):
            # 1. 汎用AIで全コマ抽出 ＆ 端の観客ノイズを排除
            raw_frames = detect_and_filter_frames(video_path, conf_threshold=0.10)

        with st.spinner("最高到達点を検知 ＆ 落下軌道を近接追跡中..."):
            # 2. チア専門ロジックでピーク検知 ＆ 落下バトンタッチ
            flyer_trajectory = analyze_cheer_flyer_descent(raw_frames)

        # 一時ファイルの削除
        os.remove(video_path)

        # 3. 診断結果の表示
        if flyer_trajectory:
            st.success(f"解析完了！ フライヤーの下降軌道を {len(flyer_trajectory)} フレーム捕捉しました。")
            
            # 結果をテーブル（表）で綺麗にStreamlit表示
            st.subheader("📊 解析データ一覧")
            st.dataframe(flyer_trajectory)
        else:
            st.warning("フライヤーの最高到達点が検知できませんでした。")
