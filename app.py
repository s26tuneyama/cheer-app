import pandas as pd  # 先頭に import pandas as pd を追加してください

# -------------------------------------------------------------
# 3. 診断結果の表示部分を以下のように書き換え
# -------------------------------------------------------------
if flyer_trajectory:
    st.success(f"解析完了！ フライヤーの下降軌道を {len(flyer_trajectory)} フレーム捕捉しました。")

    # --- 人間が見やすい表データを作成 ---
    formatted_data = []
    for pt in flyer_trajectory:
        formatted_data.append({
            "コマ (Frame)": pt['frame_idx'],
            "高さ (Y座標)": f"{pt['center'][1]:.1f} px",
            "AIの確信度": f"{pt['conf'] * 100:.0f}%",
            "診断での使い道": "⭕ 採点対象（角度・姿勢）" if pt['valid_for_scoring'] else "⚠️ 軌道追跡のみ（キャッチ・速度）"
        })

    # Pandasの綺麗なテーブルにしてStreamlitで表示
    df = pd.DataFrame(formatted_data)

    st.subheader("📊 追跡データ一覧")
    st.dataframe(df, use_container_width=True) # 画面幅に合わせて綺麗に表示

    # サマリー（まとめ）の表示例
    peak_frame = flyer_trajectory[0]['frame_idx']
    catch_frame = flyer_trajectory[-1]['frame_idx']
    
    st.metric(label="最高到達点（ピーク）", value=f"Frame {peak_frame}")
    st.metric(label="キャッチ完了位置", value=f"Frame {catch_frame}")

else:
    st.warning("フライヤーの最高到達点が検知できませんでした。")

