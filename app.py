# app.py の解析完了後の表示部分を以下のように修正

from cheer_logic import analyze_cheer_flyer_descent, render_flyer_capture  # render_flyer_capture を追加インポート

# （中略 ... ボタンが押されて解析が終わった後）

if flyer_trajectory:
    st.success(f"解析完了！ フライヤーの下降軌道と骨格角度を {len(flyer_trajectory)} コマ抽出しました。")

    # ---------------------------------------------------------
    # 📸 【新機能】画像キャプチャ表示セクション
    # ---------------------------------------------------------
    st.subheader("🖼️ AI判断チェック（主要コマのキャプチャ画像）")
    
    # ピークコマと最大開脚コマの抽出
    peak_data = flyer_trajectory[0]
    
    valid_pts = [p for p in flyer_trajectory if p['valid_for_scoring'] and p['split_angle'] is not None]
    max_split_data = max(valid_pts, key=lambda x: x['split_angle']) if valid_pts else None

    col_img1, col_img2 = st.columns(2)

    with col_img1:
        st.markdown("  **⭐ 最高到達点（ピーク）**")
        peak_img = render_flyer_capture(video_path, peak_data)
        if peak_img is not None:
            st.image(peak_img, use_container_width=True)

    with col_img2:
        st.markdown("  **🤸 最大開脚の瞬間**")
        if max_split_data:
            split_img = render_flyer_capture(video_path, max_split_data)
            if split_img is not None:
                st.image(split_img, use_container_width=True)
        else:
            st.info("開脚計測可能なコマがありませんでした。")

    # ---------------------------------------------------------
    # 📊 データテーブルの表示
    # ---------------------------------------------------------
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

    # 画面下部で一時ファイルを削除
    if os.path.exists(video_path):
        os.remove(video_path)

