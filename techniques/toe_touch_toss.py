# techniques/toe_touch_toss.py

def select_best_frames(trajectory):
    """
    トータッチ・トス：根拠画像（コマ撮り）用の4主要コマを選出
    1. 空中初期 (early_data)
    2. ピーク直前 (just_before_data)
    3. 最高到達点 (peak_data)
    4. 反り・アーチ/キャッチ (arch_data)
    """
    if not trajectory:
        return None, None, None, None

    # ピーク（最高到達点）
    peak_data = min(trajectory, key=lambda x: x.get('ankle_y', 9999))
    peak_f_idx = peak_data['frame_idx']

    ascent_frames = [d for d in trajectory if d['frame_idx'] < peak_f_idx]
    ascent_frames.sort(key=lambda x: x['frame_idx'])

    post_peak_frames = [d for d in trajectory if d['frame_idx'] > peak_f_idx]
    post_peak_frames.sort(key=lambda x: x['frame_idx'])

    # ① 空中初期（離空直後）
    early_data = ascent_frames[0] if ascent_frames else peak_data

    # ② ピーク直前
    just_before_data = ascent_frames[-1] if len(ascent_frames) >= 2 else early_data

    # ④ 反り・アーチ局面
    if post_peak_frames:
        arch_candidates = [d for d in post_peak_frames if (peak_f_idx + 3) <= d['frame_idx'] <= (peak_f_idx + 10)]
        arch_data = arch_candidates[0] if arch_candidates else post_peak_frames[0]
    else:
        arch_data = peak_data

    return early_data, just_before_data, peak_data, arch_data


def generate_diagnosis(early_data, just_before_data, peak_data, arch_data, trajectory=None):
    """
    トータッチ・トス専用 AIフォーム診断
    """
    improvements = []  # 🚨 修正・改善ポイント
    good_points = []   # 🎯 ナイスポイント

    if not peak_data:
        return ["⚠️ 解析データが不足しています。"]

    # -------------------------------------------------------------
    # 1. 【空中局面の最初】脚閉じ判定 (YOLOアスペクト比 / split_angle)
    # -------------------------------------------------------------
    if early_data:
        e_split = early_data.get('split_angle')
        e_box = early_data.get('bbox', [0, 0, 1, 1])
        e_w, e_h = max(1.0, e_box[2] - e_box[0]), max(1.0, e_box[3] - e_box[1])
        e_aspect = e_w / e_h  # 横/縦比

        if (e_split is not None and e_split <= 50) or e_aspect <= 0.65:
            good_points.append("✨ **【① 空中初期】脚の締め**: 離空直後に脚をしっかり閉じられています！")
        else:
            improvements.append("💡 **【① 空中初期】脚の締め**: 離空直後は脚をしっかり閉じましょう（開き始めが早すぎます）")

    # -------------------------------------------------------------
    # 2. 【最高到達点直前】開脚展開スピード（足りない角度の算出）
    # -------------------------------------------------------------
    TARGET_SPEED = 20.0  # ピーク直前からピークへの目標開脚変化量（度）

    if just_before_data and peak_data:
        jb_split = just_before_data.get('split_angle')
        pk_split = peak_data.get('split_angle')

        if jb_split is not None and pk_split is not None:
            diff = pk_split - jb_split
            if diff >= TARGET_SPEED:
                good_points.append(f"✨ **【② 最高到達点直前】開脚スピード**: ピーク直前に素早く一気に開脚できています！（変化量: +{diff:.1f}°）")
            else:
                shortage = TARGET_SPEED - diff
                improvements.append(f"💡 **【② 最高到達点直前】開脚スピード**: 開脚スピードが **あと {shortage:.1f}°** 足りません（現在: +{diff:.1f}° / 目標: +{TARGET_SPEED:.1f}°）。ピーク直前に一気に開く意識を持ちましょう")

    # -------------------------------------------------------------
    # 3. 【最高到達点】姿勢評価（※上体判定は除外）
    # -------------------------------------------------------------
    # ① 開脚角度
    split_angle = peak_data.get('split_angle')
    if split_angle is not None:
        if split_angle < 90:
            improvements.append("💡 **【③ 最高到達点】開脚角度**: 開脚角度を上げましょう")
        else:
            good_points.append("✨ **【③ 最高到達点】開脚角度**: 開脚角度は良好です！")

    # ② つま先
    toe_extended = peak_data.get('toe_extended')
    if toe_extended is True:
        good_points.append("✨ **【③ 最高到達点】つま先**: つま先が伸びています！")
    else:
        improvements.append("💡 **【③ 最高到達点】つま先**: つま先を伸ばしましょう")

    # ③ 左右対称性
    leg_symmetry = peak_data.get('leg_symmetry')
    if leg_symmetry is False:
        improvements.append("💡 **【③ 最高到達点】左右差**: 脚の上がり方の左右差があります")
    else:
        good_points.append("✨ **【③ 最高到達点】左右差**: 脚の上がり方は対称です！")

    # -------------------------------------------------------------
    # 4. 【最高到達点後】「反り（アーチ）」＆「脚閉じ」 (YOLO縦横サイズ変化で判定)
    # -------------------------------------------------------------
    if peak_data and arch_data:
        # ピーク時のバウンディングボックス（開脚中で最も横長）
        p_box = peak_data.get('bbox', [0, 0, 1, 1])
        p_w = max(1.0, p_box[2] - p_box[0])

        # アーチ/後続コマ時のバウンディングボックス
        a_box = arch_data.get('bbox', [0, 0, 1, 1])
        a_w, a_h = max(1.0, a_box[2] - a_box[0]), max(1.0, a_box[3] - a_box[1])
        a_aspect = a_w / a_h

        # ピークの横幅に対する縮小率
        width_ratio = a_w / p_w

        # 横幅がピーク時の60%以下に縮小、または縦長（アスペクト比0.7以下）になっていれば反り＆脚閉じ成功！
        if width_ratio <= 0.60 or a_aspect <= 0.70:
            good_points.append("✨ **【④ 最高到達点後】反り（アーチ）＆脚閉じ**: 開脚後に足を素早く閉じ、綺麗な反り（アーチ）姿勢ができています！")
        else:
            improvements.append("💡 **【④ 最高到達点後】反り（アーチ）＆脚閉じ**: 体・脚が開いたままになっています。開脚後はすぐに足を閉じて胸を張り、コンパクトに体を反らせましょう")

    # -------------------------------------------------------------
    # レポート組み立て（要改善ポイントを上部に配置）
    # -------------------------------------------------------------
    diagnoses = []
    if improvements:
        diagnoses.append("### 🚨 修正・改善ポイント")
        diagnoses.extend(improvements)
        
    if good_points:
        if improvements:
            diagnoses.append("---")
        diagnoses.append("### 🎯 ナイスポイント（Good!）")
        diagnoses.extend(good_points)

    return diagnoses

