# techniques/toe_touch_jump.py

def select_best_frames(trajectory):
    """
    トータッチ・ジャンプ：根拠画像（コマ撮り）用の4主要コマを選出
    1. 空中初期 (early_data)
    2. ピーク直前 (just_before_data)
    3. 最高到達点 (peak_data)
    4. 着地局面 (descent_data)
    """
    if not trajectory:
        return None, None, None, None

    peak_data = trajectory[0]
    peak_f_idx = peak_data['frame_idx']

    ascent_frames = [d for d in trajectory if d['frame_idx'] < peak_f_idx]
    ascent_frames.sort(key=lambda x: x['frame_idx'])

    descent_frames = [d for d in trajectory if d['frame_idx'] > peak_f_idx]
    descent_frames.sort(key=lambda x: x['frame_idx'])

    early_data = ascent_frames[0] if ascent_frames else peak_data
    just_before_data = ascent_frames[-1] if len(ascent_frames) >= 2 else early_data

    if descent_frames:
        def get_aspect_ratio(det):
            b = det['bbox']
            w, h = b[2] - b[0], b[3] - b[1]
            return w / h if h > 0 else 999.0

        descent_data = min(descent_frames, key=get_aspect_ratio)
    else:
        descent_data = peak_data

    return early_data, just_before_data, peak_data, descent_data


def generate_diagnosis(early_data, just_before_data, peak_data, descent_data, trajectory=None):
    """
    トータッチ・ジャンプ専用 AIフォーム診断（※上体判定あり）
    """
    improvements = []  # 🚨 修正・改善ポイント
    good_points = []   # 🎯 ナイスポイント

    if not peak_data:
        return ["⚠️ 解析データが不足しています。"]

    # -------------------------------------------------------------
    # 1. 【最高到達点】の評価
    # -------------------------------------------------------------

    # ① 開脚角度
    split_angle = peak_data.get('split_angle')
    if split_angle is not None:
        if split_angle < 90:
            improvements.append("💡 **【最高到達点】開脚角度**: 開脚角度を上げましょう")
        else:
            good_points.append("✨ **【最高到達点】開脚角度**: 開脚角度は良好です！")

    # ② つま先
    toe_extended = peak_data.get('toe_extended')
    if toe_extended is True:
        good_points.append("✨ **【最高到達点】つま先**: つま先が伸びています！")
    else:
        improvements.append("💡 **【最高到達点】つま先**: つま先を伸ばしましょう")

    # ③ 上体姿勢（ソロジャンプ用として維持）
    posture_angle = peak_data.get('posture_angle')
    if posture_angle is not None:
        if posture_angle < 130:
            improvements.append("💡 **【最高到達点】上体**: 上体をしっかり起こしましょう")
        else:
            good_points.append("✨ **【最高到達点】上体**: 上体がしっかり起こせています！")

    # ④ 左右対称性
    leg_symmetry = peak_data.get('leg_symmetry')
    if leg_symmetry is False:
        improvements.append("💡 **【最高到達点】左右差**: 脚の上がり方の左右差があります")
    else:
        good_points.append("✨ **【最高到達点】左右差**: 脚の上がり方は対称です！")

    # -------------------------------------------------------------
    # 2. 【着地】の評価 (YOLOアスペクト比 / feet_closed)
    # -------------------------------------------------------------
    descent_split = descent_data.get('split_angle')
    feet_closed = descent_data.get('feet_closed')
    d_box = descent_data.get('bbox', [0, 0, 1, 1])
    d_aspect = (d_box[2] - d_box[0]) / max(1.0, d_box[3] - d_box[1])

    if (descent_split is not None and descent_split <= 40) or feet_closed is True or d_aspect <= 0.65:
        good_points.append("✨ **【着地】足閉じ**: 着地の足はしっかり閉じられています！")
    else:
        improvements.append("💡 **【着地】足閉じ**: 着地の足をしっかり閉じましょう")

    # -------------------------------------------------------------
    # レポート組み立て（改善点優先表示）
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

