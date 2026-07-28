# techniques/toe_touch_jump.py

def select_best_frames(trajectory):
    """トータッチ・ジャンプ：最高到達点 ＆ 着地スナップ（縦長＝アスペクト比最小）コマ選出"""
    if not trajectory:
        return None, None

    peak_data = trajectory[0]
    peak_f_idx = peak_data['frame_idx']
    
    high_altitude_window = [
        d for d in trajectory 
        if (peak_f_idx + 2) <= d['frame_idx'] <= (peak_f_idx + 12)
    ]

    if high_altitude_window:
        def get_aspect_ratio(det):
            b = det['bbox']
            w, h = b[2] - b[0], b[3] - b[1]
            return w / h if h > 0 else 999.0

        descent_data = min(high_altitude_window, key=get_aspect_ratio)
    else:
        fallback_candidates = [d for d in trajectory if d['frame_idx'] > peak_f_idx]
        descent_data = fallback_candidates[0] if fallback_candidates else peak_data

    return peak_data, descent_data


def generate_diagnosis(peak_data, descent_data):
    """トータッチ・ジャンプ専用 AIアドバイス（ダメな要素先頭＆見出し明確化）"""
    
    improvements = []  # 要改善点（💡）
    good_points = []   # 良好点（✨）

    # -------------------------------------------------------------
    # 1. 【最高到達点】の評価
    # -------------------------------------------------------------
    
    # ① 開脚角度 (90度基準)
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

    # ③ 上体
    posture_angle = peak_data.get('posture_angle')
    if posture_angle is not None and posture_angle < 130:
        improvements.append("💡 **【最高到達点】上体**: 上体を起こしましょう")
    else:
        good_points.append("✨ **【最高到達点】上体**: 上体がしっかり起こせています！")

    # ④ 左右対称性
    leg_symmetry = peak_data.get('leg_symmetry')
    if leg_symmetry is False:
        improvements.append("💡 **【最高到達点】左右差**: 脚の上がり方の左右差があります")
    else:
        good_points.append("✨ **【最高到達点】左右差**: 脚の上がり方は対称です！")

    # -------------------------------------------------------------
    # 2. 【着地】の評価
    # -------------------------------------------------------------
    
    descent_split = descent_data.get('split_angle')
    feet_closed = descent_data.get('feet_closed')
    
    if (descent_split is not None and descent_split <= 40) or feet_closed is True:
        good_points.append("✨ **【着地】足閉じ**: 着地の足は閉じられています！")
    else:
        improvements.append("💡 **【着地】足閉じ**: 着地の足を閉じましょう")

    # -------------------------------------------------------------
    # グループ化して返却（改善点を一番上へ！）
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

