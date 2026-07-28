# techniques/toe_touch_jump.py

def select_best_frames(trajectory):
    """トータッチ・ジャンプ：最高到達点 ＆ 着地スナップ（縦長＝アスペクト比最小）コマ選出"""
    if not trajectory:
        return None, None

    peak_data = trajectory[0]
    peak_f_idx = peak_data['frame_idx']
    
    # ピーク直後（2〜12コマ以内）の高空〜着地移行域
    high_altitude_window = [
        d for d in trajectory 
        if (peak_f_idx + 2) <= d['frame_idx'] <= (peak_f_idx + 12)
    ]

    if high_altitude_window:
        # 着地に向けて脚をピタッと閉じると、縦長（アスペクト比 w/h が最小）になる
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
    """トータッチ・ジャンプ専用 AIアドバイス（ご指定の判定基準）"""
    diagnoses = []
    
    # -------------------------------------------------------------
    # 1. 最高到達点の姿勢について
    # -------------------------------------------------------------
    
    # ① 開脚角度の判定
    split_angle = peak_data.get('split_angle')
    if split_angle is not None:
        if split_angle >= 90:
            diagnoses.append("✨ **開脚角度**: 開脚角度は良好です！")
        else:
            diagnoses.append("💡 **開脚角度**: 開脚角度を上げましょう")
    else:
        diagnoses.append("⚠️ **開脚角度**: 骨格が検出できませんでした。")

    # ② つま先の伸び判定 (toe_extended: 膝-足首-つま先の角度が直線に近いか)
    toe_extended = peak_data.get('toe_extended')
    if toe_extended is True:
        diagnoses.append("✨ **つま先**: つま先が伸びています！")
    elif toe_extended is False:
        diagnoses.append("💡 **つま先**: つま先を伸ばしましょう")
    else:
        diagnoses.append("💡 **つま先**: つま先をしっかり伸ばす意識を持ちましょう。")

    # ③ 上体の倒れ判定 (posture_angle: 130度未満だと倒れすぎ)
    posture_angle = peak_data.get('posture_angle')
    if posture_angle is not None:
        if posture_angle < 130:
            diagnoses.append("💡 **上体の姿勢**: 上体を起こしましょう")
        else:
            diagnoses.append("✨ **上体の姿勢**: 上体がしっかり起こせています！")
    else:
        diagnoses.append("✨ **上体の姿勢**: 上体がしっかり起こせています！")

    # -------------------------------------------------------------
    # 2. 着地（スナップコマ）について
    # -------------------------------------------------------------
    
    # ④ かかと（着地の足）が閉じているか判定
    # descent_data での開脚角度が小さい (例: 40度以下) か、足の横幅が十分に狭い場合
    descent_split = descent_data.get('split_angle')
    feet_closed = descent_data.get('feet_closed')
    
    # 開脚角度が小さい、または足間隔が閉じているか
    if (descent_split is not None and descent_split <= 40) or feet_closed is True:
        diagnoses.append("✨ **着地**: 着地の足は閉じられています！")
    else:
        diagnoses.append("💡 **着地**: 着地の足を閉じましょう")

    return diagnoses

