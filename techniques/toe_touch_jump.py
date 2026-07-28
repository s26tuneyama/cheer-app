# techniques/toe_touch_jump.py

def select_best_frames(trajectory):
    """トータッチ・ジャンプ：最高到達点 ＆ 脚閉じ（アスペクト比最小）コマ選出"""
    if not trajectory:
        return None, None

    peak_data = trajectory[0]
    peak_f_idx = peak_data['frame_idx']
    
    high_altitude_window = [
        d for d in trajectory 
        if (peak_f_idx + 2) <= d['frame_idx'] <= (peak_f_idx + 12)
    ]

    if high_altitude_window:
        # 直立・脚閉じは「一番細長くなる（アスペクト比 w/h が最小）」！
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
    """トータッチ・ジャンプ専用 AIアドバイス"""
    diagnoses = []
    
    split_angle = peak_data.get('split_angle')
    if split_angle is not None:
        if split_angle >= 150:
            diagnoses.append("✨ **ジャンプ開脚力**: 高いジャンプから美しく180度近く開けています！")
        else:
            diagnoses.append("💡 **ジャンプ開脚力**: 床の蹴り出しを強め、股関節の引き込みを意識しましょう。")

    descent_split = descent_data.get('split_angle')
    if descent_split is not None and descent_split <= 35:
        diagnoses.append("⚡ **着地スナップ**: 着地に向けて素早く足を閉じられています！")
    else:
        diagnoses.append("👍 **着地スナップ**: ピーク後にスピーディーに脚を閉じて着地に備えましょう。")

    return diagnoses
