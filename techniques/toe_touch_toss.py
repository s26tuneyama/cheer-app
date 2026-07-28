# techniques/toe_touch_toss.py

def select_best_frames(trajectory):
    """トータッチ・トス：最高到達点 ＆ 反り（1点凝縮＝枠面積最小）コマ選出"""
    if not trajectory:
        return None, None

    peak_data = trajectory[0]
    peak_f_idx = peak_data['frame_idx']
    
    # 高空域（ピーク直後 2〜12コマ以内）
    high_altitude_window = [
        d for d in trajectory 
        if (peak_f_idx + 2) <= d['frame_idx'] <= (peak_f_idx + 12)
    ]

    if high_altitude_window:
        # 反り（アーチ）姿勢は正面カメラで「1点に凝縮（枠の面積が最小）」する！
        def get_bbox_area(det):
            b = det['bbox']
            return (b[2] - b[0]) * (b[3] - b[1])  # 横幅 × 縦幅

        descent_data = min(high_altitude_window, key=get_bbox_area)
    else:
        fallback_candidates = [d for d in trajectory if d['frame_idx'] > peak_f_idx]
        descent_data = fallback_candidates[0] if fallback_candidates else peak_data

    return peak_data, descent_data

def generate_diagnosis(peak_data, descent_data):
    """トータッチ・トス専用 AIアドバイス"""
    diagnoses = []
    
    split_angle = peak_data.get('split_angle')
    if split_angle is not None:
        if split_angle >= 150:
            diagnoses.append("✨ **開脚力（トス・ピーク）**: 素晴らしい柔軟性と高空でのキープ力です！")
        elif split_angle >= 120:
            diagnoses.append("👍 **開脚力（トス・ピーク）**: 十分な開脚度です。トスの高さを活かしてさらに大きく開きましょう！")
        else:
            diagnoses.append("💡 **開脚力（トス・ピーク）**: 跳び出しの力を利用して、高空で素早くつま先を引き込みましょう。")

    diagnoses.append("🔄 **空中での反り（アーチ姿勢）**: 高空での反り込み動作（シルエットの凝縮）を検出・解析しました。")
    return diagnoses
