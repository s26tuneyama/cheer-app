# techniques/toe_touch_jump.py

def select_best_frames(trajectory):
    """
    トータッチ・ジャンプ：2主要コマ選出（シンプル評価）
    1. 最高到達点 (peak_data)
    2. 着地 (landing_data)
    """
    if not trajectory:
        return None, None

    # 最高到達点（ピーク）
    peak_data = min(trajectory, key=lambda x: x.get('ankle_y', 9999))
    peak_idx = peak_data['frame_idx']

    # 着地コマ (ピーク後の降下フレームで足閉じ・接地直前)
    descent_frames = [d for d in trajectory if d['frame_idx'] > peak_idx]
    
    if descent_frames:
        def get_landing_score(det):
            b = det.get('bbox', [0, 0, 1, 1])
            w, h = b[2] - b[0], b[3] - b[1]
            aspect = w / h if h > 0 else 999.0
            return aspect # アスペクト比が最小（縦長・足閉じ状態）
        landing_data = min(descent_frames, key=get_landing_score)
    else:
        landing_data = peak_data

    return peak_data, landing_data


def generate_diagnosis(peak_data, landing_data, trajectory=None):
    """
    トータッチ・ジャンプ専用 AIフォーム診断
    評価対象：①最高到達点、②着地 の2項目のみ
    """
    improvements = []  # 🚨 修正・改善ポイント
    good_points = []   # 🎯 ナイスポイント

    if not peak_data:
        return ["⚠️ 解析データが不足しています。"]

    # -------------------------------------------------------------
    # 1. 【① 最高到達点】の評価
    # -------------------------------------------------------------
    # ① 開脚角度
    split_angle = peak_data.get('split_angle')
    pk_box = peak_data.get('bbox', [0, 0, 1, 1])
    pk_aspect = (pk_box[2] - pk_box[0]) / max(1.0, pk_box[3] - pk_box[1])

    if (split_angle is not None and split_angle >= 90) or pk_aspect >= 1.1:
        good_points.append("✨ **【① 最高到達点】開脚角度**: 開脚角度は良好です！")
    else:
        improvements.append("💡 **【① 最高到達点】開脚角度**: 開脚角度をしっかり上げましょう")

    # ② つま先
    toe_extended = peak_data.get('toe_extended')
    if toe_extended is True:
        good_points.append("✨ **【① 最高到達点】つま先**: つま先が綺麗に伸ばせています！")
    elif toe_extended is False:
        improvements.append("💡 **【① 最高到達点】つま先**: つま先まで意識を向けましょう")

    # ③ 上体姿勢
    posture_angle = peak_data.get('posture_angle')
    if posture_angle is not None:
        if posture_angle < 130:
            improvements.append("💡 **【① 最高到達点】上体姿勢**: 前かがみにならず、上体を起こしましょう")
        else:
            good_points.append("✨ **【① 最高到達点】上体姿勢**: 上体をしっかり起こせています！")

    # ④ 左右対称性
    leg_symmetry = peak_data.get('leg_symmetry')
    if leg_symmetry is False:
        improvements.append("💡 **【① 最高到達点】左右差**: 脚の上がり方に左右差があります")
    elif leg_symmetry is True:
        good_points.append("✨ **【① 最高到達点】左右差**: 左右バランス良く上がっています！")

    # -------------------------------------------------------------
    # 2. 【② 着地】の評価
    # -------------------------------------------------------------
    if landing_data:
        descent_split = landing_data.get('split_angle')
        feet_closed = landing_data.get('feet_closed')
        d_box = landing_data.get('bbox', [0, 0, 1, 1])
        d_aspect = (d_box[2] - d_box[0]) / max(1.0, d_box[3] - d_box[1])

        if (descent_split is not None and descent_split <= 40) or feet_closed is True or d_aspect <= 0.65:
            good_points.append("✨ **【② 着地】足閉じ**: 着地に向けて足を素早く閉じられています！")
        else:
            improvements.append("💡 **【② 着地】足閉じ**: 着地の瞬間に足をしっかり閉じましょう")

    # -------------------------------------------------------------
    # レポート組み立て
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

