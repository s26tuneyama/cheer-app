# techniques/toe_touch_toss.py

def select_best_frames(trajectory):
    """
    トータッチ・トス：
    1. 最高到達点 (peak_data)
    2. 最高到達点後のアーチ/キャッチ準備コマ (arch_data)
    を選出して返します。
    """
    if not trajectory:
        return None, None

    # 最も高い位置（ankle_yが最小）をピークとする
    peak_data = min(trajectory, key=lambda x: x.get('ankle_y', 9999))
    peak_f_idx = peak_data['frame_idx']

    # 最高到達点よりあとのフレーム（下降・アーチ局面）
    post_peak_frames = [d for d in trajectory if d['frame_idx'] > peak_f_idx]

    if post_peak_frames:
        # ピークから少し落ちたあたり（3〜10フレーム後）をアーチ局面とする
        arch_candidates = [d for d in post_peak_frames if (peak_f_idx + 3) <= d['frame_idx'] <= (peak_f_idx + 10)]
        arch_data = arch_candidates[0] if arch_candidates else post_peak_frames[0]
    else:
        arch_data = peak_data

    return peak_data, arch_data


def generate_diagnosis(peak_data, descent_data, trajectory=None):
    """
    トータッチ・トス専用 AIフォーム診断
    
    1. 【空中局面の最初】脚を閉じられているか
    2. 【最高到達点直前】脚を素早く開けているか
    3. 【最高到達点】ジャンプ共通（開脚角・つま先・上体・左右差）
    4. 【最高到達点後】「反り（アーチ）」ができているか／脚が閉じられているか
    """
    improvements = []  # 🚨 修正・改善ポイント
    good_points = []   # 🎯 ナイスポイント

    if not peak_data:
        return ["⚠️ 解析データが不足しています。"]

    peak_f_idx = peak_data['frame_idx']

    # 上昇局面（ピーク以前）のフレームを取得
    ascent_frames = [d for d in (trajectory or []) if d['frame_idx'] < peak_f_idx]
    ascent_frames.sort(key=lambda x: x['frame_idx'])

    # -------------------------------------------------------------
    # 1. 【空中局面の最初】脚閉じ判定
    # -------------------------------------------------------------
    if ascent_frames:
        early_frame = ascent_frames[0]  # 離空直後のコマ
        early_split = early_frame.get('split_angle')
        early_closed = early_frame.get('feet_closed')

        if (early_split is not None and early_split <= 50) or early_closed is True:
            good_points.append("✨ **【空中局面の最初】脚の締め**: 離空直後に脚をしっかり閉じられています！")
        else:
            improvements.append("💡 **【空中局面の最初】脚の締め**: 離空直後は脚をしっかり閉じましょう（開きが早すぎます）")

    # -------------------------------------------------------------
    # 2. 【最高到達点直前】素早い開脚動作
    # -------------------------------------------------------------
    if len(ascent_frames) >= 2:
        just_before_frame = ascent_frames[-1]  # ピーク直前のコマ
        jb_split = just_before_frame.get('split_angle')
        pk_split = peak_data.get('split_angle')

        # 直前からピークにかけて大きく開けているか
        if jb_split is not None and pk_split is not None:
            if (pk_split - jb_split) >= 15:  # 直前に一気に開いている
                good_points.append("✨ **【最高到達点直前】タイミング**: ピーク直前に素早く脚を開けています！")
            else:
                improvements.append("💡 **【最高到達点直前】タイミング**: 最高到達点直前で一気に脚を開く意識を持ちましょう")

    # -------------------------------------------------------------
    # 3. 【最高到達点】姿勢評価（ジャンプ共通）
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

    # ③ 上体
    posture_angle = peak_data.get('posture_angle')
    if posture_angle is not None and posture_angle < 130:
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
    # 4. 【最高到達点後】「反り（アーチ）」と「脚閉じ」の評価
    # -------------------------------------------------------------
    if descent_data:
        descent_split = descent_data.get('split_angle')
        feet_closed = descent_data.get('feet_closed')
        posture_angle = descent_data.get('posture_angle')

        # A. 脚の引きつけ・閉じ
        if (descent_split is not None and descent_split <= 50) or feet_closed is True:
            good_points.append("✨ **【最高到達点後】脚閉じ**: 開脚後に素早く脚を閉じられています！")
        else:
            improvements.append("💡 **【最高到達点後】脚閉じ**: 開脚のあとは素早く脚を閉じましょう")

        # B. 体の「反り（アーチ）」判定
        # 上体が起こされたまま（起きっぱなし）だとNG、少し胸を張って反る姿勢ができていればOK
        if posture_angle is not None and posture_angle >= 155:
            good_points.append("✨ **【最高到達点後】反り（アーチ）**: トス後の美しい反り動作（アーチ）ができています！")
        else:
            improvements.append("💡 **【最高到達点後】反り（アーチ）**: 上体が起きっぱなしです。トス後は胸を張って体をしっかり反らせましょう")

    # -------------------------------------------------------------
    # グループ化して返却（改善点優先）
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

