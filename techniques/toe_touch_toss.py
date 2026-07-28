# techniques/toe_touch_toss.py

def select_best_frames(trajectory):
    """
    トータッチ・トス：相対アスペクト比判定による4主要コマ自動選出仕様
    """
    if not trajectory or len(trajectory) < 3:
        return None, None, None, None

    # --- 1. 最高到達点 (Peak: ③) ---
    peak_data = min(trajectory, key=lambda x: x.get('center', [0, 9999])[1])
    peak_idx = peak_data['frame_idx']
    peak_y = peak_data['center'][1]

    # 全体ジャンプ高の算出
    max_y = max(d['center'][1] for d in trajectory)
    height_range = max_y - peak_y

    # --- 2. 空中初期 (Airborne Start: ①) ---
    airborne_candidates = [
        d for d in trajectory 
        if d['frame_idx'] < peak_idx and d['center'][1] <= (peak_y + height_range * 0.45)
    ]
    airborne_start = airborne_candidates[0] if airborne_candidates else peak_data
    airborne_idx = airborne_start['frame_idx']

    # BBoxアスペクト比 (横幅 w / 縦幅 h) の算出関数
    def get_aspect(d):
        b = d.get('bbox', [0, 0, 1, 1])
        w, h = b[2] - b[0], b[3] - b[1]
        return w / max(1.0, h)

    # --- 3. 開脚直前 (Pre-Peak: ②) ---
    base_aspect = get_aspect(airborne_start)

    pre_peak_candidates = [
        d for d in trajectory 
        if airborne_idx <= d['frame_idx'] < peak_idx
    ]
    
    if pre_peak_candidates:
        closed_frames = [
            d for d in pre_peak_candidates 
            if get_aspect(d) <= (base_aspect * 1.25)
        ]
        
        if closed_frames:
            pre_peak = max(closed_frames, key=lambda d: d['frame_idx'])
        else:
            pre_peak = min(pre_peak_candidates, key=get_aspect)
    else:
        pre_peak = airborne_start

    # --- 4. 反り・着地 (Arch / Landing: ④) ---
    post_peak_candidates = [d for d in trajectory if d['frame_idx'] > peak_idx]
    if post_peak_candidates:
        search_post = post_peak_candidates[2:] if len(post_peak_candidates) >= 3 else post_peak_candidates
        arch_landing = min(search_post, key=get_aspect)
    else:
        arch_landing = peak_data

    return airborne_start, pre_peak, peak_data, arch_landing


def generate_diagnosis(f1, f2, f3, f4, trajectory=None):
    """
    スナップ速度・引き上げスピード・締め速度の【厳格判定】AIレポート出力
    """
    improvements = []
    good_points = []

    def get_aspect(data):
        if not data: return 1.0
        b = data.get('bbox', [0, 0, 1, 1])
        w, h = b[2] - b[0], b[3] - b[1]
        return w / max(1.0, h)

    # 1. ① 空中初期：縦の伸び率（引き上げスピード） - 厳格化 (14.0 px/f 以上)
    if f1 and trajectory:
        prev_frames = [d for d in trajectory if d['frame_idx'] < f1['frame_idx']]
        if prev_frames:
            prev_data = max(prev_frames, key=lambda d: d['frame_idx'])
            dt = f1['frame_idx'] - prev_data['frame_idx']
            dy = prev_data['center'][1] - f1['center'][1]
            upward_speed = dy / max(1, dt)

            if upward_speed >= 14.0:
                good_points.append(f"✨ **【① トップの引き上げ】(Frame {f1['frame_idx']})**: 立ち上がりが非常に鋭く、上空へ爆発的な引き上げができています！（速度: {upward_speed:.1f} px/f）")
            else:
                improvements.append(f"💡 **【① トップの引き上げ】(Frame {f1['frame_idx']})**: 離空直後の引き上げ速度がやや緩やかです。もっと鋭く上へ立ち上がりましょう！（速度: {upward_speed:.1f} px/f / 目安: 14.0+）")
        else:
            good_points.append(f"✨ **【① 空中初期】(Frame {f1['frame_idx']})**: スムーズに離空して浮上を開始できています！")

    # 2. ② 開脚スナップ速度（②から③にかかったコマ数） - 厳格化 (4コマ以内)
    if f2 and f3:
        frame_diff_open = f3['frame_idx'] - f2['frame_idx']
        if frame_diff_open <= 4:
            good_points.append(f"✨ **【② 開脚スナップ】(Frame {f2['frame_idx']} → {f3['frame_idx']})**: 脚を閉じた状態からわずか {frame_diff_open} コマで瞬時に開脚できており、完璧なキレです！")
        else:
            improvements.append(f"💡 **【② 開脚スナップ】(Frame {f2['frame_idx']} → {f3['frame_idx']})**: 開脚完了までに {frame_diff_open} コマかかっています（理想は4コマ以内）。ギリギリまで足を閉じて一気にパッと開きましょう！")

    # 3. ③ 最高到達点
    asp3 = get_aspect(f3)
    if asp3 >= 1.0:
        good_points.append(f"✨ **【③ 最高到達点】(Frame {f3['frame_idx']})**: トップで綺麗な開脚フォームが作れています！")
    else:
        improvements.append(f"💡 **【③ 最高到達点】(Frame {f3['frame_idx']})**: ピークでさらに足を高く引き上げましょう")

    # 4. ④ 反り・締め速度（③から④にかかったコマ数） - 厳格化 (5コマ以内)
    if f3 and f4:
        frame_diff_close = f4['frame_idx'] - f3['frame_idx']
        asp4 = get_aspect(f4)
        
        if frame_diff_close <= 5 and asp4 < asp3:
            good_points.append(f"✨ **【④ 反り・締め速度】(Frame {f3['frame_idx']} → {f4['frame_idx']})**: ピークからわずか {frame_diff_close} コマで素早く足を閉じ、体幹を締められています！")
        else:
            improvements.append(f"💡 **【④ 反り・締め速度】(Frame {f3['frame_idx']} → {f4['frame_idx']})**: ピーク後の締め動作に {frame_diff_close} コマかかっています（理想は5コマ以内）。トップを過ぎたら一気に体を締めましょう！")

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

