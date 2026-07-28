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
    # ピーク高度から全体の45%以上浮上した（＝完全に離空した）最古のコマ
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
    # ①（空中初期）の直立アスペクト比を「脚閉じの基準値」とする
    base_aspect = get_aspect(airborne_start)

    pre_peak_candidates = [
        d for d in trajectory 
        if airborne_idx <= d['frame_idx'] < peak_idx
    ]
    
    if pre_peak_candidates:
        # 基準値の1.25倍以内（＝まだ大きく開脚していない）コマに絞り込み
        closed_frames = [
            d for d in pre_peak_candidates 
            if get_aspect(d) <= (base_aspect * 1.25)
        ]
        
        if closed_frames:
            # まだ足を閉じている「最後のコマ」＝スナップ開脚の直前起点
            pre_peak = max(closed_frames, key=lambda d: d['frame_idx'])
        else:
            pre_peak = min(pre_peak_candidates, key=get_aspect)
    else:
        pre_peak = airborne_start

    # --- 4. 反り・着地 (Arch / Landing: ④) ---
    post_peak_candidates = [d for d in trajectory if d['frame_idx'] > peak_idx]
    if post_peak_candidates:
        # 直後すぎるコマを除外し、降下中で最も体がまとまった（アスペクト比最小）コマ
        search_post = post_peak_candidates[2:] if len(post_peak_candidates) >= 3 else post_peak_candidates
        arch_landing = min(search_post, key=get_aspect)
    else:
        arch_landing = peak_data

    return airborne_start, pre_peak, peak_data, arch_landing


def generate_diagnosis(f1, f2, f3, f4, trajectory=None):
    """
    スナップ速度とフォームのAI判定・レポート出力
    """
    improvements = []
    good_points = []

    def get_aspect(data):
        if not data: return 1.0
        b = data.get('bbox', [0, 0, 1, 1])
        w, h = b[2] - b[0], b[3] - b[1]
        return w / max(1.0, h)

    # 1. 空中初期
    if f1:
        good_points.append(f"✨ **【① 空中初期】(Frame {f1['frame_idx']})**: スムーズに踏み切って浮上を開始できています！")

    # 2. 開脚スナップ速度（②から③にかかったフレーム数）
    if f2 and f3:
        frame_diff = f3['frame_idx'] - f2['frame_idx']
        if frame_diff <= 6:
            good_points.append(f"✨ **【② 開脚スナップ】(Frame {f2['frame_idx']} → {f3['frame_idx']})**: 脚を閉じた状態からわずか {frame_diff} コマで瞬時に開脚できており、素晴らしい爆発力です！")
        else:
            improvements.append(f"💡 **【② 開脚スナップ】(Frame {f2['frame_idx']} → {f3['frame_idx']})**: 開脚完了までに {frame_diff} コマかかっています。ギリギリまで足を閉じて一気にパッと開くスナップを意識しましょう！")

    # 3. 最高到達点
    asp3 = get_aspect(f3)
    if asp3 >= 1.0:
        good_points.append(f"✨ **【③ 最高到達点】(Frame {f3['frame_idx']})**: トップで綺麗な開脚フォームが作れています！")
    else:
        improvements.append(f"💡 **【③ 最高到達点】(Frame {f3['frame_idx']})**: ピークでさらに足を高く引き上げましょう")

    # 4. 反り・着地
    asp4 = get_aspect(f4)
    if asp4 < asp3:
        good_points.append(f"✨ **【④ 反り・着地】(Frame {f4['frame_idx']})**: ピーク後に素早く足を閉じてキャッチ姿勢へ入れています！")
    else:
        improvements.append(f"💡 **【④ 反り・着地】(Frame {f4['frame_idx']})**: ピークを過ぎたら素早く足を閉じましょう")

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

