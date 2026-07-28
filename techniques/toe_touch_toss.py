# techniques/toe_touch_toss.py

def select_best_frames(trajectory):
    """
    トータッチ・トス：スナップ速度評価仕様
    1. 空中初期 (airborne_start): 離空して浮上中のコマ
    2. 開脚直前 (pre_peak): 脚が閉じている状態の最後（開脚動作の直前起点）
    3. 最高到達点 (peak): 開脚しきったピーク
    4. 反り・着地 (arch_landing): 降下して足を閉じたキャッチ姿勢
    """
    if not trajectory or len(trajectory) < 3:
        return None, None, None, None

    # --- 1. 最高到達点 (Peak) ---
    peak_data = min(trajectory, key=lambda x: x.get('center', [0, 9999])[1])
    peak_idx = peak_data['frame_idx']
    peak_y = peak_data['center'][1]

    max_y = max(d['center'][1] for d in trajectory)
    height_range = max_y - peak_y

    # --- 2. 空中初期 (Airborne Start) ---
    airborne_candidates = [
        d for d in trajectory 
        if d['frame_idx'] < peak_idx and d['center'][1] <= (peak_y + height_range * 0.45)
    ]
    airborne_start = airborne_candidates[0] if airborne_candidates else peak_data
    airborne_idx = airborne_start['frame_idx']

    # --- 3. 開脚直前 (Pre-Peak / 閉じ姿勢の最後) ---
    # airborne_idx 〜 peak_idx の間で「アスペクト比 w/h が小さい（＝足が閉じている）」コマのうち、
    # 最もピーク直前に位置するコマを選出
    pre_peak_candidates = [
        d for d in trajectory 
        if airborne_idx <= d['frame_idx'] < peak_idx
    ]
    
    if pre_peak_candidates:
        def get_aspect(d):
            b = d.get('bbox', [0, 0, 1, 1])
            w, h = b[2] - b[0], b[3] - b[1]
            return w / max(1.0, h)

        # アスペクト比が 0.85 以下の「足閉じコマ」を抽出
        closed_frames = [d for d in pre_peak_candidates if get_aspect(d) <= 0.85]
        
        if closed_frames:
            # 足が閉じている最後のコマ（最もピークに近いコマ）
            pre_peak = max(closed_frames, key=lambda d: d['frame_idx'])
        else:
            # 万が一途中で足を閉じ切っていない場合は、全候補の中で最もアスペクト比が小さいコマ
            pre_peak = min(pre_peak_candidates, key=get_aspect)
    else:
        pre_peak = airborne_start

    # --- 4. 反り・着地 (Arch / Landing) ---
    post_peak_candidates = [d for d in trajectory if d['frame_idx'] > peak_idx]
    if post_peak_candidates:
        def get_aspect(d):
            b = d.get('bbox', [0, 0, 1, 1])
            w, h = b[2] - b[0], b[3] - b[1]
            return w / max(1.0, h)
            
        search_post = post_peak_candidates[2:] if len(post_peak_candidates) >= 3 else post_peak_candidates
        arch_landing = min(search_post, key=get_aspect)
    else:
        arch_landing = peak_data

    return airborne_start, pre_peak, peak_data, arch_landing


def generate_diagnosis(f1, f2, f3, f4, trajectory=None):
    """
    スナップ速度とフォームのAI判定
    """
    improvements = []
    good_points = []

    # 1. 空中初期
    if f1:
        good_points.append(f"✨ **【① 空中初期】(Frame {f1['frame_idx']})**: スムーズに踏み切って浮上を開始できています！")

    # 2. 開脚スナップ速度（②から③にかかったコマ数）
    if f2 and f3:
        frame_diff = f3['frame_idx'] - f2['frame_idx']
        if frame_diff <= 5:
            good_points.append(f"✨ **【② 開脚スナップ】(Frame {f2['frame_idx']} → {f3['frame_idx']})**: 足を閉じた状態からわずか {frame_diff} コマで瞬時に開脚できており、素晴らしい爆発力です！")
        else:
            improvements.append(f"💡 **【② 開脚スナップ】(Frame {f2['frame_idx']} → {f3['frame_idx']})**: 開脚完了までに {frame_diff} コマかかっています。ギリギリまで足を閉じて一気にパッと開くスナップを意識しましょう！")

    # 3. 最高到達点
    def get_aspect(data):
        if not data: return 1.0
        b = data.get('bbox', [0, 0, 1, 1])
        w, h = b[2] - b[0], b[3] - b[1]
        return w / max(1.0, h)

    asp3 = get_aspect(f3)
    if asp3 >= 1.0:
        good_points.append(f"✨ **【③ 最高到達点】(Frame {f3['frame_idx']})**: トップで綺麗な開脚フォームが作れています！")
    else:
        improvements.append(f"💡 **【③ 最高到達点】(Frame {f3['frame_idx']})**: ピークでさらに足を高く引き上げましょう")

    # 4. 反り・着地（締め）
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

