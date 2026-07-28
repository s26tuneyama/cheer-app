# techniques/toe_touch_toss.py

def select_best_frames(trajectory):
    """
    トータッチ・トス：YOLOのBBox情報に特化した4主要コマ選出
    MediaPipeに頼らず、BBoxの位置(Y座標)と形状(アスペクト比 w/h)のみで判定

    1. 空中初期 (airborne_start): 地上（ベース）を排除し、フライヤーが離空して上昇を始めたコマ
    2. ピーク直前 (pre_peak): 空中初期〜最高到達点の間で、開脚が始まり横長になり始めるコマ
    3. 最高到達点 (peak): Y座標が最も小さく（高く）、かつアスペクト比が横長なコマ
    4. 反り・着地 (arch_landing): 最高到達点以降の降下・締めフェーズのコマ
    """
    if not trajectory or len(trajectory) < 3:
        return None, None, None, None

    # --- 1. 最高到達点 (Peak) の選出 ---
    # Y座標（center_y）が最も小さく（画面上で最も高く）位置するコマ
    sorted_by_y = sorted(trajectory, key=lambda x: x.get('center', [0, 9999])[1])
    peak_data = sorted_by_y[0]
    peak_idx = peak_data['frame_idx']
    peak_y = peak_data['center'][1]

    # --- 2. 空中初期 (Airborne Start) の選出 ---
    # ピークより前のフレームの中から、地上（ベース）を除外し、
    # 高度がピークに近く（空中に存在し）、かつ最も古い（上昇を開始した）コマを抽出
    valid_pre_peak = [d for d in trajectory if d['frame_idx'] < peak_idx]
    
    if valid_pre_peak:
        # 地上付近の誤検出（Frame 0等）を除外するため、ピーク高度からの距離閾値を設定
        airborne_candidates = [
            d for d in valid_pre_peak 
            if d['center'][1] < peak_y + 400  # ピークから極端に離れていない（空中にいる）検出結果のみ
        ]
        
        if airborne_candidates:
            airborne_start = airborne_candidates[0]  # 空中に入った最初のコマ
        else:
            airborne_start = valid_pre_peak[0]
    else:
        airborne_start = peak_data

    airborne_idx = airborne_start['frame_idx']

    # --- 3. ピーク直前 (Pre-Peak) の選出 ---
    # 空中初期〜ピークの間で、最も開脚（BBoxの横長化）が進行しているコマ
    mid_frames = [d for d in trajectory if airborne_idx < d['frame_idx'] < peak_idx]
    
    if mid_frames:
        def calc_aspect(d):
            b = d.get('bbox', [0, 0, 1, 1])
            w, h = b[2] - b[0], b[3] - b[1]
            return w / max(1.0, h)
        
        pre_peak = max(mid_frames, key=calc_aspect)
    else:
        pre_peak = airborne_start

    # --- 4. 反り・着地 (Arch / Landing) の選出 ---
    # ピーク以降の降下フェーズ（足を閉じ始め、キャッチ体勢に入るコマ）
    post_peak_frames = [d for d in trajectory if d['frame_idx'] > peak_idx]
    
    if post_peak_frames:
        arch_landing = post_peak_frames[min(2, len(post_peak_frames) - 1)]
    else:
        arch_landing = peak_data

    return airborne_start, pre_peak, peak_data, arch_landing


def generate_diagnosis(f1, f2, f3, f4, trajectory=None):
    """
    YOLOのBBox形状変化（アスペクト比 w/h）を中心としたトス用AIフォーム診断
    """
    improvements = []
    good_points = []

    def get_aspect(data):
        if not data: return 1.0
        b = data.get('bbox', [0, 0, 1, 1])
        w, h = b[2] - b[0], b[3] - b[1]
        return w / max(1.0, h)

    # 1. 踏み切り・持ち上げ（空中初期）
    if f1:
        good_points.append("✨ **【① 空中初期】離空タイミング**: 地上を正しく除外し、離空直後の正確な姿勢を捉えられています！")

    # 2. 開脚スピード（ピーク直前のアスペクト比変化）
    asp1 = get_aspect(f1)
    asp2 = get_aspect(f2)
    asp3 = get_aspect(f3)

    if asp2 > 1.0 or (asp2 - asp1 > 0.3):
        good_points.append("✨ **【② ピーク直前】開脚スピード**: 最高到達点に入る前からスピーディーに脚を開けています！")
    else:
        improvements.append("💡 **【② ピーク直前】開脚スピード**: ピークに達する前にもう少し早めに開脚動作を開始しましょう")

    # 3. 最高到達点でのフォーム（アスペクト比）
    if asp3 >= 1.1:
        good_points.append("✨ **【③ 最高到達点】開脚幅**: 高空でしっかりと横幅のある綺麗な開脚フォームが作れています！")
    else:
        improvements.append("💡 **【③ 最高到達点】開脚幅**: ピーク時にもう少し足を高く引き上げて開脚幅を広げましょう")

    # 4. 反り・キャッチ準備（BBoxの収束）
    asp4 = get_aspect(f4)
    if asp4 < asp3:
        good_points.append("✨ **【④ 反り・着地】収束動作**: ピーク後に迅速に足を閉じ、キャッチ姿勢へ移行できています！")
    else:
        improvements.append("💡 **【④ 反り・着地】締め**: ピークを過ぎたら素早く足を閉じて体幹を締めましょう")

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

