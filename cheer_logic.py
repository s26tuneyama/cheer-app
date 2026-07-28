# cheer_logic.py の analyze_cheer_flyer_descent 関数を以下に差し替え

def analyze_cheer_flyer_descent(video_path, raw_frames, max_jump_distance=350.0, min_size_ratio=0.01, min_peak_conf=0.20):
    candidates = []
    for frame_info in raw_frames:
        f_idx = frame_info['frame_idx']
        f_height = frame_info.get('frame_height', 1080)
        min_box_h = f_height * min_size_ratio

        for det in frame_info['detections']:
            if det['is_edge']: continue
            if det.get('box_height', 0) < min_box_h: continue
            if det['conf'] < min_peak_conf: continue

            c_data = det.copy()
            c_data['frame_idx'] = f_idx
            candidates.append(c_data)

    if not candidates:
        return []

    # ピーク候補（足首の位置が最も高い人物）
    candidates.sort(key=lambda x: x['center'][1])

    cap = cv2.VideoCapture(video_path)
    valid_flyer_candidates = []

    with mp_pose.Pose(static_image_mode=True, model_complexity=1, min_detection_confidence=0.3) as pose:
        for cand in candidates[:20]:
            cap.set(cv2.CAP_PROP_POS_FRAMES, cand['frame_idx'])
            ret, frame = cap.read()
            if not ret or frame is None: continue

            mp_res = extract_mediapipe_angles(frame, cand['bbox'], pose)
            
            if len(mp_res.get('mp_kpts', {})) >= 5:
                cand_data = cand.copy()
                cand_data['split_angle'] = mp_res['split_angle']
                cand_data['arch_angle'] = mp_res['arch_angle']
                cand_data['mp_kpts'] = mp_res['mp_kpts']
                cand_data['full_bbox'] = mp_res['full_bbox']
                cand_data['ankle_y'] = mp_res['ankle_y']
                valid_flyer_candidates.append(cand_data)

        if not valid_flyer_candidates:
            cap.release()
            return []

        # ★ 決定打：足首・つま先（ankle_y）が最も高い位置に達したコマ＝【最高到達点 兼 最大開脚】
        peak_detection = min(valid_flyer_candidates, key=lambda x: x['ankle_y'])

        peak_frame_idx = peak_detection['frame_idx']
        trajectory = [peak_detection]
        prev_center = peak_detection['center']

        descent_frames = [f for f in raw_frames if f['frame_idx'] > peak_frame_idx]

        for frame_info in descent_frames:
            f_idx = frame_info['frame_idx']
            valid_dets = [d for d in frame_info['detections'] if not d['is_edge']]
            if not valid_dets: break

            # ★ フライヤー追跡の決定版ルール：
            # 1. 前回のフライヤー位置から近くて（距離制限内）
            # 2. 複数の候補（ベース等）がいる場合は「一番高い位置（Y座標が最小）」にいる人物をフライヤーと判定！
            
            close_dets = []
            for d in valid_dets:
                dist = np.sqrt((d['center'][0]-prev_center[0])**2 + (d['center'][1]-prev_center[1])**2)
                # 落下中なので、前回の位置より下（または同等）で一定距離内の候補を集める
                if dist <= max_jump_distance:
                    close_dets.append((dist, d))

            if not close_dets:
                break

            # 近い候補の中で、最も頭/足が高い位置（center[1]が最小）にある検出を採用
            best_cand = min([item[1] for item in close_dets], key=lambda x: x['center'][1])

            cap.set(cv2.CAP_PROP_POS_FRAMES, f_idx)
            ret, frame = cap.read()
            if ret and frame is not None:
                mp_res = extract_mediapipe_angles(frame, best_cand['bbox'], pose)
                tracked_flyer = best_cand.copy()
                tracked_flyer['frame_idx'] = f_idx
                tracked_flyer['split_angle'] = mp_res['split_angle']
                tracked_flyer['arch_angle'] = mp_res['arch_angle']
                tracked_flyer['mp_kpts'] = mp_res['mp_kpts']
                tracked_flyer['full_bbox'] = mp_res['full_bbox']
                tracked_flyer['ankle_y'] = mp_res['ankle_y']
                trajectory.append(tracked_flyer)
                prev_center = tracked_flyer['center']

    cap.release()
    return trajectory
