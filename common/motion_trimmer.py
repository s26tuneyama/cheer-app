import cv2
import numpy as np

def detect_active_motion_range(video_path, threshold_ratio=0.015, padding_sec=2.0, step=3):
    """
    【第1段階：共通粗削りカット】
    OpenCVのフレーム差分を用いて、画面内で大きな動き（技）がある時間帯を爆速検出します。

    :param video_path: 入力動画のパス
    :param threshold_ratio: 画面全体に対する「動きのある面積」の判定閾値 (1.5%程度)
    :param padding_sec: 動きのある区間の前後に残すゆとり（秒数）
    :param step: スキャン時に間引くフレーム数 (3コマに1コマチェックでさらに高速化)
    :return: (start_frame, end_frame, fps, total_frames)
    """
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print(f"Error: 動画ファイルが開けません ({video_path})")
        return None, None, None, None

    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    frame_area = width * height

    prev_gray = None
    motion_frames = []

    frame_idx = 0
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        # 指定ステップごとに処理（高速化）
        if frame_idx % step == 0:
            # グレースケール化 & ぼかし（ノイズ除去）
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            gray = cv2.GaussianBlur(gray, (21, 21), 0)

            if prev_gray is not None:
                # 前後のフレーム差分を計算
                delta = cv2.absdiff(prev_gray, gray)
                thresh = cv2.threshold(delta, 25, 255, cv2.THRESH_BINARY)[1]
                
                # 動いたピクセル数をカウント
                motion_pixel_count = cv2.countNonZero(thresh)
                motion_ratio = motion_pixel_count / float(frame_area)

                # 閾値以上の動きがあれば「アクティブコマ」として記録
                if motion_ratio >= threshold_ratio:
                    motion_frames.append(frame_idx)

            prev_gray = gray

        frame_idx += 1

    cap.release()

    if not motion_frames:
        # 動きが検出できなかった場合は全編を返す
        print("⚠️ 動きが検出されなかったため、全編を解析対象にします。")
        return 0, total_frames - 1, fps, total_frames

    # --- 前後にゆとり（バッファー）を付与 ---
    raw_start = min(motion_frames)
    raw_end = max(motion_frames)

    padding_frames = int(fps * padding_sec)
    
    # 0未満や最大フレーム数を超えないようガード
    start_frame = max(0, raw_start - padding_frames)
    end_frame = min(total_frames - 1, raw_end + padding_frames)

    print(f"🎬 [第1段階完了] 全 {total_frames} フレームから 動きのある区間 (Frame {start_frame} 〜 {end_frame}) を抽出しました！")
    return start_frame, end_frame, fps, total_frames
