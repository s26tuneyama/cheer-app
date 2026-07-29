# main.py
from detector import detect_and_filter_frames
from cheer_logic import analyze_cheer_flyer_descent

def run_pipeline(video_path):
    print("1. [汎用AI] 全画面YOLOスキャン & 端のノイズマーク中...")
    raw_frames = detect_and_filter_frames(video_path, conf_threshold=0.10)
    
    print("2. [チア専門] ピーク検出 & 空間バトンタッチ追跡中...")
    flyer_trajectory = analyze_cheer_flyer_descent(raw_frames)
    
    print(f"\n--- 解析完了: フライヤー追跡フレーム数 {len(flyer_trajectory)} ---")
    for pt in flyer_trajectory:
        status = "⭕ 角度採点OK" if pt['valid_for_scoring'] else "⚠️ 位置追跡のみ"
        print(f"Frame {pt['frame_idx']:03d} | Conf: {pt['conf']:.2f} | {status}")

if __name__ == "__main__":
    run_pipeline("cheer_sample.mp4")
