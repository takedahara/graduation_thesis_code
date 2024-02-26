import numpy as np
import cv2
import os

# フレーム差分を計算する関数
def frame_diff(frame1, frame2):
    frame1_gray = cv2.cvtColor(frame1, cv2.COLOR_BGR2GRAY)
    frame2_gray = cv2.cvtColor(frame2, cv2.COLOR_BGR2GRAY)
    diff = cv2.absdiff(frame1_gray, frame2_gray)
    return diff

# ビデオファイルからフレーム差分の要素の和を計算する関数
def calculate_frame_diffs(filepass, step=100):
    cap = cv2.VideoCapture(filepass)
    frame_list = []
    count = 0
    ret, frame1 = cap.read()
    while True:
        ret, frame2 = cap.read()
        if not ret:
            break
        if count % step == 0:
            diff = frame_diff(frame1, frame2)
            diff_sum = np.sum(diff)  # フレームの要素の和を計算
            frame_list.append(diff_sum)
        count += 1
        frame1 = frame2
    cap.release()
    return frame_list

# ラベルを保存するための関数
def label_and_save(video_file_path, frame_rate, steps, seconds):
    frame_diffs = calculate_frame_diffs(video_file_path, step=steps)
    frames_per_interval = int(frame_rate * seconds / steps)
    labels = []

    for i in range(0, len(frame_diffs), frames_per_interval):
        current_interval = frame_diffs[i:i + frames_per_interval]
        if len(current_interval) == 0:
            continue

        threshold = np.mean(current_interval)
        label = int(input(f"Enter label for {video_file_path}, Interval {i // frames_per_interval} (Threshold: {threshold}): "))

        for j in range(i, min(i + frames_per_interval, len(frame_diffs))):
            if frame_diffs[j] > threshold:
                labels.append(label)
            else:
                labels.append(0)

    file_name = os.path.basename(video_file_path)  # ファイルパスからファイル名の部分を取得
    step_str = f"step{steps}"  # step の値を文字列に変換
    label_file_name = file_name.replace(".mp4", f"_{step_str}_labels.npy")  # ファイル名を修正してラベルファイルの名前を生成

    # ラベルファイルを 'label' ディレクトリ内に保存
    label_dir = 'label'
    os.makedirs(label_dir, exist_ok=True)  # 'label' ディレクトリが存在しない場合、作成
    label_file_path = os.path.join(label_dir, label_file_name)  # 'label' ディレクトリ内にパスを生成
    np.save(label_file_path, np.array(labels).reshape(1, -1))

if __name__ == "__main__":
    video_file_paths = [
        "/LARGE0/gr10034/b31927/movie/dev121/09/Camera01_20220909002017.mp4",
        "/LARGE0/gr10034/b31927/movie/dev121/09/Camera01_20220909003017.mp4",
        "/LARGE0/gr10034/b31927/movie/dev121/09/Camera01_20220909005017.mp4"
    ]
    frame_rate = 25
    steps = 10
    seconds = 60

    for video_file_path in video_file_paths:
        label_and_save(video_file_path, frame_rate, steps, seconds)
