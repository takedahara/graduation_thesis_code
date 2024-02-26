import numpy as np
import os
import cv2
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
from sklearn.model_selection import train_test_split
import time

# フレーム差分を計算する関数
def frame_diff(frame1, frame2):
    frame1_gray = cv2.cvtColor(frame1, cv2.COLOR_BGR2GRAY)
    frame2_gray = cv2.cvtColor(frame2, cv2.COLOR_BGR2GRAY)
    diff = cv2.absdiff(frame1_gray, frame2_gray)
    return diff

# ビデオファイルからフレーム差分の要素の和を計算する関数
def calculate_frame_diffs(filepass, step=10):
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

# RNN モデルの定義
class RNNClassifier(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers, num_classes):
        super(RNNClassifier, self).__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.rnn = nn.RNN(input_size, hidden_size, num_layers, batch_first=True)
        self.fc = nn.Linear(hidden_size, num_classes)

    def forward(self, x):
        h0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size).to(x.device)
        out, _ = self.rnn(x, h0)
        out = self.fc(out[:, -1, :])
        
if __name__ == "__main__":
    input_size = 1
    hidden_size = 64
    num_layers = 2
    num_classes = 2
    sequence_length = 100
    steps = 10  # フレーム差分を計算する間隔
    label_dir = 'label_2'  # ラベルが保存されているディレクトリ

    # ビデオファイルパスのリスト
    video_file_paths = [
"/LARGE0/gr10034/b31927/movie/dev121/09/Camera01_20220909002017.mp4",
        "/LARGE0/gr10034/b31927/movie/dev121/09/Camera01_20220909003017.mp4",
        "/LARGE0/gr10034/b31927/movie/dev121/09/Camera01_20220909005017.mp4",
    "/LARGE0/gr10034/b31927/movie/dev121/09/Camera01_20220909010017.mp4",
    "/LARGE0/gr10034/b31927/movie/dev121/09/Camera01_20220909011017.mp4",
    "/LARGE0/gr10034/b31927/movie/dev121/09/Camera01_20220909012018.mp4",
    "/LARGE0/gr10034/b31927/movie/dev121/09/Camera01_20220909015018.mp4",
    "/LARGE0/gr10034/b31927/movie/dev121/09/Camera01_20220909021018.mp4",
        "/LARGE0/gr10034/b31927/movie/dev121/09/Camera01_20220909022018.mp4" 
    ]
    # 各ビデオのフレーム差分とラベルをまとめる
    all_diffs = []
    all_labels = []
    for video_file_path in video_file_paths:
        # ビデオからフレーム差分の和を計算
        summed_diffs = calculate_frame_diffs(video_file_path, step=steps)
        all_diffs.extend(summed_diffs)
        # ラベルをロード （実際のロードロジックに置き換える）
        file_name = os.path.basename(video_file_path)
        step_str = f"step{steps}"
        label_file_name = file_name.replace(".mp4", f"_{step_str}_labels.npy")
        label_file_path = os.path.join(label_dir, label_file_name)
        labels = np.load(label_file_path)  # このビデオのラベル
        if labels.ndim > 1:
            labels = labels.reshape(-1)
        all_labels.extend(labels)
    # 全ビデオのフレーム差分とラベルをNumPy 配列に変換
    all_diffs = np.array(all_diffs)
    all_labels = np.array(all_labels)
    # 結合されたフレーム差分からシーケンスを生成し、対応するラベルを割り当てる
    sequences, sequence_labels = [], []
    for i in range(0, len(all_diffs) - sequence_length + 1, sequence_length):
        sequence = all_diffs[i:i + sequence_length]
        sequences.append(sequence)
        # シーケンスの中央のインデックスに対応するラベルを取得
        middle_index = i + sequence_length // 2
        sequence_label = all_labels[middle_index] if middle_index < len(all_labels) else all_labels[-1]
        sequence_labels.append(sequence_label)
    # NumPy 配列に変換
    sequences = np.array(sequences, dtype=np.float32)
    sequence_labels = np.array(sequence_labels, dtype=np.long)
    # データセットを作成し、データローダーを準備
    dataset = TensorDataset(
        torch.tensor(sequences).unsqueeze(-1),
        torch.tensor(sequence_labels)
    )
    train_dataset, test_dataset = train_test_split(dataset, test_size=0.2, random_state=41)
    train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False)
    # 統合されたデータセットにRNN モデルを初期化し、訓練する
    model = RNNClassifier(input_size, hidden_size, num_layers, num_classes)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    epochs = 100
    start_time_training = time.time() # 訓練の開始時間
    for epoch in range(epochs):  # 訓練ループ
        model.train()
        total_loss = 0
        correct = 0
        total = 0
        class_correct = [0] * num_classes
        class_total = [0] * num_classes

        for inputs, labels in train_loader:
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()

            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
            # クラス別の正確性を計算する
            for i in range(num_classes):
                class_mask = (labels == i)
                class_correct[i] += (predicted[class_mask] == labels[class_mask]).sum().item()
                class_total[i] += class_mask.sum().item()

        # 訓練の正確性と損失を出力する
        print(f'Epoch {epoch+1}/{epochs}, Loss: {total_loss / len(train_loader)}')
        print(f'Epoch {epoch+1}/{epochs}, Training Accuracy: {100 * correct / total}%')
        
        # 訓練データのクラスごとの正確性を表示する
        for i in range(num_classes):
            class_acc = 100 * class_correct[i] / class_total[i] if class_total[i] > 0 else 0
            print(f'Class {i} Training Accuracy: {class_acc:.2f}%')

        # 各エポックでテストデータを評価する
        model.eval()
        test_correct = 0
        test_total = 0
        test_class_correct = [0] * num_classes
        test_class_total = [0] * num_classes

        with torch.no_grad():
            for test_inputs, test_labels in test_loader:
                test_outputs = model(test_inputs)
                _, test_predicted = torch.max(test_outputs.data, 1)
                test_total += test_labels.size(0)
                test_correct += (test_predicted == test_labels).sum().item()
                
                # クラス別の正確性を計算する
                for i in range(num_classes):
                    test_class_mask = (test_labels == i)
                    test_class_correct[i] += (test_predicted[test_class_mask] == test_labels[test_class_mask]).sum().item()
                    test_class_total[i] += test_class_mask.sum().item()

        test_accuracy = 100 * test_correct / test_total
        print(f'Epoch {epoch+1}/{epochs}, Test Accuracy: {test_accuracy}%')
        
        # テストデータのクラスごとの正確性を表示する
        for i in range(num_classes):
            if test_class_total[i] > 0:
                test_class_acc = 100 * test_class_correct[i] / test_class_total[i]
                print(f'Epoch {epoch+1}/{epochs}, Class {i} Test Accuracy: {test_class_acc:.2f}%')
            else:
                print(f'Epoch {epoch+1}/{epochs}, Class {i} Test Accuracy: N/A (No samples)')

    # 訓練の終了時間
    end_time_training = time.time()
    time_taken_training = end_time_training - start_time_training
    print(f"Time taken for training: {time_taken_training} seconds.")
