import numpy as np
import cv2
import os
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
from sklearn.model_selection import train_test_split
import time

num_classes = 4

def frame_diff(frame1, frame2):
    frame1_gray = cv2.cvtColor(frame1, cv2.COLOR_BGR2GRAY)
    frame2_gray = cv2.cvtColor(frame2, cv2.COLOR_BGR2GRAY)
    diff = cv2.absdiff(frame1_gray, frame2_gray)
    return diff

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
            sum_result=np.sum(diff, axis=0) 
            frame_list.append(sum_result)
        count += 1
        frame1 = frame2
    cap.release()
    return frame_list

def calculate_frame_diffs_for_all_videos(video_file_paths, steps):
    start_time = time.time()  # 計算開始時間を記録

    output_list = []
    for video_file_path in video_file_paths:
        sum_result = calculate_frame_diffs(video_file_path, step=steps)
        output_list.append(sum_result)
    np.save(f'input_data_{steps}.npy', output_list)

    end_time = time.time()  # 計算終了時間を記録
    elapsed_time = end_time - start_time  # 経過時間を計算
    print(f"Frame diffs calculation time: {elapsed_time:.2f} seconds")
class SimpleClassifier(nn.Module):
    def __init__(self, input_size, hidden_size):
        super(SimpleClassifier, self).__init__()
        self.fc1 = nn.Linear(input_size, hidden_size)
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(hidden_size, num_classes)

    def forward(self, x):
        x = x.view(x.size(0), -1)
        x = self.fc1(x)
        x = self.relu(x)
        x = self.fc2(x)
        return x

def train_and_evaluate_model(video_file_paths, steps):
    start_time = time.time()  # 計算開始時間を記録
    label_dir = 'label_4'  # ４分類のラベルファイルが保存されているディレクトリ
    step_str = f"step{steps}"  # ステップ数を文字列に変換
    all_labels = []
    for video_file_path in video_file_paths:
        file_name = os.path.basename(video_file_path)  # ファイル名を取得
        label_file_name = file_name.replace(".mp4", f"_{step_str}_labels.npy")  # ラベルファイル名を生成
        label_file_path = os.path.join(label_dir, label_file_name)  # ラベルファイルのフルパスを生成
        labels = np.load(label_file_path)  # ラベルファイルを読み込む
        all_labels.extend(labels)  # 全ラベルリストに追加
    # 各ビデオファイルのフレーム差分のデータを読み込む
    input_data_path = f'input_data_{steps}.npy'
    X_combined = np.load(input_data_path).astype(np.float32)
    y_combined = np.array(all_labels).flatten()  # 全ラベルを1次元配列に変換
    X_combined = X_combined.reshape(-1, 1920)
    print("X_combined new shape:", X_combined.shape)
    print("X_combined shape:", X_combined.shape)
    print("y_combined shape:", y_combined.shape)

    X = torch.tensor(X_combined, dtype=torch.float32)
    y = torch.tensor(y_combined, dtype=torch.long)
    assert X.shape[0] == y.shape[0], "X and y must have the same number of samples"
        
    dataset = TensorDataset(X, y)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=40)
    train_dataset = TensorDataset(X_train, y_train)
    test_dataset = TensorDataset(X_test, y_test)

    train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False)

    model = SimpleClassifier(input_size=X.shape[1], hidden_size=64)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.0005)
    epochs = 100
    for epoch in range(epochs):
        model.train()
        for inputs, labels in train_loader:
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, labels.view(-1))
            loss.backward()
            optimizer.step()

        model.eval()
        correct = 0
        total = 0
        correct_per_class = [0] * num_classes
        total_per_class = [0] * num_classes
        with torch.no_grad():
            for inputs, labels in test_loader:
                outputs = model(inputs)
                _, predicted = torch.max(outputs, 1)
                total += labels.size(0)
                correct += (predicted == labels.view(-1)).sum().item()

                for label, prediction in zip(labels.view(-1), predicted):
                    total_per_class[label.item()] += 1
                    if label == prediction:
                        correct_per_class[label.item()] += 1

        accuracy = correct / total
        print(f"Epoch {epoch+1}/{epochs}, Test Accuracy: {accuracy}")

        for i in range(num_classes):
            if total_per_class[i] > 0:
                accuracy_per_class = correct_per_class[i] / total_per_class[i]
                print(f"Class {i}: Accuracy = {accuracy_per_class:.4f}")
            else:
                print(f"Class {i}: No samples")

    end_time = time.time()  # 計算終了時間を記録
    elapsed_time = end_time - start_time  # 経過時間を計算
    print(f"Training and evaluation time: {elapsed_time:.2f} seconds")
    
if __name__ == "__main__":
    video_file_paths = [
        "/LARGE0/gr10034/b31927/movie/dev121/09/Camera01_20220909002017.mp4",
    "/LARGE0/gr10034/b31927/movie/dev121/09/Camera01_20220909003017.mp4",

        # 他のファイルパス...
    ]
    steps = 10
    frame_rate = 25
    seconds = 60

    calculate_frame_diffs_for_all_videos(video_file_paths, steps)

    input_data_path = f'input_data_{steps}.npy'

    train_and_evaluate_model(video_file_paths, steps)
