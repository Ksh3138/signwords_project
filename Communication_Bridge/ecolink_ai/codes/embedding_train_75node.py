import os
import json
import torch
from torch.utils.data import Dataset, DataLoader
import torch.nn as nn
import torch.optim as optim
# from models.st_gcn_18 import STGCNModel
from models.st_gcn_18_10 import STGCNModel
import matplotlib.pyplot as plt
from torch.utils.data import random_split, DataLoader

# 0919 수정 - epoch마다 loss를 plt로 그림
# 0926 수정 - train/validation 분할



class JsonDataset(Dataset):
    def __init__(self, root_dir, fixed_length=60):
        self.samples = []
        self.labels = []
        label_names = [d for d in sorted(os.listdir(root_dir)) if os.path.isdir(os.path.join(root_dir, d))]
        self.label2idx = {label: idx for idx, label in enumerate(label_names)}
        for label in label_names:
            folder = os.path.join(root_dir, label)
            for file in os.listdir(folder):
                if file.endswith('.json'):
                    self.samples.append(os.path.join(folder, file))
                    self.labels.append(self.label2idx[label])
        self.fixed_length = fixed_length

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        with open(self.samples[idx], 'r') as f:
            data = json.load(f)

        x = self.process_keypoints(data)
        y = self.labels[idx]
        return x, y

    def process_keypoints(self, frames_json):
        import numpy as np
        num_pose = 33
        num_hand = 21
        num_nodes = num_pose + num_hand * 2

        keypoints_all = []
        for frame in frames_json:
            pose = frame.get('pose_landmarks') or []
            lhand = frame.get('left_hand_landmarks') or []
            rhand = frame.get('right_hand_landmarks') or []

            frame_kps = []
            for i in range(num_pose):
                if i < len(pose):
                    kp = pose[i]
                    frame_kps.append([kp['x'], kp['y'], kp['z'], kp['visibility']])
                else:
                    frame_kps.append([0, 0, 0, 0])
            for i in range(num_hand):
                if i < len(lhand):
                    kp = lhand[i]
                    frame_kps.append([kp['x'], kp['y'], kp['z'], kp['visibility']])
                else:
                    frame_kps.append([0, 0, 0, 0])
            for i in range(num_hand):
                if i < len(rhand):
                    kp = rhand[i]
                    frame_kps.append([kp['x'], kp['y'], kp['z'], kp['visibility']])
                else:
                    frame_kps.append([0, 0, 0, 0])
            keypoints_all.append(frame_kps)

        T = len(keypoints_all)
        if T < self.fixed_length:
            padding = [[0, 0, 0, 0]] * num_nodes
            for _ in range(self.fixed_length - T):
                keypoints_all.append(padding)
        elif T > self.fixed_length:
            keypoints_all = keypoints_all[:self.fixed_length]

        arr = np.array(keypoints_all).transpose(2, 0, 1)  # (C, T, V)
        return torch.tensor(arr, dtype=torch.float32)

# 나누지 않음
# def train(root_dir,checkpoint_file):
#     #학습 데이터 준비
#     dataset = JsonDataset(root_dir)
#     dataloader = DataLoader(dataset, batch_size=32, shuffle=True, num_workers=2)

#     device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
#     # 모델 준비
#     model = STGCNModel(in_channels=4, num_class=len(dataset.label2idx)).to(device)
#     criterion = nn.CrossEntropyLoss()
#     optimizer = optim.Adam(model.parameters(), lr=1e-3)
#     best_loss = float('inf') #초기값을 무한대로 설정


#     # loss 기록
#     loss_values=[]


#     # 학습
#     for epoch in range(30):
#         model.train()
#         total_loss = 0

#         # 입력데이터 x, 레이블 y
#         for x, y in dataloader:
#             x, y = x.to(device), y.to(device)
#             optimizer.zero_grad() # 이전배치의gradient 초기화

#             outputs = model(x)
#             loss = criterion(outputs, y)
#             loss.backward()
#             optimizer.step()
#             total_loss += loss.item() * x.size(0)


#         # 현재 epoch까지의 평균 loss 계산
#         avg_loss = total_loss / len(dataset)
#         print(f"Epoch {epoch+1}, Loss: {avg_loss:.4f}")
        
#         # loss 기록
#         loss_values.append(avg_loss)

#         # loss가 이전보다 작으면, 모델과 파라미터와 optimizer 상태를, 체크포인트 파일에 저장
#         if avg_loss < best_loss:
#             best_loss = avg_loss
#             torch.save({
#                 'epoch': epoch + 1,
#                 'model_state_dict': model.state_dict(),
#                 'optimizer_state_dict': optimizer.state_dict(),
#                 'loss': avg_loss
#             }, checkpoint_file)
#             print("Checkpoint saved.")

#     return loss_values


# train / validation 나눔, epoch 50
def train(root_dir,checkpoint_file):
    #학습 데이터 준비

    # 1. 전체 데이터셋 생성
    full_dataset = JsonDataset(root_dir)

    # 2. 분할 비율 계산
    train_ratio = 0.8
    val_ratio = 0.2
    dataset_size = len(full_dataset)
    train_size = int(train_ratio * dataset_size)
    val_size = dataset_size - train_size

    # 3. 데이터셋 분할
    train_dataset, val_dataset = random_split(full_dataset, [train_size, val_size])

    # 4. DataLoader 생성
    train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True, num_workers=2)
    val_loader = DataLoader(val_dataset, batch_size=32, shuffle=False, num_workers=2)



    # cuda or cpu 준비
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    # 모델 준비
    model = STGCNModel(in_channels=4, num_class=len(full_dataset.label2idx)).to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=1e-3)
    best_loss = float('inf') #초기값을 무한대로 설정


    # loss 기록
    loss_values=[]


    # 학습
    for epoch in range(50):
        # 1. Train 단계
        model.train()
        total_train_loss = 0
        for x_train, y_train in train_loader:
            x_train, y_train = x_train.to(device), y_train.to(device)
            optimizer.zero_grad()
            outputs = model(x_train)
            loss = criterion(outputs, y_train)
            loss.backward()
            optimizer.step()
            total_train_loss += loss.item() * x_train.size(0)
        avg_train_loss = total_train_loss / len(train_dataset)

        # 2. Validation 단계
        model.eval()
        total_val_loss = 0
        with torch.no_grad():
            for x_val, y_val in val_loader:
                x_val, y_val = x_val.to(device), y_val.to(device)
                outputs = model(x_val)
                loss = criterion(outputs, y_val)
                total_val_loss += loss.item() * x_val.size(0)
        avg_val_loss = total_val_loss / len(val_dataset)
        loss_values.append(avg_val_loss)
        print(f"Epoch {epoch+1}, Train Loss: {avg_train_loss:.4f}, Val Loss: {avg_val_loss:.4f}")

        # validation loss가 가장 작을 때만 체크포인트 저장
        if avg_val_loss < best_loss:
            best_loss = avg_val_loss
            torch.save({
                'epoch': epoch + 1,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'loss': avg_val_loss
            }, checkpoint_file)
            print("Checkpoint saved.")

    return loss_values











if __name__ == '__main__':
    root_dir = 'C:/Users/DS/Desktop/kimsihyun/Communication_Bridge/ecolink_ai/datas/keypoints_pose_all/keypoints_10_normalization' 
    checkpoint_file = 'C:/Users/DS/Desktop/kimsihyun/Communication_Bridge/ecolink_ai/datas/75node10_movingNor_augmen0.pth'
    loss_values=train(root_dir,checkpoint_file)

    plt.plot(loss_values, marker='o')  # 인덱스가 자동으로 x축
    plt.xlabel('epoch')
    plt.ylabel('loss')
    plt.title('-')
    plt.grid(True)
    plt.show()
    
