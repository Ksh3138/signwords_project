import os
import json
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from models.st_gcn_18 import ST_GCN_18 

# 2. 데이터셋 클래스
class JsonVectorDataset(Dataset):
    def __init__(self, root_dir, fixed_length=60, fixed_joints=137):
        self.samples = []
        self.labels = []
        self.label2idx = {}
        self.fixed_length = fixed_length
        self.fixed_joints = fixed_joints
        label_names = sorted(os.listdir(root_dir))
        for idx, label in enumerate(label_names):
            label_path = os.path.join(root_dir, label)
            if not os.path.isdir(label_path):
                continue
            self.label2idx[label] = idx
            for fname in os.listdir(label_path):
                if fname.endswith('.json'):
                    self.samples.append(os.path.join(label_path, fname))
                    self.labels.append(idx)
        print(f"총 데이터: {len(self.samples)}개, 라벨 종류: {len(self.label2idx)}개")
        if len(self.samples) == 0:
            raise ValueError("데이터가 없습니다. 폴더/파일 경로 및 확장자를 확인하세요.")

    # 
    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        with open(self.samples[idx], 'r', encoding='utf-8') as f:
            data = json.load(f)
        keypoints = data['keypoints']  # [T, V, C]
        x = torch.tensor(keypoints, dtype=torch.float32)  # (T, V, C)
        T, V, C = x.shape
        fixed_T = self.fixed_length
        fixed_V = self.fixed_joints

        # 프레임 수 맞추기
        if T > fixed_T:
            x = x[:fixed_T]
        elif T < fixed_T:
            pad = torch.zeros((fixed_T - T, V, C), dtype=torch.float32)
            x = torch.cat([x, pad], dim=0)

        # 관절 수 맞추기
        if V < fixed_V:
            pad = torch.zeros((fixed_T, fixed_V - V, C), dtype=torch.float32)
            x = torch.cat([x, pad], dim=1)
        elif V > fixed_V:
            x = x[:, :fixed_V, :]  

        x = x.permute(2, 0, 1)  # (C, T, V)
        y = self.labels[idx]
        return x, y

print(torch.__version__)  # 설치된 PyTorch 버전
print(torch.cuda.is_available())  # CUDA 사용 가능 여부 (True면 GPU 지원)
print(torch.version.cuda)         # PyTorch가 인식하는 CUDA 버전 출력

# 1. 데이터 불러오기
root_dir = 'C:/Users/DS/Desktop/kimsihyun/my/datas/vector'
batch_size = 32
num_epochs = 30
learning_rate = 1e-3

# 2. 데이터셋과 데이터로더 준비
dataset = JsonVectorDataset(root_dir)
num_class = len(dataset.label2idx)
dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

# 3. 모델, 손실 함수, 옵티마이저 설정
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Using device: {device}")
model = ST_GCN_18(in_channels=3, num_class=num_class).to(device)  # 실제 모델로 교체
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=learning_rate)


best_loss = float('inf')
best_checkpoint = None
# 4. 학습 진행
for epoch in range(num_epochs):
    model.train()
    total_loss = 0
    for X, y in dataloader:
        X = X.to(device)
        y = y.to(device)
        optimizer.zero_grad()
        outputs = model(X)
        loss = criterion(outputs, y)
        loss.backward()
        optimizer.step()
        total_loss += loss.item() * X.size(0)
    avg_loss = total_loss / len(dataset)
    print(f"Epoch {epoch+1}/{num_epochs}, Loss: {avg_loss:.4f}")

    # 최고 성능 체크포인트 저장
    if avg_loss < best_loss:
        best_loss = avg_loss
        best_checkpoint = {
            'epoch': epoch + 1,
            'state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'loss': avg_loss,
        }

# 학습이 끝난 뒤, 최종 체크포인트 파일로 저장
if best_checkpoint is not None:
    torch.save(best_checkpoint, "C:/Users/DS/Desktop/kimsihyun/my/st_gcn_master/checkpoints/modified_stgcn_init2.pth")
    print("최종 체크포인트 저장 완료!")