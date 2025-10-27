import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import sys



sys.path.append('C:/Users/DS/Desktop/kimsihyun/my/st_gcn_master')


class ST_GCN_18(nn.Module):
    def __init__(self, in_channels=3, num_class=256):
        super(ST_GCN_18, self).__init__()
        
        self.num_points = 137  # body(25) + face(70) + hands(21*2)
        
        self.graph = self.create_graph()
        A = torch.tensor(self.graph, dtype=torch.float32, requires_grad=False)
        self.register_buffer('A', A)
        
        self.data_bn = nn.BatchNorm1d(in_channels * self.num_points)
        
        self.st_gcn_networks = nn.ModuleList([
            STGCNBlock(in_channels, 128, A.size(0), residual=False),
            STGCNBlock(128, 128, A.size(0)),
            STGCNBlock(128, 128, A.size(0)),
            STGCNBlock(128, 256, A.size(0), stride=2),
            STGCNBlock(256, 256, A.size(0)),
            STGCNBlock(256, 256, A.size(0)),
            STGCNBlock(256, 512, A.size(0), stride=2),
            STGCNBlock(512, 512, A.size(0)),
            STGCNBlock(512, 512, A.size(0)),
        ])
        
        # self.fcn = nn.Conv2d(512, num_class, kernel_size=1)
        self.embedding = nn.Linear(512, 256)

    def create_graph(self):
        self_link = [(i, i) for i in range(self.num_points)]
        
        body_links = [
            (1, 0), (1, 2), (1, 5), (1, 8),
            (8, 9), (8, 12),
            (2, 3), (3, 4),
            (5, 6), (6, 7),
            (9, 10), (10, 11),
            (11, 22), (11, 24),
            (12, 13), (13, 14),
            (14, 19), (14, 21),
            (0, 15), (0, 16), (15, 17), (16, 18)
        ]
        
        face_start_idx = 25
        face_links = []
        for i in range(16):
            face_links.append((face_start_idx + i, face_start_idx + i + 1))
        for i in range(4):
            face_links.append((face_start_idx + 17 + i, face_start_idx + 17 + i + 1))
            face_links.append((face_start_idx + 22 + i, face_start_idx + 22 + i + 1))
        for i in range(3):
            face_links.append((face_start_idx + 27 + i, face_start_idx + 27 + i + 1))
        for i in range(5):
            face_links.append((face_start_idx + 36 + i, face_start_idx + 36 + i + 1))
            face_links.append((face_start_idx + 42 + i, face_start_idx + 42 + i + 1))
        for i in range(11):
            face_links.append((face_start_idx + 48 + i, face_start_idx + 48 + i + 1))
            face_links.append((face_start_idx + 60 + i, face_start_idx + 60 + i + 1))
        
        lhand_start_idx = 95
        rhand_start_idx = 116
        
        hand_links = []
        for hand_start in [lhand_start_idx, rhand_start_idx]:
            for i in range(3):
                hand_links.append((hand_start + i, hand_start + i + 1))
            for i in range(3):
                hand_links.append((hand_start + 4 + i, hand_start + 4 + i + 1))
            for i in range(3):
                hand_links.append((hand_start + 8 + i, hand_start + 8 + i + 1))
            for i in range(3):
                hand_links.append((hand_start + 12 + i, hand_start + 12 + i + 1))
            for i in range(3):
                hand_links.append((hand_start + 16 + i, hand_start + 16 + i + 1))
            hand_links.extend([
                (hand_start, hand_start + 4),
                (hand_start + 4, hand_start + 8),
                (hand_start + 8, hand_start + 12),
                (hand_start + 12, hand_start + 16)
            ])
        
        body_face_links = [
            (0, face_start_idx + 27),
            (15, face_start_idx + 36),
            (16, face_start_idx + 42),
        ]
        
        body_hand_links = [
            (4, rhand_start_idx),
            (7, lhand_start_idx),
        ]
        
        all_links = (self_link + body_links + face_links + hand_links +
                     body_face_links + body_hand_links)
        
        A = np.zeros((self.num_points, self.num_points))
        for i, j in all_links:
            A[i, j] = 1
            A[j, i] = 1
        
        return A

    def forward(self, x):
        N, C, T, V = x.size()
        x = x.permute(0, 3, 1, 2).contiguous()  # (N, V, C, T)
        x = x.view(N, V * C, T)
        x = self.data_bn(x)
        x = x.view(N, V, C, T).permute(0, 2, 3, 1).contiguous()  # (N, C, T, V)
        
        for gcn in self.st_gcn_networks:
            x = gcn(x, self.A)
        
        x = F.avg_pool2d(x, x.size()[2:])  # (N, 512, 1, 1)
        x = x.view(N, -1)  # (N, 512)
        x = self.embedding(x)  # (N, 256)  # 추가된 부분
        
        return x  # 256차원 임베딩 출력

# ST-GCN의 기본 블록 
class STGCNBlock(nn.Module):
    def __init__(self, in_channels, out_channels, A_size, stride=1, residual=True):
        super(STGCNBlock, self).__init__()
        
        self.gcn = nn.Conv2d(
            in_channels,
            out_channels,
            kernel_size=(1, 1),
            stride=(1, 1)
        )
        
        self.tcn = nn.Sequential(
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(
                out_channels,
                out_channels,
                kernel_size=(9, 1),
                stride=(stride, 1),
                padding=(4, 0),
            ),
            nn.BatchNorm2d(out_channels),
            nn.Dropout(0.5, inplace=True)
        )
        
        if not residual:
            self.residual = lambda x: 0
        elif stride == 1 and in_channels == out_channels:
            self.residual = lambda x: x
        else:
            self.residual = nn.Sequential(
                nn.Conv2d(
                    in_channels,
                    out_channels,
                    kernel_size=1,
                    stride=(stride, 1)
                ),
                nn.BatchNorm2d(out_channels)
            )
        
        self.relu = nn.ReLU(inplace=True)
        
    def forward(self, x, A):
        res = self.residual(x)
        x = self.gcn(x)
        x = torch.einsum('nctv,vw->nctw', (x, A))
        x = self.tcn(x)
        x = self.relu(x + res)
        return x

