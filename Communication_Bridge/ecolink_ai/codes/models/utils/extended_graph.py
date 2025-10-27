import numpy as np

class ExtendedGraph():
    """
    확장된 OpenPose 포맷을 위한 그래프 구조
    - Body: 25개 키포인트
    - Face: 70개 키포인트
    - Hands: 각 21개 키포인트
    총 137개 키포인트
    """
    def __init__(self, strategy='spatial', max_hop=1, dilation=1):
        self.max_hop = max_hop
        self.dilation = dilation
        
        # 그래프 구조 초기화
        self.get_edge()
        self.hop_dis = self.get_hop_distance(
            self.num_node, self.edge, max_hop=max_hop)
        self.get_adjacency(strategy)

    def get_edge(self):
        # 전체 노드 수
        self.num_node = 137  # 25 + 70 + 21 + 21
        
        # 1. Self-connections
        self_link = [(i, i) for i in range(self.num_node)]
        
        # 2. Body connections (BODY_25 format)
        body_edges = [
            # 몸통
            (1, 0), (1, 2), (1, 5), (1, 8),  # Neck -> [Nose, RShoulder, LShoulder, MidHip]
            (8, 9), (8, 12),  # MidHip -> [RHip, LHip]
            # 오른쪽 팔
            (2, 3), (3, 4),  # RShoulder -> RElbow -> RWrist
            # 왼쪽 팔
            (5, 6), (6, 7),  # LShoulder -> LElbow -> LWrist
            # 오른쪽 다리
            (9, 10), (10, 11),  # RHip -> RKnee -> RAnkle
            (11, 22), (11, 24),  # RAnkle -> [RBigToe, RHeel]
            # 왼쪽 다리
            (12, 13), (13, 14),  # LHip -> LKnee -> LAnkle
            (14, 19), (14, 21),  # LAnkle -> [LBigToe, LHeel]
            # 얼굴
            (0, 15), (0, 16), (15, 17), (16, 18)  # Nose -> [REye, LEye] -> [REar, LEar]
        ]
        
        # 3. Face connections (70 points)
        face_start_idx = 25
        face_edges = []
        # 얼굴 윤곽선
        for i in range(16):
            face_edges.append((face_start_idx + i, face_start_idx + i + 1))
        # 눈썹
        for i in range(4):
            face_edges.append((face_start_idx + 17 + i, face_start_idx + 17 + i + 1))  # 오른쪽
            face_edges.append((face_start_idx + 22 + i, face_start_idx + 22 + i + 1))  # 왼쪽
        # 코
        for i in range(3):
            face_edges.append((face_start_idx + 27 + i, face_start_idx + 27 + i + 1))
        # 눈
        for i in range(5):
            face_edges.append((face_start_idx + 36 + i, face_start_idx + 36 + i + 1))  # 오른쪽
            face_edges.append((face_start_idx + 42 + i, face_start_idx + 42 + i + 1))  # 왼쪽
        # 입술
        for i in range(11):
            face_edges.append((face_start_idx + 48 + i, face_start_idx + 48 + i + 1))  # 외부
            face_edges.append((face_start_idx + 60 + i, face_start_idx + 60 + i + 1))  # 내부
        
        # 4. Hand connections (각 21 points)
        # 왼손 시작 인덱스
        lhand_start_idx = 95
        # 오른손 시작 인덱스
        rhand_start_idx = 116
        
        hand_edges = []
        for hand_start in [lhand_start_idx, rhand_start_idx]:
            # 엄지
            for i in range(3):
                hand_edges.append((hand_start + i, hand_start + i + 1))
            # 검지
            for i in range(3):
                hand_edges.append((hand_start + 4 + i, hand_start + 4 + i + 1))
            # 중지
            for i in range(3):
                hand_edges.append((hand_start + 8 + i, hand_start + 8 + i + 1))
            # 약지
            for i in range(3):
                hand_edges.append((hand_start + 12 + i, hand_start + 12 + i + 1))
            # 소지
            for i in range(3):
                hand_edges.append((hand_start + 16 + i, hand_start + 16 + i + 1))
            # 손바닥 연결
            hand_edges.extend([
                (hand_start, hand_start + 4),
                (hand_start + 4, hand_start + 8),
                (hand_start + 8, hand_start + 12),
                (hand_start + 12, hand_start + 16)
            ])
        
        # 5. Body-Face connections
        body_face_edges = [
            (0, face_start_idx + 27),  # nose to nose bridge
            (15, face_start_idx + 36),  # right eye to right eye corner
            (16, face_start_idx + 42),  # left eye to left eye corner
        ]
        
        # 6. Body-Hand connections
        body_hand_edges = [
            (4, rhand_start_idx),   # right wrist to right hand
            (7, lhand_start_idx),   # left wrist to left hand
        ]
        
        # 모든 엣지 통합
        self.edge = self_link + body_edges + face_edges + hand_edges + body_face_edges + body_hand_edges
        self.center = 1  # neck as center

    def get_hop_distance(self, num_node, edge, max_hop=1):
        A = np.zeros((num_node, num_node))
        for i, j in edge:
            A[j, i] = 1
            A[i, j] = 1

        # compute hop steps
        hop_dis = np.zeros((num_node, num_node)) + np.inf
        transfer_mat = [np.linalg.matrix_power(A, d) for d in range(max_hop + 1)]
        arrive_mat = (np.stack(transfer_mat) > 0)
        for d in range(max_hop, -1, -1):
            hop_dis[arrive_mat[d]] = d
        return hop_dis

    def get_adjacency(self, strategy):
        valid_hop = range(0, self.max_hop + 1, self.dilation)
        adjacency = np.zeros((self.num_node, self.num_node))
        for hop in valid_hop:
            adjacency[self.hop_dis == hop] = 1
        normalize_adjacency = self.normalize_digraph(adjacency)

        if strategy == 'uniform':
            A = np.zeros((1, self.num_node, self.num_node))
            A[0] = normalize_adjacency
            self.A = A
        elif strategy == 'distance':
            A = np.zeros((len(valid_hop), self.num_node, self.num_node))
            for i, hop in enumerate(valid_hop):
                A[i][self.hop_dis == hop] = normalize_adjacency[self.hop_dis == hop]
            self.A = A
        elif strategy == 'spatial':
            A = []
            for hop in valid_hop:
                a_root = np.zeros((self.num_node, self.num_node))
                a_close = np.zeros((self.num_node, self.num_node))
                a_further = np.zeros((self.num_node, self.num_node))
                for i in range(self.num_node):
                    for j in range(self.num_node):
                        if self.hop_dis[j, i] == hop:
                            if self.hop_dis[j, self.center] == self.hop_dis[i, self.center]:
                                a_root[j, i] = normalize_adjacency[j, i]
                            elif self.hop_dis[j, self.center] > self.hop_dis[i, self.center]:
                                a_close[j, i] = normalize_adjacency[j, i]
                            else:
                                a_further[j, i] = normalize_adjacency[j, i]
                if hop == 0:
                    A.append(a_root)
                else:
                    A.append(a_root + a_close)
                    A.append(a_further)
            A = np.stack(A)
            self.A = A
        else:
            raise ValueError("Do Not Exist This Strategy")

    def normalize_digraph(self, A):
        Dl = np.sum(A, 0)
        num_node = A.shape[0]
        Dn = np.zeros((num_node, num_node))
        for i in range(num_node):
            if Dl[i] > 0:
                Dn[i, i] = Dl[i]**(-1)
        AD = np.dot(A, Dn)
        return AD 