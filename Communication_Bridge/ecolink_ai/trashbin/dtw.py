import numpy as np
import json

def load_landmarks_from_json(json_path):
    with open(json_path, 'r') as f:
        data = json.load(f)

    seq = []
    num_pose = 33
    num_hand = 21
    num_nodes = num_pose + 2 * num_hand  # 총 75 노드

    for frame in data:
        frame_kps = []

        # 포즈 관절
        pose = frame.get('pose_landmarks') or []
        for i in range(num_pose):
            if i < len(pose):
                kp = pose[i]
                frame_kps.extend([kp['x'], kp['y'], kp['z'], kp['visibility']])
            else:
                frame_kps.extend([0, 0, 0, 0])

        # 왼손 관절
        lhand = frame.get('left_hand_landmarks') or []
        for i in range(num_hand):
            if i < len(lhand):
                kp = lhand[i]
                frame_kps.extend([kp['x'], kp['y'], kp['z'], kp['visibility']])
            else:
                frame_kps.extend([0, 0, 0, 0])

        # 오른손 관절
        rhand = frame.get('right_hand_landmarks') or []
        for i in range(num_hand):
            if i < len(rhand):
                kp = rhand[i]
                frame_kps.extend([kp['x'], kp['y'], kp['z'], kp['visibility']])
            else:
                frame_kps.extend([0, 0, 0, 0])

        seq.append(frame_kps)

    # 결과 ndarray: (프레임수, 75*4) 즉 (T, 300)
    return np.array(seq)

def euclidean_distance(vec1, vec2):
    # 두 벡터의 유클리드 거리 계산
    return np.linalg.norm(vec1 - vec2)

def dtw_multidim(seq1, seq2):
    n, m = len(seq1), len(seq2)
    dtw = np.full((n+1, m+1), np.inf)
    dtw[0, 0] = 0

    for i in range(1, n+1):
        for j in range(1, m+1):
            # 두 프레임의 전체 관절 좌표를 1차원 벡터로 변환 후 거리 계산
            cost = euclidean_distance(seq1[i-1].flatten(), seq2[j-1].flatten())
            dtw[i, j] = cost + min(dtw[i-1, j],    # 삽입
                                   dtw[i, j-1],    # 삭제
                                   dtw[i-1, j-1])  # 매칭

    return dtw[n, m]

if __name__ == "__main__":
    seq1 = load_landmarks_from_json("C:/Users/DS/Desktop/kimsihyun/Communication_Bridge/ecolink_ai/datas/keypoints_10_normalization/계단/KETI_SL_0000000128.json")
    seq2 = load_landmarks_from_json("C:/Users/DS/Desktop/kimsihyun/Communication_Bridge/ecolink_ai/datas/keypoints_10_normalization/계단/KETI_SL_0000000966.json")
    seq3 = load_landmarks_from_json("C:/Users/DS/Desktop/kimsihyun/Communication_Bridge/ecolink_ai/datas/keypoints_10_normalization/공원/KETI_SL_0000000974.json")

    # DTW 비교를 위해 reshape (프레임 수, 관절수, xyz)
    seq1 = seq1.reshape(len(seq1), 75, 4)
    seq2 = seq2.reshape(len(seq2), 75, 4)
    seq3 = seq3.reshape(len(seq3), 75, 4)

    # dtw 유사도 측정
    distance = dtw_multidim(seq1, seq2)
    print(f"1-2 DTW 거리: {distance}")
    distance = dtw_multidim(seq1, seq3)
    print(f"1-3 DTW 거리: {distance}")
