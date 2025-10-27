import numpy as np
import json
import os
from pathlib import Path
from datetime import datetime

# 1. 파일 오픈
def load_json_data(input_file):
    with open(input_file, 'r') as f:
        data = json.load(f)
    
    return data

# 2. json 데이터를 numpy 배열로 만들기
def process_keypoints(json_data):
    frames = []
    
    json_data = json_data["frames"] #frames 키를 통해 모든프레임의 모든키포인트 좌표값 가져옴

    #프레임마다의 신체,얼굴,왼손,오른손 키포인트 좌표를 차례로 리스트에 추가
    for frame_idx, frame_data in enumerate(json_data):
        keypoints = []
        
        if 'body_keypoints_2d' in frame_data:
            body_keypoints = np.array(frame_data['body_keypoints_2d']).reshape(-1, 3)
            keypoints.extend(body_keypoints.tolist())
        else:
            keypoints.extend([[0, 0, 0]] * 25)
            
        if 'face_keypoints_2d' in frame_data:
            face_keypoints = np.array(frame_data['face_keypoints_2d']).reshape(-1, 3)
            keypoints.extend(face_keypoints.tolist())
        else:
            keypoints.extend([[0, 0, 0]] * 70)
            
        if ('hand_left_keypoints_2d' in frame_data and 
            len(frame_data['hand_left_keypoints_2d']) > 0 and 
            not all(x == 0 for x in frame_data['hand_left_keypoints_2d'])):
            left_hand = np.array(frame_data['hand_left_keypoints_2d']).reshape(-1, 3)
            keypoints.extend(left_hand.tolist())
        else:
            keypoints.extend([[0, 0, 0]] * 21)
            
        if ('hand_right_keypoints_2d' in frame_data and 
            len(frame_data['hand_right_keypoints_2d']) > 0 and 
            not all(x == 0 for x in frame_data['hand_right_keypoints_2d'])):
            right_hand = np.array(frame_data['hand_right_keypoints_2d']).reshape(-1, 3)
            keypoints.extend(right_hand.tolist())
        else:
            keypoints.extend([[0, 0, 0]] * 21)
        
        frames.append(keypoints)
    
    # numpy array 형태로 변경 - 딥러닝 모델 입력에 사용하기 위해
    frames_array = np.array(frames)
    
    return frames_array


# 3-1. 신체 조건에 대해 정규화
def normalize_by_body_reference(data):
    # 어깨 중심점 계산
    right_shoulder = data[:, 2, :2] 
    left_shoulder = data[:, 5, :2]  
    shoulder_center = (right_shoulder + left_shoulder) / 2
    
    # 어깨 사이 거리 계산
    shoulder_distance = np.linalg.norm(right_shoulder - left_shoulder, axis=1)
    shoulder_distance = shoulder_distance.reshape(-1, 1)
    shoulder_distance[shoulder_distance == 0] = 1
    
    # 어깨중심이 (0,0) 되도록 나머지 좌표 이동, 어깨사이거리로 나눠서 -1~1 사이 값으로 만들기
    normalized_data = data.copy()
    for i in range(data.shape[1]): 
        normalized_data[:, i, :2] = (data[:, i, :2] - shoulder_center) / shoulder_distance
    
    return normalized_data

# 3-2. 동작 속도에 대해 정규화
def normalize_sequence(data):
    sequence_data = data.copy()
    
    # 키포인트마다 모든 프레임의 좌표 평균,표준편차 계산
    mean_pose = np.mean(data[:, :, :2], axis=0)
    std_pose = np.std(data[:, :, :2], axis=0)
    std_pose[std_pose == 0] = 1
    
    # z-score 정규화: 키포인트마다 모든 프레임의 좌표 평균이0,표준편차가1이 되도록 값 수정
    sequence_data[:, :, :2] = (data[:, :, :2] - mean_pose) / std_pose
    
    return sequence_data

# 3-3. 신뢰도가 낮은 키포인트 필터링
def filter_low_confidence(data, confidence_threshold=0.3):
    filtered_data = data.copy()
    confidence_scores = data[:, :, 2]
    
    # 키포인트별로 신뢰도가 0.3미만이면 true, 이상이면 false인 mask
    low_confidence_mask = confidence_scores < confidence_threshold
    
    # mask가 true면 이전 프레임 좌표로 대체
    for frame in range(1, len(data)):
        current_low_conf = low_confidence_mask[frame]
        if np.any(current_low_conf):
            filtered_data[frame, current_low_conf, :2] = filtered_data[frame-1, current_low_conf, :2]
    
    return filtered_data


# 3. 정규화 
def normalize_coordinates(data):
    if len(data) == 0:
        return data
    
    # numpy형태인지확인
    if not isinstance(data, np.ndarray):
        data = np.array(data)
    
    data = normalize_by_body_reference(data)
    data = normalize_sequence(data)
    data = filter_low_confidence(data)
    
    return data

# 4. 결과 저장
def save_vector_data(data, metadata, output_path):
    output_data = {
        "metadata": metadata,
        "keypoints": data,
        "created_at": datetime.now().isoformat()
    }
    
    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(output_data, f, ensure_ascii=False, indent=2)

# openpose로 추출한 좌표 정보를 정규화
def vector_normalization(input_file, output_dir):
    input_file = Path(input_file)
    output_dir = Path(output_dir)
    
    json_data = load_json_data(input_file)
    vector_data = process_keypoints(json_data)
    vector_data = normalize_coordinates(vector_data)
    
    metadata = {
        "original_file": input_file.name,
        "frame_count": vector_data.shape[0] if vector_data is not None and vector_data.size > 0 else 0,
        "keypoint_count": vector_data.shape[1] if vector_data is not None and vector_data.size > 0 else 0,
        "dimensions": 3,  # x, y, confidence
        "normalized": True,
        "normalization_info": {
            "body_reference": "shoulder_center",
            "sequence_normalized": True,
            "confidence_filtered": True
        },
        "keypoint_info": {
            "body": 25,
            "face": 70,
            "hand_left": 21,
            "hand_right": 21,
            "total": 137
        }
    }

    # 정규화 결과를 json 파일에 저장
    output_path = output_dir / f"{input_file.stem}_vector.json"
    save_vector_data(vector_data.tolist() if vector_data is not None else [], metadata, output_path)
    print(f"정규화 완료 {input_file.name} -> {output_path.name}")

    return str(output_path).replace('\\', '/')

if __name__ == "__main__":
    input_file = "C:/Users/DS/Desktop/kimsihyun/my/datas/keypoints/KETI_SL_0000006871.json"  
    output_dir = "C:/Users/DS/Desktop/kimsihyun/my/datas/vector" 
    
    print(vector_normalization(input_file, output_dir))



    # input_dir = "C:/Users/DS/Desktop/kimsihyun/my/datas/keypoints"
    # output_dir = "C:/Users/DS/Desktop/kimsihyun/my/datas/vector"

    # # input_dir의 모든 파일 리스트 얻기
    # files = [f for f in os.listdir(input_dir) if os.path.isfile(os.path.join(input_dir, f))]

    # for file in files:
    #     input_file = os.path.join(input_dir, file)
    #     print(vector_normalization(input_file, output_dir))

