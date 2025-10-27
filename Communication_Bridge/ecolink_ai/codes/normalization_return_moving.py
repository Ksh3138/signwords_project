import json
import os
import numpy as np


# 1. 잡음 - 앞뒤로 움직이지 않는 프레임 제거

# 2. 신체조건에대해..
    # 2.1. 프레임을 잘라서 직사각형을 만든다. 조건: 직사각형 안에는 모든 프레임의 모든 키포인트가 들어가야함
    # 2.2. 직사각형의 우측상단을 (1,0) 좌측상단을 (0,0) 좌측하단을 (1,1) 우측하단을 (0,1)로 두고 나머지 keypoint들의 좌표값을 바꾼다

# 3. 속도 - 보류
    # 3.방법1. 보간or삭제 해서 프레임수 맞추기
    # 3.방법2. z score
    # 3.방법3. DTW




def is_moving(prev_points, curr_points):
    flow = curr_points - prev_points  # shape: (N, 2)
    magnitude = np.linalg.norm(flow, axis=1)  # shape: (N,)
    threshold=0.2
    moved=0
    for i, (dx, dy) in enumerate(flow):
        if (dx*dx+dy*dy) > threshold :
            moved+=1
    
    if moved>5:
        state="moving"
    else:
        state="stop"

    return state


def get_points(frame):
    points = []

    pose = frame.get('pose_landmarks') or []
    for i in range(33):
        if i < len(pose):
            kp = pose[i]
            points.append([kp['x'], kp['y']])
        else:
            points.append([0, 0])
    # 왼손 21개
    lhand = frame.get('left_hand_landmarks') or []
    for i in range(21):
        if i < len(lhand):
            kp = lhand[i]
            points.append([kp['x'], kp['y']])
        else:
            points.append([0, 0])
    # 오른손 21개
    rhand = frame.get('right_hand_landmarks') or []
    for i in range(21):
        if i < len(rhand):
            kp = rhand[i]
            points.append([kp['x'], kp['y']])
        else:
            points.append([0, 0])
    return np.array(points)  # shape: (54, 2)





def normalization(data):
    # # 1. 파일 오픈
    # with open(input_file, 'r') as f:
    #     data = json.load(f)



    # 2. 정규화


    # 2.1. 잡음 - 앞뒤로 양손이 모두 None인 프레임 제거
    try:
        start_idx = None
        for i in range(len(data) - 1):
            prev_points = get_points(data[i])
            curr_points = get_points(data[i+1])
            state = is_moving(prev_points, curr_points)
            if state == "moving":
                start_idx = i
                break

        end_idx = None
        for i in range(len(data) - 2, -1, -1):
            prev_points = get_points(data[i])
            curr_points = get_points(data[i+1])
            state = is_moving(prev_points, curr_points)
            if state == "moving":
                end_idx = i+1
                break

        data = data[start_idx:end_idx + 1]
        for new_idx, frame in enumerate(data):
            frame['frame'] = new_idx
    except:
        pass



    # 2.2. 신체조건에대해 - 직사각형 잘라서 landmark들 이동
    keypoints = []
    for frame in data:
        for part in ['pose_landmarks', 'left_hand_landmarks', 'right_hand_landmarks']:
            landmarks = frame.get(part)
            if isinstance(landmarks, list) and len(landmarks) > 0:
                for landmark in landmarks:
                    keypoints.append([landmark['x'], landmark['y'], landmark['z']])
    keypoints = np.array(keypoints)

    min_x, min_y = np.min(keypoints[:, 0]), np.min(keypoints[:, 1])
    max_x, max_y = np.max(keypoints[:, 0]), np.max(keypoints[:, 1])
    norm_x = (keypoints[:, 0] - min_x) / (max_x - min_x)
    norm_y = (keypoints[:, 1] - min_y) / (max_y - min_y)
    norm_z = keypoints[:, 2]

    keypoints = np.stack([norm_x, norm_y, norm_z], axis=1)


    # 2.3. 동작 수행 속도에 대해 - 일단 보류





    # 3. 결과 저장
    idx = 0
    for frame in data:
        for part in ['pose_landmarks', 'left_hand_landmarks', 'right_hand_landmarks']:
            if part in frame and frame[part] is not None:
                for landmark in frame[part]:
                    landmark['x'] = keypoints[idx, 0]
                    landmark['y'] = keypoints[idx, 1]
                    landmark['z'] = keypoints[idx, 2]
                    idx += 1

                

    # with open(output_file, 'w') as f:
    #     json.dump(data, f, indent=2)
    return data




# if __name__ == "__main__":
#     input_dir = 'C:/Users/DS/Desktop/kimsihyun/Communication_Bridge/ecolink_ai/datas/keypoints_10'
#     output_dir = 'C:/Users/DS/Desktop/kimsihyun/Communication_Bridge/ecolink_ai/datas/keypoints_10_normalization'

#     for root, dirs, files in os.walk(input_dir):
#         for file in files:
#             input_file = os.path.join(root, file).replace('\\', '/')
#             rel_path = os.path.relpath(input_file, input_dir).replace('\\', '/')
#             output_file = os.path.join(output_dir, rel_path).replace('\\', '/')
#             os.makedirs(os.path.dirname(output_file), exist_ok=True)
#             normalization(input_file, output_file)
#             print(f'Normalized and saved: {output_file}')




