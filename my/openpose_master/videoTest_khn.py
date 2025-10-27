import cv2
import cv2 as cv
import numpy as np
import os
import json
import time
from datetime import datetime

# 신체 키포인트 정의
BODY_PARTS = {
    "Nose": 0, "Neck": 1, "RShoulder": 2, "RElbow": 3, "RWrist": 4,
    "LShoulder": 5, "LElbow": 6, "LWrist": 7, "MidHip": 8, "RHip": 9,
    "RKnee": 10, "RAnkle": 11, "LHip": 12, "LKnee": 13, "LAnkle": 14,
    "REye": 15, "LEye": 16, "REar": 17, "LEar": 18, "LBigToe": 19,
    "LSmallToe": 20, "LHeel": 21, "RBigToe": 22, "RSmallToe": 23, "RHeel": 24
}

POSE_PAIRS = [
    ["Neck", "Nose"],
    ["Neck", "RShoulder"], ["RShoulder", "RElbow"], ["RElbow", "RWrist"],
    ["Neck", "LShoulder"], ["LShoulder", "LElbow"], ["LElbow", "LWrist"],
    ["Neck", "MidHip"],
    ["MidHip", "RHip"], ["RHip", "RKnee"], ["RKnee", "RAnkle"],
    ["RAnkle", "RHeel"], ["RHeel", "RBigToe"], ["RBigToe", "RSmallToe"],
    ["MidHip", "LHip"], ["LHip", "LKnee"], ["LKnee", "LAnkle"],
    ["LAnkle", "LHeel"], ["LHeel", "LBigToe"], ["LBigToe", "LSmallToe"],
    ["Nose", "REye"], ["REye", "REar"],
    ["Nose", "LEye"], ["LEye", "LEar"]
]

HAND_PARTS = {
    "Wrist": 0,
    "ThumbMetacarpal": 1, "ThumbProximal": 2, "ThumbMiddle": 3, "ThumbDistal": 4,
    "IndexFingerMetacarpal": 5, "IndexFingerProximal": 6, "IndexFingerMiddle": 7, "IndexFingerDistal": 8,
    "MiddleFingerMetacarpal": 9, "MiddleFingerProximal": 10, "MiddleFingerMiddle": 11, "MiddleFingerDistal": 12,
    "RingFingerMetacarpal": 13, "RingFingerProximal": 14, "RingFingerMiddle": 15, "RingFingerDistal": 16,
    "LittleFingerMetacarpal": 17, "LittleFingerProximal": 18, "LittleFingerMiddle": 19, "LittleFingerDistal": 20,
}

HAND_PAIRS = [
    ["Wrist", "ThumbMetacarpal"], ["ThumbMetacarpal", "ThumbProximal"],
    ["ThumbProximal", "ThumbMiddle"], ["ThumbMiddle", "ThumbDistal"],
    ["Wrist", "IndexFingerMetacarpal"], ["IndexFingerMetacarpal", "IndexFingerProximal"],
    ["IndexFingerProximal", "IndexFingerMiddle"], ["IndexFingerMiddle", "IndexFingerDistal"],
    ["Wrist", "MiddleFingerMetacarpal"], ["MiddleFingerMetacarpal", "MiddleFingerProximal"],
    ["MiddleFingerProximal", "MiddleFingerMiddle"], ["MiddleFingerMiddle", "MiddleFingerDistal"],
    ["Wrist", "RingFingerMetacarpal"], ["RingFingerMetacarpal", "RingFingerProximal"],
    ["RingFingerProximal", "RingFingerMiddle"], ["RingFingerMiddle", "RingFingerDistal"],
    ["Wrist", "LittleFingerMetacarpal"], ["LittleFingerMetacarpal", "LittleFingerProximal"],
    ["LittleFingerProximal", "LittleFingerMiddle"], ["LittleFingerMiddle", "LittleFingerDistal"]
]

FACE_PARTS = {
    "Jaw1": 0, "Jaw2": 1, "Jaw3": 2, "Jaw4": 3, "Jaw5": 4, "Jaw6": 5, "Jaw7": 6, "Jaw8": 7, "Jaw9": 8,
    "REyebrow1": 17, "REyebrow2": 18, "REyebrow3": 19, "REyebrow4": 20, "REyebrow5": 21,
    "LEyebrow1": 22, "LEyebrow2": 23, "LEyebrow3": 24, "LEyebrow4": 25, "LEyebrow5": 26,
    "NoseBridge1": 27, "NoseBridge2": 28, "NoseBridge3": 29, "NoseBridge4": 30,
    "NoseBottom1": 31, "NoseBottom2": 32, "NoseBottom3": 33, "NoseBottom4": 34, "NoseBottom5": 35,
    "REye1": 36, "REye2": 37, "REye3": 38, "REye4": 39, "REye5": 40, "REye6": 41,
    "LEye1": 42, "LEye2": 43, "LEye3": 44, "LEye4": 45, "LEye5": 46, "LEye6": 47,
    "MouthOut1": 48, "MouthOut2": 49, "MouthOut3": 50, "MouthOut4": 51, "MouthOut5": 52, "MouthOut6": 53,
    "MouthOut7": 54, "MouthOut8": 55, "MouthOut9": 56, "MouthOut10": 57, "MouthOut11": 58, "MouthOut12": 59,
    "MouthIn1": 60, "MouthIn2": 61, "MouthIn3": 62, "MouthIn4": 63, "MouthIn5": 64, "MouthIn6": 65,
    "MouthIn7": 66, "MouthIn8": 67, "MouthIn9": 68, "MouthIn10": 69
}

FACE_PAIRS = [
    ["Jaw1", "Jaw2"], ["Jaw2", "Jaw3"], ["Jaw3", "Jaw4"], ["Jaw4", "Jaw5"],
    ["Jaw5", "Jaw6"], ["Jaw6", "Jaw7"], ["Jaw7", "Jaw8"], ["Jaw8", "Jaw9"],
    ["REyebrow1", "REyebrow2"], ["REyebrow2", "REyebrow3"], ["REyebrow3", "REyebrow4"], ["REyebrow4", "REyebrow5"],
    ["LEyebrow1", "LEyebrow2"], ["LEyebrow2", "LEyebrow3"], ["LEyebrow3", "LEyebrow4"], ["LEyebrow4", "LEyebrow5"],
    ["NoseBridge1", "NoseBridge2"], ["NoseBridge2", "NoseBridge3"], ["NoseBridge3", "NoseBridge4"],
    ["NoseBottom1", "NoseBottom2"], ["NoseBottom2", "NoseBottom3"], ["NoseBottom3", "NoseBottom4"], ["NoseBottom4", "NoseBottom5"],
    ["REye1", "REye2"], ["REye2", "REye3"], ["REye3", "REye4"], ["REye4", "REye5"], ["REye5", "REye6"], ["REye6", "REye1"],
    ["LEye1", "LEye2"], ["LEye2", "LEye3"], ["LEye3", "LEye4"], ["LEye4", "LEye5"], ["LEye5", "LEye6"], ["LEye6", "LEye1"],
    ["MouthOut1", "MouthOut2"], ["MouthOut2", "MouthOut3"], ["MouthOut3", "MouthOut4"], ["MouthOut4", "MouthOut5"],
    ["MouthOut5", "MouthOut6"], ["MouthOut6", "MouthOut7"], ["MouthOut7", "MouthOut8"], ["MouthOut8", "MouthOut9"],
    ["MouthOut9", "MouthOut10"], ["MouthOut10", "MouthOut11"], ["MouthOut11", "MouthOut12"], ["MouthOut12", "MouthOut1"],
    ["MouthIn1", "MouthIn2"], ["MouthIn2", "MouthIn3"], ["MouthIn3", "MouthIn4"], ["MouthIn4", "MouthIn5"],
    ["MouthIn5", "MouthIn6"], ["MouthIn6", "MouthIn7"], ["MouthIn7", "MouthIn8"], ["MouthIn8", "MouthIn9"],
    ["MouthIn9", "MouthIn10"], ["MouthIn10", "MouthIn1"]
]

# 모델 경로 설정
bodyProtoFile = "C:/Users/DS/Desktop/kimsihyun/my/openpose_master/models/pose/body_25/pose_deploy.prototxt"
bodyWeightsFile = "C:/Users/DS/Desktop/kimsihyun/my/openpose_master/models/pose/body_25/pose_iter_584000.caffemodel"
handProtoFile = "C:/Users/DS/Desktop/kimsihyun/my/openpose_master/models/hand/pose_deploy.prototxt"
handWeightsFile = "C:/Users/DS/Desktop/kimsihyun/my/openpose_master/models/hand/pose_iter_102000.caffemodel"
faceProtoFile = "C:/Users/DS/Desktop/kimsihyun/my/openpose_master/models/face/pose_deploy.prototxt"
faceWeightsFile = "C:/Users/DS/Desktop/kimsihyun/my/openpose_master/models/face/pose_iter_116000.caffemodel"

# 네트워크 로드
bodyNet = cv.dnn.readNetFromCaffe(bodyProtoFile, bodyWeightsFile)
handNet = cv.dnn.readNetFromCaffe(handProtoFile, handWeightsFile)
faceNet = cv.dnn.readNetFromCaffe(faceProtoFile, faceWeightsFile)

# 반드시 백엔드와 타겟을 모두 CUDA로!
bodyNet.setPreferableBackend(cv.dnn.DNN_BACKEND_CUDA)
bodyNet.setPreferableTarget(cv.dnn.DNN_TARGET_CUDA)
handNet.setPreferableBackend(cv.dnn.DNN_BACKEND_CUDA)
handNet.setPreferableTarget(cv.dnn.DNN_TARGET_CUDA)
faceNet.setPreferableBackend(cv.dnn.DNN_BACKEND_CUDA)
faceNet.setPreferableTarget(cv.dnn.DNN_TARGET_CUDA)

print(cv2.getBuildInformation())

# 상수 정의
MAX_FINGER_LENGTH = 100
HAND_CROP_SIZE = 300

def body(frame):
    """신체 키포인트 감지 함수"""
    frameWidth = frame.shape[1]
    frameHeight = frame.shape[0]
   
    inWidth = 368
    inHeight = 368
    threshold = 0.1
   
    inpBlob = cv.dnn.blobFromImage(frame, 1.0/255, (inWidth, inHeight), (0, 0, 0), swapRB=False, crop=False)
   
    bodyNet.setInput(inpBlob)
    output = bodyNet.forward()
   
    points = []
    confidences = []
   
    for i in range(len(BODY_PARTS)):
        probMap = output[0, i, :, :]
       
        minVal, prob, minLoc, point = cv.minMaxLoc(probMap)
       
        x = int((frameWidth * point[0]) / output.shape[3])
        y = int((frameHeight * point[1]) / output.shape[2])
       
        if prob > threshold:
            points.append((x, y))
            confidences.append(float(prob))
        else:
            points.append(None)
            confidences.append(0.0)
   
    return points, confidences

def interpolate_missing_keypoints(points):
    """감지되지 않은 키포인트를 보간"""
    if not points:
        return points
       
    # 각 손가락마다 처리
    for finger in range(5):
        base_idx = finger*4 + 1
       
        # 경우 1: 첫 관절과 마지막 관절만 있는 경우
        if points[base_idx] and points[base_idx+3] and not points[base_idx+1] and not points[base_idx+2]:
            # 선형 보간으로 중간 관절 위치 추정
            points[base_idx+1] = (
                (2*points[base_idx][0] + points[base_idx+3][0])//3,
                (2*points[base_idx][1] + points[base_idx+3][1])//3
            )
            points[base_idx+2] = (
                (points[base_idx][0] + 2*points[base_idx+3][0])//3,
                (points[base_idx][1] + 2*points[base_idx+3][1])//3
            )
       
        # 경우 2: 중간 관절만 없는 경우
        elif points[base_idx] and points[base_idx+2] and not points[base_idx+1]:
            points[base_idx+1] = (
                (points[base_idx][0] + points[base_idx+2][0])//2,
                (points[base_idx][1] + points[base_idx+2][1])//2
            )
   
    return points

def apply_anatomical_constraints(points):
    """손의 해부학적 구조를 기반으로 키포인트 정확도 개선"""
    if not points[0]:  # 손목이 없으면 처리 불가
        return points
       
    # 손가락 길이 제약 (실제 손에서 마디 길이는 일정 범위 내에 있음)
    for finger in range(5):
        base = finger*4 + 1  # 각 손가락 시작점
       
        # 각 마디 길이가 해부학적으로 가능한 범위인지 확인
        for i in range(3):  # 각 손가락의 3개 관절
            if points[base+i] and points[base+i+1]:
                distance = np.sqrt((points[base+i][0]-points[base+i+1][0])**2 +
                                 (points[base+i][1]-points[base+i+1][1])**2)
               
                # 길이가 비정상적으로 길거나 짧으면 보정
                if distance > MAX_FINGER_LENGTH:  # 예시 임계값
                    # 방향은 유지하되 길이 조정
                    direction = np.array([points[base+i+1][0]-points[base+i][0],
                                        points[base+i+1][1]-points[base+i][1]])
                    if np.linalg.norm(direction) > 0:
                        direction = direction / np.linalg.norm(direction) * 30  # 정상 길이로 조정
                        points[base+i+1] = (int(points[base+i][0] + direction[0]),
                                          int(points[base+i][1] + direction[1]))
   
    return points

def filter_unstable_keypoints(current_points, prev_points, prev_prev_points=None, threshold=10):
    """개선된 키포인트 필터링 함수 - 움직임 예측과 적응형 필터링 적용"""
    if prev_points is None:
        return current_points
    
    filtered_points = []
    
    # 손가락의 자연스러운 움직임 유지를 위한 손가락별 처리
    finger_groups = [
        [0],                        # 손목
        [1, 2, 3, 4],               # 엄지
        [5, 6, 7, 8],               # 검지
        [9, 10, 11, 12],            # 중지
        [13, 14, 15, 16],           # 약지
        [17, 18, 19, 20]            # 소지
    ]
    
    for i, (curr, prev) in enumerate(zip(current_points, prev_points)):
        # 1. 기본 필터링: 현재 또는 이전 키포인트 없으면 있는 것 사용
        if not curr or not prev:
            filtered_points.append(curr or prev)
            continue
        
        # 현재 키포인트가 속한 손가락 찾기
        finger_idx = -1
        for f_idx, finger in enumerate(finger_groups):
            if i in finger:
                finger_idx = f_idx
                break
        
        # 2. 움직임 일관성 체크 (같은 손가락 내의 다른 관절들과 움직임 비교)
        if finger_idx > 0:  # 손목이 아닌 경우
            finger_points = finger_groups[finger_idx]
            finger_movements = []
            
            # 같은 손가락 내 다른 관절들의 평균 이동 방향
            for p_idx in finger_points:
                if p_idx != i and p_idx < len(current_points) and p_idx < len(prev_points):
                    if current_points[p_idx] and prev_points[p_idx]:
                        move_x = current_points[p_idx][0] - prev_points[p_idx][0]
                        move_y = current_points[p_idx][1] - prev_points[p_idx][1]
                        finger_movements.append((move_x, move_y))
            
            # 같은 손가락의 다른 관절들이 일관된 방향으로 움직인다면 해당 방향 적용
            if finger_movements:
                avg_move_x = sum(m[0] for m in finger_movements) / len(finger_movements)
                avg_move_y = sum(m[1] for m in finger_movements) / len(finger_movements)
                
                # 움직임 일관성 확인을 위한 각도 계산
                curr_move_x = curr[0] - prev[0]
                curr_move_y = curr[1] - prev[1]
                
                # 방향 벡터의 내적으로 일관성 체크
                consistency = True
                if abs(avg_move_x) > 1 or abs(avg_move_y) > 1:  # 충분한 움직임이 있는 경우
                    dot_product = curr_move_x * avg_move_x + curr_move_y * avg_move_y
                    magnitude1 = (curr_move_x**2 + curr_move_y**2)**0.5
                    magnitude2 = (avg_move_x**2 + avg_move_y**2)**0.5
                    
                    if magnitude1 > 0 and magnitude2 > 0:
                        cos_angle = dot_product / (magnitude1 * magnitude2)
                        # 방향이 크게 다르면 일관성 없음으로 판단
                        if cos_angle < 0.5:  # 약 60도 이상 차이
                            consistency = False
        
        # 3. 움직임 예측 (이전 두 프레임 기반, 가속도 고려)
        predicted_x, predicted_y = prev[0], prev[1]  # 기본값
        
        if prev_prev_points and i < len(prev_prev_points) and prev_prev_points[i]:
            # 이전 움직임을 기반으로 현재 위치 예측
            prev_move_x = prev[0] - prev_prev_points[i][0]
            prev_move_y = prev[1] - prev_prev_points[i][1]
            
            # 기본 예측 (일정 속도 가정)
            predicted_x = prev[0] + prev_move_x
            predicted_y = prev[1] + prev_move_y
        
        # 4. 적응형 필터링 (신뢰도 기반)
        distance = ((curr[0] - prev[0])**2 + (curr[1] - prev[1])**2)**0.5
        
        if distance > threshold:
            # 신뢰도 기반 가중 평균 계산 (현재값과 예측값 사이)
            # 손가락 끝(4,8,12,16,20)은 더 민감하게, 손목과 메타카팔은 더 안정적으로
            if i in [0, 1, 5, 9, 13, 17]:  # 손목 및 메타카팔
                alpha = 0.7  # 이전 값 가중치 높게 (안정성)
            elif i in [4, 8, 12, 16, 20]:   # 손가락 끝
                alpha = 0.3  # 현재 값 가중치 높게 (민감성)
            else:  # 중간 관절
                alpha = 0.5  # 중간 가중치
                
            # 예측값과 실제값 가중 평균
            filtered_x = int(alpha * predicted_x + (1-alpha) * curr[0])
            filtered_y = int(alpha * predicted_y + (1-alpha) * curr[1])
            filtered_points.append((filtered_x, filtered_y))
        else:
            # 움직임이 작으면 현재 값 사용
            filtered_points.append(curr)
    
    return filtered_points

def detect_finger_states(points):
    """손가락 접힘/펴짐 상태를 더 정확히 감지"""
    finger_states = [False, False, False, False, False]
    
    # 손목 좌표
    wrist = points[0]
    if not wrist:
        return finger_states
    
    # 각 손가락별 처리
    for i in range(5):
        # 마디 관절 정보
        meta = points[i*4 + 1]
        prox = points[i*4 + 2]
        mid = points[i*4 + 3]
        dist = points[i*4 + 4]
        
        # 손가락이 펴진 상태 감지 조건 강화
        if wrist and dist:
            # 손가락 끝과 손목 간 거리 계산
            dist_to_wrist = np.sqrt((dist[0]-wrist[0])**2 + (dist[1]-wrist[1])**2)
            
            # 손가락 시작점(메타카팔)과 손목 간 거리 계산
            if meta:
                meta_to_wrist = np.sqrt((meta[0]-wrist[0])**2 + (meta[1]-wrist[1])**2)
                
                # 엄지는 특별히 처리
                if i == 0:
                    # 엄지 벡터와 나머지 손가락들의 기본 방향에 대한 각도 계산
                    base_vec = np.array([meta[0] - wrist[0], meta[1] - wrist[1]])
                    tip_vec = np.array([dist[0] - wrist[0], dist[1] - wrist[1]])
                    
                    if np.linalg.norm(base_vec) > 0 and np.linalg.norm(tip_vec) > 0:
                        # 벡터 정규화
                        base_vec = base_vec / np.linalg.norm(base_vec)
                        tip_vec = tip_vec / np.linalg.norm(tip_vec)
                        
                        # 내적으로 각도 계산
                        dot_product = np.dot(base_vec, tip_vec)
                        # -1.0에서 1.0 범위로 제한
                        angle = np.arccos(np.clip(dot_product, -1.0, 1.0))
                        # 각도가 크고, 엄지 끝이 손목에서 충분히 멀면 펴진 상태
                        finger_states[i] = angle > np.pi/6 and dist_to_wrist > meta_to_wrist * 1.2
                else:
                    # 다른 손가락들: 끝마디가 기본 마디보다 1.3배 이상 멀고,
                    # 중간 관절이 구부러져있지 않으면 펴진 상태로 간주
                    is_extended = dist_to_wrist > meta_to_wrist * 1.3
                    
                    # 중간 관절 확인 (구부러졌는지 판단)
                    is_straight = True
                    if mid and prox:
                        # 두 마디 사이의 벡터
                        vec1 = np.array([prox[0] - meta[0], prox[1] - meta[1]])
                        vec2 = np.array([mid[0] - prox[0], mid[1] - prox[1]])
                        vec3 = np.array([dist[0] - mid[0], dist[1] - mid[1]])
                        
                        # 벡터의 길이가 0이 아니면 정규화
                        if np.linalg.norm(vec1) > 0 and np.linalg.norm(vec2) > 0:
                            vec1 = vec1 / np.linalg.norm(vec1)
                            vec2 = vec2 / np.linalg.norm(vec2)
                            # 벡터 간 각도
                            dot12 = np.clip(np.dot(vec1, vec2), -1.0, 1.0)
                            angle12 = np.arccos(dot12)
                            # 첫번째와 두번째 벡터 사이 각도가 크면 구부러짐
                            if angle12 > np.pi/4:
                                is_straight = False
                        
                        if np.linalg.norm(vec2) > 0 and np.linalg.norm(vec3) > 0:
                            vec2 = vec2 / np.linalg.norm(vec2)
                            vec3 = vec3 / np.linalg.norm(vec3)
                            # 벡터 간 각도
                            dot23 = np.clip(np.dot(vec2, vec3), -1.0, 1.0)
                            angle23 = np.arccos(dot23)
                            # 두번째와 세번째 벡터 사이 각도가 크면 구부러짐
                            if angle23 > np.pi/4:
                                is_straight = False
                    finger_states[i] = is_extended and is_straight
    
    return finger_states

def recognize_gesture(finger_states):
    """손가락 상태를 기반으로 제스처 인식"""
    # finger_states = [엄지, 검지, 중지, 약지, 소지]
   
    # 주먹
    if not any(finger_states):
        return "fist"
   
    # 보자기 (모든 손가락 펴짐)
    if all(finger_states):
        return "open_palm"
   
    # 검지 포인팅
    if finger_states == [False, True, False, False, False]:
        return "pointing"
   
    # V 사인
    if finger_states == [False, True, True, False, False]:
        return "v_sign"
   
    # 엄지척
    if finger_states == [True, False, False, False, False]:
        return "thumbs_up"
   
    # 전화 제스처
    if finger_states == [True, False, False, False, True]:
        return "call_me"
   
    # 락 사인
    if finger_states == [True, False, False, False, True] and not any(finger_states[1:4]):
        return "rock_sign"
   
    return "unknown"

def hand(crop, elbow_pos=None, wrist_pos=None, prev_points=None, prev_confidences=None, prev_prev_points=None):
    """개선된 손 키포인트 감지 함수"""
    inWidth = 762   
    inHeight = 762
    threshold = 0.09
    
    # 원본 이미지 저장
    original_crop = crop.copy()
    cropHeight, cropWidth = crop.shape[:2]
    
    # 1. 전처리: 이미지 선명도 향상
    sharpened = crop.copy()
    blur = cv.GaussianBlur(crop, (0, 0), 3)
    sharpened = cv.addWeighted(crop, 1.5, blur, -0.5, 0)
    
    # 2. 피부색 마스크 정교화
    hsv_crop = cv.cvtColor(sharpened, cv.COLOR_BGR2HSV)
    ycrcb_crop = cv.cvtColor(sharpened, cv.COLOR_BGR2YCrCb)
    
    # HSV 색공간에서 피부색 마스크
    lower_hsv = np.array([0, 20, 70], dtype=np.uint8)
    upper_hsv = np.array([20, 255, 255], dtype=np.uint8)
    skin_mask_hsv = cv.inRange(hsv_crop, lower_hsv, upper_hsv)
    
    # 넓은 범위의 피부색도 포함 (붉은색 계열)
    lower_hsv2 = np.array([170, 20, 70], dtype=np.uint8)
    upper_hsv2 = np.array([180, 255, 255], dtype=np.uint8)
    skin_mask_hsv2 = cv.inRange(hsv_crop, lower_hsv2, upper_hsv2)
    
    # YCrCb 색공간에서 피부색 마스크
    lower_ycrcb = np.array([0, 135, 85], dtype=np.uint8)
    upper_ycrcb = np.array([255, 180, 135], dtype=np.uint8)
    skin_mask_ycrcb = cv.inRange(ycrcb_crop, lower_ycrcb, upper_ycrcb)
    
    # 마스크 결합
    skin_mask = cv.bitwise_or(skin_mask_hsv, skin_mask_hsv2)
    skin_mask = cv.bitwise_and(skin_mask, skin_mask_ycrcb)
    
    # 마스크 정제
    kernel = np.ones((5, 5), np.uint8)
    skin_mask = cv.morphologyEx(skin_mask, cv.MORPH_CLOSE, kernel, iterations=2)
    skin_mask = cv.GaussianBlur(skin_mask, (5, 5), 0)
    _, skin_mask = cv.threshold(skin_mask, 127, 255, cv.THRESH_BINARY)
    
    # 3. 손 윤곽선 검출 개선
    contours, _ = cv.findContours(skin_mask, cv.RETR_EXTERNAL, cv.CHAIN_APPROX_SIMPLE)
    hand_contour = None
    hand_mask = np.zeros((cropHeight, cropWidth), dtype=np.uint8)
    
    if contours:
        # 일정 크기 이상의 윤곽선만 고려
        valid_contours = [cnt for cnt in contours if cv.contourArea(cnt) > cropWidth*cropHeight*0.05]
        if valid_contours:
            hand_contour = max(valid_contours, key=cv.contourArea)
            
            # 손 모양 윤곽선 단순화
            epsilon = 0.005 * cv.arcLength(hand_contour, True)
            hand_contour = cv.approxPolyDP(hand_contour, epsilon, True)
            
            # 윤곽선 기반 마스크 생성 및 확장
            cv.drawContours(hand_mask, [hand_contour], -1, 255, -1)
            hand_mask = cv.dilate(hand_mask, kernel, iterations=2)
    
    # 4. 손 영역 내 대비 강화
    masked_hand = cv.bitwise_and(sharpened, sharpened, mask=hand_mask)
    
    # 특히 손가락 마디 부분 강화를 위해 Gabor 필터 적용
    ksize = 15
    sigma = 3.0
    theta = np.pi/4
    lambd = 10.0
    gamma = 0.5
    
    g_kernel = cv.getGaborKernel((ksize, ksize), sigma, theta, lambd, gamma, 0, ktype=cv.CV_32F)
    filtered = cv.filter2D(masked_hand, -1, g_kernel)
    
    # 최종 처리된 입력 이미지
    final_input = cv.addWeighted(masked_hand, 0.7, filtered, 0.3, 0)
    
    # 실제 네트워크 입력으로 사용
    inpBlob = cv.dnn.blobFromImage(final_input, 1.0/255, (inWidth, inHeight), (0, 0, 0), swapRB=False, crop=False)
    handNet.setInput(inpBlob)
    output = handNet.forward()
   
    detected_points = 0
    points = []
    confidences = []
    
    # 5. 키포인트 검출 및 제약 적용
    for i in range(21):
        probMap = output[0, i, :, :]
        minVal, prob, minLoc, point = cv.minMaxLoc(probMap)
       
        x = int((cropWidth * point[0]) / output.shape[3])
        y = int((cropHeight * point[1]) / output.shape[2])
        
        inside_contour = True
        if hand_contour is not None and prob > threshold:
            inside_contour = cv.pointPolygonTest(hand_contour, (x, y), False) >= 0
        
        if prob > threshold and inside_contour:
            points.append((x, y))
            confidences.append(float(prob))
            detected_points += 1
        else:
            points.append(None)
            confidences.append(0.0)
    
    # 6. 해부학적 손가락 구조 모델 기반 키포인트 보강
    # 손가락 길이 비율 모델 (메타카팔 기준)
    # [엄지, 검지, 중지, 약지, 소지]에 대한 관절 길이 비율
    finger_length_ratio = {
        "proximal": [0.5, 0.65, 0.7, 0.65, 0.5],  # 첫 번째 관절
        "middle": [0.45, 0.4, 0.45, 0.4, 0.35],   # 두 번째 관절
        "distal": [0.4, 0.3, 0.3, 0.3, 0.3]       # 세 번째 관절
    }
    
    # 손목 키포인트가 있으면 기초 구조 설정
    if points[0] is not None:
        for finger in range(5):
            # 기본 메타카팔 관절 (첫 번째 마디)
            meta_idx = finger*4 + 1
            if points[meta_idx] is not None:
                # 이후 관절들 처리
                for joint, ratio_key in enumerate(["proximal", "middle", "distal"]):
                    joint_idx = meta_idx + joint + 1
                    prev_idx = meta_idx + joint
                    
                    # 이전 관절이 있고 현재 관절이 없으면 예측
                    if points[prev_idx] is not None and points[joint_idx] is None:
                        # 방향 벡터 계산 로직 (기존 코드 유지)
                        if finger == 0:  # 엄지는 방향이 다름
                            dir_vec = np.array(points[prev_idx]) - np.array(points[0])
                        else:
                            # 다른 손가락은 손바닥 중심으로부터의 방향 사용
                            palm_center = np.mean([p for p in [points[1], points[5], points[9], points[13], points[17]] if p is not None], axis=0)
                            dir_vec = np.array(points[prev_idx]) - palm_center
                        
                        if np.linalg.norm(dir_vec) > 0:
                            dir_vec = dir_vec / np.linalg.norm(dir_vec)
                            
                            # 이전 관절과 그 이전 관절 사이의 길이 계산 (예외 처리 추가)
                            if prev_idx > 0 and points[prev_idx-1] is not None:
                                length = np.linalg.norm(np.array(points[prev_idx]) - np.array(points[prev_idx-1])) * finger_length_ratio[ratio_key][finger]
                            else:
                                # 이전 관절이 없으면 기본값 사용
                                length = 20 * finger_length_ratio[ratio_key][finger]
                                
                            new_point = np.array(points[prev_idx]) + dir_vec * length
                            new_x, new_y = int(new_point[0]), int(new_point[1])
                            
                            # 윤곽선 내부 확인
                            if hand_contour is None or cv.pointPolygonTest(hand_contour, (new_x, new_y), False) >= 0:
                                points[joint_idx] = (new_x, new_y)
                                confidences[joint_idx] = confidences[prev_idx] * 0.8
                                detected_points += 1
    
    # 7. 불완전한 키포인트 보간
    points = interpolate_missing_keypoints(points)
   
    # 8. 해부학적 제약 적용
    points = apply_anatomical_constraints(points)
   
    # 9. 이전 프레임 데이터로 필터링
    if prev_points:
        if prev_prev_points:  # 이전 두 프레임이 모두 있는 경우
            points = filter_unstable_keypoints(points, prev_points, prev_prev_points)
        else:  # 이전 한 프레임만 있는 경우
            points = filter_unstable_keypoints(points, prev_points)
    
    # 10. 손가락 상태 감지
    finger_states = detect_finger_states(points)
   
    # 11. 제스처 인식
    gesture = recognize_gesture(finger_states)
    
    # 12. 템플릿 적용 (필요한 경우)
    if detected_points < 10 or points[0] is None:
        # 기존 템플릿 코드...
        
        # 템플릿 적용 후 윤곽선 내부로 제한
        if hand_contour is not None:
            for i in range(len(points)):
                if points[i]:
                    x, y = points[i]
                    if cv.pointPolygonTest(hand_contour, (x, y), False) < 0:
                        # 윤곽선 외부 점 처리
                        points[i] = None
                        confidences[i] = 0.0
    
    # 13. 디버깅용 시각화
    debug_image = original_crop.copy()
    
    # 윤곽선 표시
    if hand_contour is not None:
        cv.drawContours(debug_image, [hand_contour], -1, (0, 255, 0), 2)
    
    # 키포인트와 연결선 표시
    for i, point in enumerate(points):
        if point:
            cv.circle(debug_image, point, 5, (0, 0, 255), -1)
            cv.putText(debug_image, str(i), (point[0]-5, point[1]-5), cv.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 0), 1)
    
    # 손가락 연결 표시
    for finger in range(5):
        base = finger*4 + 1
        for i in range(3):
            if points[base+i] and points[base+i+1]:
                cv.line(debug_image, points[base+i], points[base+i+1], (0, 255, 255), 2)
        
        # 손가락 루트와 손목 연결
        if points[base] and points[0]:
            cv.line(debug_image, points[base], points[0], (255, 0, 255), 2)
    
    return points, confidences, gesture, hand_contour, debug_image
def face(crop):
    """얼굴 키포인트 감지 함수"""
    inWidth = 656
    inHeight = 656
    threshold = 0.1
   
    inpBlob = cv.dnn.blobFromImage(crop, 1.0/255, (inWidth, inHeight), (0, 0, 0), swapRB=False, crop=False)
   
    faceNet.setInput(inpBlob)
    output = faceNet.forward()
   
    cropHeight, cropWidth = crop.shape[:2]
    points = []
    confidences = []
   
    for i in range(len(FACE_PARTS)):
        probMap = output[0, i, :, :]
        minVal, prob, minLoc, point = cv.minMaxLoc(probMap)
       
        x = int((cropWidth * point[0]) / output.shape[3])
        y = int((cropHeight * point[1]) / output.shape[2])
       
        if prob > threshold:
            points.append((x, y))
            confidences.append(float(prob))
        else:
            points.append(None)
            confidences.append(0.0)
   
    while len(points) < 70:
        points.append(None)
        confidences.append(0.0)
       
    return points, confidences

def video_to_keypoints(cap, json_dir, filename):
    """비디오에서 키포인트 추출 및 제스처 인식 후 JSON 저장"""
    frames_data = []
    frame_count = 0
    processed_count = 0
   
    base_filename = os.path.basename(filename)
    json_filename = os.path.splitext(base_filename)[0] + '.json'
    json_path = f"{json_dir}/{json_filename}".replace('\\', '/')
   
    print(f'키포인트 추출 시작')
   
    # 이전 프레임 데이터 저장용
    prev_hand_points = None
    prev_hand_confidences = None
    
    # 왼손과 오른손의 이전 포인트를 별도로 관리
    prev_left_hand_points = None
    prev_right_hand_points = None
    prev_left_hand_confidences = None
    prev_right_hand_confidences = None

    # 이전 이전 프레임 데이터 저장용
    prev_prev_left_hand_points = None
    prev_prev_right_hand_points = None
   
    # 총 프레임 수 계산
    total_frames = int(cap.get(cv.CAP_PROP_FRAME_COUNT))
   
    # 결과 표시 창 생성
    # cv.namedWindow('Hand Detection', cv.WINDOW_NORMAL)
    # cv.resizeWindow('Hand Detection', 1280, 720)
   
    # 영상 저장용 VideoWriter 추가
    fourcc = cv2.VideoWriter_fourcc(*'XVID')
    out = None

    while True:
        ret, frame = cap.read()
        if not ret:
            break
           
        frame_count += 1
       
        # 표시용 프레임 복사
        display_frame = frame.copy()
        
        # 처리 속도를 위해 프레임 크기 조정
        height, width = frame.shape[:2]
       
        # 신체 키포인트 감지
        body_points, body_confidences = body(frame)
       
        # 프레임별 결과 저장용 딕셔너리
        frame_data = {
            "frame": frame_count,
            "body": {
                "points": [],
                "confidences": []
            },
            "hands": {
                "left": {
                    "points": [],
                    "confidences": [],
                    "gesture": "none"
                },
                "right": {
                    "points": [],
                    "confidences": [],
                    "gesture": "none"
                }
            },
            "face": {
                "points": [],
                "confidences": []
            }
        }
       
        # 신체 키포인트 그리기
        for pair in POSE_PAIRS:
            partA = BODY_PARTS[pair[0]]
            partB = BODY_PARTS[pair[1]]
            
            if body_points[partA] and body_points[partB]:
                cv.line(display_frame, body_points[partA], body_points[partB], (0, 255, 0), 2)
                cv.circle(display_frame, body_points[partA], 5, (0, 0, 255), -1)
                cv.circle(display_frame, body_points[partB], 5, (0, 0, 255), -1)
        
        # 신체 키포인트 결과 저장
        for i, (point, conf) in enumerate(zip(body_points, body_confidences)):
            if point:
                # 좌표를 0-1 범위로 정규화
                norm_x = point[0] / width
                norm_y = point[1] / height
                frame_data["body"]["points"].append([norm_x, norm_y])
            else:
                frame_data["body"]["points"].append(None)
           
            frame_data["body"]["confidences"].append(conf)
       
        # 손 처리 (오른손, 왼손 모두)
        for hand_type in ["right", "left"]:
            # 오른손/왼손 구분
            if hand_type == "right" and body_points[BODY_PARTS["RWrist"]] and body_points[BODY_PARTS["RElbow"]]:
                wrist_pos = body_points[BODY_PARTS["RWrist"]]
                elbow_pos = body_points[BODY_PARTS["RElbow"]]
                prev_points = prev_right_hand_points
                prev_confs = prev_right_hand_confidences
                color = (255, 0, 0)  # 오른손은 파란색
            elif hand_type == "left" and body_points[BODY_PARTS["LWrist"]] and body_points[BODY_PARTS["LElbow"]]:
                wrist_pos = body_points[BODY_PARTS["LWrist"]]
                elbow_pos = body_points[BODY_PARTS["LElbow"]]
                prev_points = prev_left_hand_points
                prev_confs = prev_left_hand_confidences
                color = (0, 0, 255)  # 왼손은 빨간색
            else:
                continue
               
            # 손목 위치에 따라 손 영역 크롭
            if wrist_pos:
                # 손 영역 표시
                hand_box_size = HAND_CROP_SIZE//2
                x1 = max(0, wrist_pos[0] - hand_box_size)
                y1 = max(0, wrist_pos[1] - hand_box_size)
                x2 = min(width, wrist_pos[0] + hand_box_size)
                y2 = min(height, wrist_pos[1] + hand_box_size)
                
                # 손 영역 사각형 표시
                cv.rectangle(display_frame, (x1, y1), (x2, y2), color, 2)
                
                # 손 영역 크롭
                hand_crop = frame[y1:y2, x1:x2]
                if hand_crop.size == 0:
                    continue
                
                # 피부색 마스크 생성 (디버깅 용도)
                hsv_crop = cv.cvtColor(hand_crop, cv.COLOR_BGR2HSV)
                lower = np.array([0, 30, 50], dtype=np.uint8)
                upper = np.array([25, 255, 255], dtype=np.uint8)
                skinMask = cv.inRange(hsv_crop, lower, upper)
                skinMask = cv.erode(skinMask, None, iterations=1)
                skinMask = cv.dilate(skinMask, None, iterations=3)
                skinMask = cv.medianBlur(skinMask, 5)
               
                # 손 실루엣 추출
                hand_silhouette = np.zeros_like(hand_crop)
                hand_silhouette[skinMask > 0] = (0, 255, 0)  # 손 영역을 초록색으로 표시

                # 원본 이미지와 합성
                hand_overlay = hand_crop.copy()
                cv.addWeighted(hand_silhouette, 0.5, hand_overlay, 0.5, 0, hand_overlay)

                # 디버깅을 위해 실루엣 마스크 표시
                if hand_type == "right":
                    display_frame[10:10+hand_box_size, 10:10+hand_box_size] = cv.resize(hand_overlay, (hand_box_size, hand_box_size))
                else:
                    display_frame[10:10+hand_box_size, width-hand_box_size-10:width-10] = cv.resize(hand_overlay, (hand_box_size, hand_box_size))

                # # 또는 원본 크롭 영역에 손 실루엣을 오버레이할 수도 있음
                # hand_overlay_full = frame[y1:y2, x1:x2].copy()
                # hand_silhouette_full = np.zeros_like(hand_overlay_full)
                # hand_silhouette_full[skinMask > 0] = (0, 255, 0)  # 손 영역을 초록색으로 표시
                # cv.addWeighted(hand_silhouette_full, 0.3, hand_overlay_full, 0.7, 0, hand_overlay_full)
                # display_frame[y1:y2, x1:x2] = hand_overlay_full
               
                # 손 윤곽선 추출
                contours, _ = cv.findContours(skinMask, cv.RETR_EXTERNAL, cv.CHAIN_APPROX_SIMPLE)

                # 가장 큰 윤곽선 찾기 (손일 가능성이 높음)
                if contours:
                    max_contour = max(contours, key=cv.contourArea)
                    
                    # 윤곽선 그리기
                    hand_contour_img = np.zeros_like(hand_crop)
                    cv.drawContours(hand_contour_img, [max_contour], -1, (0, 255, 0), 1)
                    
                    # 디버깅을 위해 윤곽선 이미지 표시
                    if hand_type == "right":
                        display_frame[10:10+hand_box_size, 10:10+hand_box_size] = cv.resize(hand_contour_img, (hand_box_size, hand_box_size))
                    else:
                        display_frame[10:10+hand_box_size, width-hand_box_size-10:width-10] = cv.resize(hand_contour_img, (hand_box_size, hand_box_size))
                    
                    # 또는 원본 프레임에 윤곽선 오버레이
                    cv.drawContours(display_frame, [np.array([[p[0][0]+x1, p[0][1]+y1] for p in max_contour])], -1, color, 2)
               
                # 손 키포인트 감지
                prev_prev_points = None
                if hand_type == "right":
                    prev_prev_points = prev_prev_right_hand_points
                elif hand_type == "left":
                    prev_prev_points = prev_prev_left_hand_points

                hand_points, hand_confidences, gesture, detected_contour, debug_image = hand(
                    hand_crop,
                    elbow_pos,
                    wrist_pos,
                    prev_points,
                    prev_confs,
                    prev_prev_points
                )
               
                # 디버그 이미지 표시 (추가)
                if debug_image is not None:
                    debug_resize = cv.resize(debug_image, (hand_box_size, hand_box_size))
                    
                    # 디버그 이미지 위치 지정
                    if hand_type == "right":
                        display_frame[height-hand_box_size-10:height-10, 10:10+hand_box_size] = debug_resize
                    else:
                        display_frame[height-hand_box_size-10:height-10, width-hand_box_size-10:width-10] = debug_resize

                # 이전 프레임 데이터 업데이트
                if hand_type == "right":
                    prev_prev_right_hand_points = prev_right_hand_points
                    prev_right_hand_points = hand_points
                    prev_right_hand_confidences = hand_confidences
                else:
                    prev_prev_left_hand_points = prev_left_hand_points
                    prev_left_hand_points = hand_points
                    prev_left_hand_confidences = hand_confidences
               
                # 손 키포인트 그리기
                for pair in HAND_PAIRS:
                    partA = HAND_PARTS[pair[0]]
                    partB = HAND_PARTS[pair[1]]
                    
                    if hand_points[partA] and hand_points[partB]:
                        # 크롭 좌표를 원본 프레임 좌표로 변환
                        ptA = (int(hand_points[partA][0]) + x1, int(hand_points[partA][1]) + y1)
                        ptB = (int(hand_points[partB][0]) + x1, int(hand_points[partB][1]) + y1)
                        cv.line(display_frame, ptA, ptB, color, 2)
                        cv.circle(display_frame, ptA, 3, color, -1)
                        cv.circle(display_frame, ptB, 3, color, -1)
                
                # 제스처 표시
                gesture_text = f"{hand_type}: {gesture}"
                text_pos = (x1, y1 - 10) if y1 > 30 else (x1, y2 + 20)
                cv.putText(display_frame, gesture_text, text_pos, cv.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)
               
                # 손 키포인트 결과 저장 (원본 이미지 좌표계로 변환)
                for i, (point, conf) in enumerate(zip(hand_points, hand_confidences)):
                    if point:
                        # 크롭 이미지 내 좌표를 원본 이미지 내 좌표로 변환
                        orig_x = x1 + point[0]
                        orig_y = y1 + point[1]
                        # 좌표를 0-1 범위로 정규화
                        norm_x = orig_x / width
                        norm_y = orig_y / height
                        frame_data["hands"][hand_type]["points"].append([norm_x, norm_y])
                    else:
                        frame_data["hands"][hand_type]["points"].append(None)
                   
                    frame_data["hands"][hand_type]["confidences"].append(conf)
               
                # 제스처 정보 저장
                frame_data["hands"][hand_type]["gesture"] = gesture
                
                # 디버깅을 위해 피부 마스크 표시 (선택적)
                if skinMask is not None and skinMask.size > 0:
                    skin_display = cv2.resize(skinMask, (hand_box_size, hand_box_size))
                    skin_display = cv2.cvtColor(skin_display, cv2.COLOR_GRAY2BGR)
                    if hand_type == "right":
                        display_frame[10:10+hand_box_size, 10:10+hand_box_size] = skin_display
                    else:
                        display_frame[10:10+hand_box_size, width-hand_box_size-10:width-10] = skin_display
       
        # 얼굴 처리는 생략 (필요시 추가)
       
        # 프레임 데이터 저장
        frames_data.append(frame_data)
        processed_count += 1
       
        # 영상 표시
        #cv.imshow('Hand Detection', display_frame)
        
        # 프레임 번호와 진행률 표시
        progress_text = f"Frame: {frame_count}/{total_frames} ({(frame_count/total_frames)*100:.1f}%)"
        cv.putText(display_frame, progress_text, (10, height-20), cv.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
       
        # 키 입력 대기 - 'q' 키를 누르면 종료
        # key = cv.waitKey(1) & 0xFF
        # if key == ord('q'):
        #     break
       
        # 진행 상황 표시
        if frame_count % 10 == 0:
            progress = (frame_count / total_frames) * 100
            print(f"진행 상황: {progress:.1f}% ({frame_count}/{total_frames})")
   
    if out is not None:
        out.release()
    # 결과 JSON 저장
    result = {
        "filename": base_filename,
        "width": width,
        "height": height,
        "frames": frames_data
    }
   
    with open(json_path, 'w', encoding='utf-8') as f:
        json.dump(result, f, ensure_ascii=False, indent=4)
   
    # 창 닫기
    # cv.destroyAllWindows()
   
    print(f"처리 완료: {processed_count}/{frame_count} 프레임, 저장 경로: {json_path}")
    return json_path

# def main():
#     """메인 함수: 비디오 키포인트 추출 실행"""
#     # 매개변수 설정
#     video_dir = "C:/Users/DS/Desktop/kimsihyun/my/videos"
#     json_dir = "C:/Users/DS/Desktop/kimsihyun/my/keypoints"
   
#     # 폴더가 없으면 생성
#     os.makedirs(json_dir, exist_ok=True)
   
#     # 비디오 파일 목록
#     video_files = [f for f in os.listdir(video_dir) if f.endswith(('.mp4', '.avi', '.mov'))]
   
#     for video_file in video_files:
#         video_path = f"{video_dir}/{video_file}".replace('\\', '/')
       
#         # 비디오 열기
#         cap = cv.VideoCapture(video_path)
#         if not cap.isOpened():
#             print(f"Error: '{video_path}' 파일을 열 수 없습니다.")
#             continue
       
#         # 키포인트 추출 및 JSON 저장
#         start_time = time.time()
#         json_path = video_to_keypoints(cap, json_dir, video_file)
       
#         # 자원 해제
#         cap.release()
       
#         elapsed_time = time.time() - start_time
#         print(f"처리 시간: {elapsed_time:.2f}초")
#         print(f"{video_file} 처리 완료, JSON 저장 경로: {json_path}")

   
if __name__ == "__main__":
    video_path = "C:/Users/DS/Desktop/kimsihyun/my/datas/video/KETI_SL_0000002911.avi"
    json_dir = "C:/Users/DS/Desktop/kimsihyun/my/datas/keypoints"
    
    # 폴더가 없으면 생성
    os.makedirs(json_dir, exist_ok=True)
    
    filename = video_path
    cap = cv.VideoCapture(video_path)
    
    if cap.isOpened():
        json_path = video_to_keypoints(cap, json_dir, filename)
        cap.release()
        print(f"처리 완료: JSON 저장 경로: {json_path}")
    else:
        print(f"Error: '{video_path}' 파일을 열 수 없습니다.")