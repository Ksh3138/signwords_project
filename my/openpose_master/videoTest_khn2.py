import cv2
import cv2 as cv
import json
from datetime import datetime
import os
import urllib.parse
import numpy as np

# 검출할 키포인트와 연결 선언
BODY_PARTS = {  
                "Nose": 0, "Neck": 1, 
                "RShoulder": 2, "RElbow": 3, "RWrist": 4, "LShoulder": 5, "LElbow": 6, "LWrist": 7,
                "MidHip": 8,
                "RHip": 9, "RKnee": 10, "RAnkle": 11, "LHip": 12, "LKnee": 13, "LAnkle": 14,
                "REye": 15, "LEye": 16, "REar": 17, "LEar": 18,
                "LBigToe": 19, "LSmallToe": 20, "LHeel": 21, "RBigToe": 22, "RSmallToe": 23, "RHeel": 24
            }
BODY_PAIRS = [
        ["Nose", "Neck"],
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
                ["LittleFingerProximal", "LittleFingerMiddle"], ["LittleFingerMiddle", "LittleFingerDistal"]]

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


# 모델 구조와 가중치를 불러와서 네트워크 구성
bodyProtoFile = "C:/Users/DS/Desktop/kimsihyun/my/openpose_master/models/pose/body_25/pose_deploy.prototxt"
bodyWeightsFile = "C:/Users/DS/Desktop/kimsihyun/my/openpose_master/models/pose/body_25/pose_iter_584000.caffemodel"
bodyNet = cv2.dnn.readNetFromCaffe(bodyProtoFile, bodyWeightsFile)

handProtoFile = "C:/Users/DS/Desktop/kimsihyun/my/openpose_master/models/hand/pose_deploy.prototxt"
handWeightsFile = "C:/Users/DS/Desktop/kimsihyun/my/openpose_master/models/hand/pose_iter_102000.caffemodel"
handNet = cv.dnn.readNetFromCaffe(handProtoFile, handWeightsFile)

faceProtoFile = "C:/Users/DS/Desktop/kimsihyun/my/openpose_master/models/face/pose_deploy.prototxt"
faceWeightsFile = "C:/Users/DS/Desktop/kimsihyun/my/openpose_master/models/face/pose_iter_116000.caffemodel"
faceNet = cv.dnn.readNetFromCaffe(faceProtoFile, faceWeightsFile)

# body25 키포인트 예측
def body(frame, prev_points):
    # openpose에 입력하기 위해 이미지 크기 변경
    ##############################
    inWidth = 656 # 해상도 높이기위해서 수정함.
    ##############################
    inHeight = 656
    threshold = 0.09
    
    # 전처리 - 정규화, 크기 조정 등
    inpBlob = cv2.dnn.blobFromImage(frame, 1.0 / 255, (inWidth, inHeight), (0, 0, 0), swapRB=False, crop=False)

    # 네트워크에 이미지를 넣고 예측
    bodyNet.setInput(inpBlob)
    output = bodyNet.forward()

    frameHeight, frameWidth, _ = frame.shape
    points = [] 
    confidences = [] 
    for i in range(len(BODY_PARTS)): # body의 모든 키포인트에 대해
        # 예측 결과 가장 확률이 높은 좌표로 결정
        probMap = output[0, i, :, :]
        minVal, prob, minLoc, point = cv2.minMaxLoc(probMap)

        # 원본 이미지 크기에 맞게 좌표 바꾸기
        x = int((frameWidth * point[0]) / output.shape[3])
        y = int((frameHeight * point[1]) / output.shape[2])

        # 신뢰도가 0.1 이상이면 저장, 아니면 검출실패
        if prob > threshold:
            points.append((x,y)) #좌표 저장
            confidences.append(float(prob))  #신뢰도 저장
        else:
            points.append(None)
            confidences.append(0.0)

    lw_idx = BODY_PARTS["LWrist"]
    rw_idx = BODY_PARTS["RWrist"]

    if prev_points is not None and points[lw_idx] is not None and prev_points[lw_idx] is not None:
        if np.linalg.norm(np.array(points[lw_idx]) - np.array(prev_points[lw_idx])) >frameHeight/4:
            points[lw_idx] = prev_points[lw_idx]
    if prev_points is not None and points[rw_idx] is not None and prev_points[rw_idx] is not None:
        if np.linalg.norm(np.array(points[rw_idx]) - np.array(prev_points[rw_idx])) >frameHeight/4:
            points[rw_idx] = prev_points[rw_idx]

    if points[lw_idx] is not None and points[rw_idx] is not None:
        if np.linalg.norm(np.array(points[lw_idx]) - np.array(points[rw_idx]))<frameHeight/10:
            points[lw_idx] = None
            points[rw_idx] = None 

    # if prev_points is not None:
    #     # 검출 못했으면 이전 좌표로 대체
    #     if points[lw_idx] is None and prev_points[lw_idx] is not None:
    #         points[lw_idx] = prev_points[lw_idx]
    #     if points[rw_idx] is None and prev_points[rw_idx] is not None:
    #         points[rw_idx] = prev_points[rw_idx]

    #     # 거리 계산 전에 None 체크
    #     if points[lw_idx] is not None and points[ls_idx] is not None and prev_points[lw_idx] is not None and prev_points[ls_idx] is not None:
    #         cur_dist = np.linalg.norm(np.array(points[lw_idx]) - np.array(points[ls_idx]))
    #         prev_dist = np.linalg.norm(np.array(prev_points[lw_idx]) - np.array(prev_points[ls_idx]))
    #         if prev_dist * 0.7 < cur_dist < prev_dist * 0.8:
    #             points[lw_idx] = prev_points[lw_idx]

    #     if points[rw_idx] is not None and points[rs_idx] is not None and prev_points[rw_idx] is not None and prev_points[rs_idx] is not None:
    #         cur_dist = np.linalg.norm(np.array(points[rw_idx]) - np.array(points[rs_idx]))
    #         prev_dist = np.linalg.norm(np.array(prev_points[rw_idx]) - np.array(prev_points[rs_idx]))
    #         if prev_dist * 0.7 < cur_dist < prev_dist * 0.8:
    #             points[rw_idx] = prev_points[rw_idx]


    return points, confidences

# hand 키포인트 예측 (과정은 body와 동일, handNet 사용)

def hand(crop, wrist, prev_points=None):
    inWidth = 656
    inHeight = 656
    threshold=0.12

    kernel = np.array([[0, -1, 0],
                       [-1, 5, -1],
                       [0, -1, 0]])
    filtered_crop = cv2.filter2D(crop, -1, kernel)

    inpBlob = cv2.dnn.blobFromImage(filtered_crop, 1.0/255, (inWidth, inHeight), 
                                   (0, 0, 0), swapRB=False, crop=False)
    handNet.setInput(inpBlob)
    output = handNet.forward()

    cropHeight, cropWidth = crop.shape[:2]
    points = [None] * 21
    confidences = [0.0] * 21

    # 1. tip만 딥러닝으로 검출
    tip_indices = [4, 8, 12, 16, 20]
    finger_joints = {
        4: [1, 2, 3],     # Thumb: MCP, PIP, DIP
        8: [5, 6, 7],     # Index
        12: [9, 10, 11],  # Middle  
        16: [13, 14, 15], # Ring
        20: [17, 18, 19]  # Little
    }
    points[0] = wrist
    confidences[0] = 1.0

    for tip_idx in tip_indices:
        probMap = output[0, tip_idx, :, :]
        minVal, prob, minLoc, point = cv2.minMaxLoc(probMap)
        x = int((cropWidth * point[0]) / output.shape[3])
        y = int((cropHeight * point[1]) / output.shape[2])

        if prob > threshold:
            tip = np.array([x, y])
            wrist_np = np.array(wrist)
            vec = tip - wrist_np
            joints = finger_joints[tip_idx]
            ratios = [0.4, 0.6, 0.8]
            for i, ratio in enumerate(ratios):
                joint_pos = wrist_np + ratio * vec
                points[joints[i]] = (int(joint_pos[0]), int(joint_pos[1]))
                confidences[joints[i]] = prob * (0.8 - i * 0.1)
            points[tip_idx] = (x, y)
            confidences[tip_idx] = float(prob)
        elif prev_points is not None:
            joints = finger_joints[tip_idx]
            for i in range(3):
                points[joints[i]] = prev_points[joints[i]]
            points[tip_idx] = prev_points[tip_idx]

    return points, confidences













# face 키포인트 예측 (과정은 body와 동일, faceNet 사용)
def face(crop):
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


# 영상에서 좌표정보를 추출해 프레임별로 시퀀스벡터 저장
def video_to_keypoints(cap, json_dir, filename):
    # 결과를 저장할 리스트
    frames_data = []
    frame_count = 0
    processed_count = 0
    # 결과를 저장할 json 파일 경로/이름
    base_filename = os.path.basename(filename)
    json_filename = os.path.splitext(base_filename)[0] + '.json'
    json_path = f"{json_dir}/{json_filename}".replace('\\', '/')

    print(f'키포인트 추출 시작')

    prev_points = None

    # 왼손과 오른손의 이전 포인트를 별도로 관리
    prev_left_hand_points = None
    prev_right_hand_points = None
    prev_bounding_contours = None
    prev_crop_list=None

    # 피부색 범위 지정 
    lower = np.array([0, 30, 50], dtype=np.uint8)
    upper = np.array([25, 255, 255], dtype=np.uint8)
        
    # 변수 미리 초기화 (오류 방지)
    hand_left_points = [None] * len(HAND_PARTS)
    hand_left_confidences = [0.0] * len(HAND_PARTS)
    hand_right_points = [None] * len(HAND_PARTS)
    hand_right_confidences = [0.0] * len(HAND_PARTS)
    x0, y0, x2, y2 = 0, 0, 0, 0

    while cv.waitKey(1) < 0:
        # 프레임 읽기
        hasFrame, frame = cap.read()
        if not hasFrame:
            break

        frame_data = {
            "frame_id": frame_count,
            "body_keypoints_2d": [],
            "face_keypoints_2d": [],
            "hand_left_keypoints_2d": [],
            "hand_right_keypoints_2d": []
        }

        try:
            # BGR → HSV 변환
            hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
        except Exception as e:
            print(f"프레임 처리 중 오류 발생: {str(e)}")
            break

        # 피부색 마스크 생성
        skinMask = cv2.inRange(hsv, lower, upper)

        # 노이즈 제거 (선택)
        skinMask = cv2.erode(skinMask, None, iterations=2)
        skinMask = cv2.dilate(skinMask, None, iterations=2)
        skinMask = cv2.GaussianBlur(skinMask, (3, 3), 0)

        # 컨투어 검출
        contours, _ = cv2.findContours(skinMask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        # 프레임 크기 구하기
        h_frame, w_frame = frame.shape[:2]
        frame_area = h_frame * w_frame

        bounding_contours = []
        for cnt in contours:
            x, y, w, h = cv2.boundingRect(cnt)
            # 전체 프레임의 0.5% 미만인 경우 제외
            if w * h < frame_area * 0.005:
                continue
            bounding_contours.append((x, y, w, h, cnt))

        crop_list = []
        for x, y, w, h, cnt in bounding_contours:
            crop = frame[y:y+h, x:x+w]
            crop_list.append(crop)

        if len(bounding_contours) == 3:
            # 얼굴(위쪽, y가 가장 작은 것) 인덱스 찾기
            face_idx = min(range(3), key=lambda i: bounding_contours[i][1])
            # 나머지 두 개의 인덱스
            hand_indices = [i for i in range(3) if i != face_idx]
            # 왼손/오른손 결정 (x좌표 오름차순: 작은게 왼손)
            if bounding_contours[hand_indices[0]][0] < bounding_contours[hand_indices[1]][0]:
                left_idx, right_idx = hand_indices
            else:
                right_idx, left_idx = hand_indices
            new_order = [left_idx, face_idx, right_idx]

            # 두 리스트를 같은 순서로 재배치
            bounding_contours = [bounding_contours[i] for i in new_order]
            crop_list = [crop_list[i] for i in new_order]




        # body25 키포인트 추출, 저장
        # ====== 여기서 bilateral filter 적용 ======
        filtered_frame = cv2.bilateralFilter(frame, d=9, sigmaColor=75, sigmaSpace=75)
        # =========================================
        points, body_confidences = body(filtered_frame, prev_points)
        
        if len(crop_list) == 3:
            x0, y0, w0, h0, _ = bounding_contours[0]
            h0, w0_crop = crop_list[0].shape[:2]
            center_x0 = x0 + w0_crop // 2
            center_y0 = y0 + h0 // 2
            points[BODY_PARTS["RWrist"]] = (center_x0, center_y0)
           
            x2, y2, w2, h2, _ = bounding_contours[2]
            h2, w2_crop = crop_list[2].shape[:2]
            center_x2 = x2 + w2_crop // 2
            center_y2 = y2 + h2 // 2
            points[BODY_PARTS["LWrist"]] = (center_x2, center_y2)

        # json 파일에 저장
        for idx in range(len(points)):
            if points[idx]:
                frame_data["body_keypoints_2d"].extend([points[idx][0], points[idx][1], body_confidences[idx]])
            else:
                frame_data["body_keypoints_2d"].extend([0, 0, 0])
        prev_points = points.copy()  # 현재 프레임의 키포인트를 이전 프레임으로 저장

        
        # # face 키포인트 추출, 저장
        # if points[BODY_PARTS["Neck"]] is not None:
        #     frameHeight, frameWidth, _ = frame.shape
            
        #     # 목 좌표를 기준으로 이미지를 크롭하기
        #     neck = points[BODY_PARTS["Neck"]]
        #     box_size = int(min(frameWidth, frameHeight) * 0.3) #크롭이미지의 크기는 w h 전체의 30%정도
        #     x = max(0, neck[0] - box_size//2)
        #     y = max(0, neck[1] - box_size) 
        #     w = min(box_size, frameWidth - x)
        #     h = min(box_size, frameHeight - y)

        #     if w > 20 and h > 20:
        #         # 크롭한 이미지 전달, 예측 수행
        #         crop = frame[y:y+h, x:x+w].copy()
        #         face_points, face_confidences = face(crop) 
                
        #         for idx in range(len(face_points)):
        #             if face_points[idx]:
        #                 frame_data["face_keypoints_2d"].extend([
        #                     face_points[idx][0] + x,
        #                     face_points[idx][1] + y,
        #                     face_confidences[idx]
        #                 ])
        #             else:
        #                 frame_data["face_keypoints_2d"].extend([0, 0, 0])
        #         # # 프레임에 그리기
        #         # for pair in FACE_PAIRS:
        #         #     partA = pair[0]            
        #         #     partA = FACE_PARTS[partA] 
        #         #     partB = pair[1]            
        #         #     partB = FACE_PARTS[partB]  
                        
        #         #     if face_points[partA] and face_points[partB]:
        #         #         ptA = (int(face_points[partA][0]) + x, int(face_points[partA][1]) + y)
        #         #         ptB = (int(face_points[partB][0]) + x, int(face_points[partB][1]) + y)
        #         #         cv2.line(frame, ptA, ptB, (0, 255, 255), 1)
        




                
        # 왼손 hand 키포인트 추출, 저장
        if points[BODY_PARTS["LWrist"]] is not None and points[BODY_PARTS["LElbow"]] is not None:
            frameHeight, frameWidth, _ = frame.shape

            # crop과 crop_left_top(좌상단) 계산
            if len(crop_list) == 3:
                crop = crop_list[2]
                x_left, y_left, _, _, _ = bounding_contours[2]
                crop_left_top = (x_left, y_left)
            elif prev_crop_list is not None and len(prev_crop_list) == 3:
                crop = prev_crop_list[2]
                x_left, y_left, _, _, _ = prev_bounding_contours[2]
                crop_left_top = (x_left, y_left)
            else:
                crop_w, crop_h = int(frameWidth * 0.3), int(frameHeight * 0.3)
                cx, cy = points[BODY_PARTS["LWrist"]]
                x1 = max(cx - crop_w // 2, 0)
                y1 = max(cy - crop_h // 2, 0)
                x2 = min(cx + crop_w // 2, frameWidth)
                y2 = min(cy + crop_h // 2, frameHeight)
                crop = frame[y1:y2, x1:x2]
                crop_left_top = (x1, y1)

            if crop is not None:
                hsv_crop = cv2.cvtColor(crop, cv2.COLOR_BGR2HSV)
                skinMask = cv2.inRange(hsv_crop, lower, upper)
                skinMask = cv2.erode(skinMask, None, iterations=2)
                skinMask = cv2.dilate(skinMask, None, iterations=2)
                skinMask = cv2.GaussianBlur(skinMask, (3, 3), 0)
                skinMask = cv2.medianBlur(skinMask, 13)
                filtered_crop = cv2.bitwise_and(crop, crop, mask=skinMask)
                filtered_crop = cv2.bilateralFilter(filtered_crop, d=9, sigmaColor=75, sigmaSpace=75)



                cv2.imshow("hand_crop_left", filtered_crop)
                cv2.waitKey(1)

                vector = np.array(points[BODY_PARTS["LWrist"]]) - np.array(points[BODY_PARTS["LElbow"]])

                 

                wrist=points[BODY_PARTS["LWrist"]]
                elbow=points[BODY_PARTS["LElbow"]]
                wrist = (wrist[0] - crop_left_top[0], wrist[1] - crop_left_top[1])
                elbow = (elbow[0] - crop_left_top[0], elbow[1] - crop_left_top[1])

                hand_left_points, hand_left_confidences = hand(
                    filtered_crop, 
                    wrist, 
                    prev_left_hand_points)

                prev_left_hand_points = hand_left_points
                for idx in range(len(hand_left_points)):
                    if hand_left_points[idx]:
                        frame_data["hand_left_keypoints_2d"].extend([
                            hand_left_points[idx][0] + crop_left_top[0],
                            hand_left_points[idx][1] + crop_left_top[1],
                            hand_left_confidences[idx]
                        ])
                    else:
                        frame_data["hand_left_keypoints_2d"].extend([0, 0, 0])

                for pair in HAND_PAIRS:
                    partA = HAND_PARTS[pair[0]]
                    partB = HAND_PARTS[pair[1]]
                    if hand_left_points[partA] and hand_left_points[partB]:
                        ptA = (int(hand_left_points[partA][0]) + crop_left_top[0], int(hand_left_points[partA][1]) + crop_left_top[1])
                        ptB = (int(hand_left_points[partB][0]) + crop_left_top[0], int(hand_left_points[partB][1]) + crop_left_top[1])
                        cv2.line(frame, ptA, ptB, (255,255,0), 1)
                for part in HAND_PARTS:
                    if hand_left_points[HAND_PARTS[part]]:
                        pt = (int(hand_left_points[HAND_PARTS[part]][0]) + crop_left_top[0], int(hand_left_points[HAND_PARTS[part]][1]) + crop_left_top[1])
                        cv2.circle(frame, pt, 2, (0, 0,255), -1)

        # 오른손 hand 키포인트 추출, 저장
        if points[BODY_PARTS["RWrist"]] is not None and points[BODY_PARTS["RElbow"]] is not None:
            frameHeight, frameWidth, _ = frame.shape

            if len(crop_list) == 3:
                crop = crop_list[0]
                x_right, y_right, _, _, _ = bounding_contours[0]
                crop_right_top = (x_right, y_right)
            elif prev_crop_list is not None and len(prev_crop_list) == 3:
                crop = prev_crop_list[0]
                x_right, y_right, _, _, _ = prev_bounding_contours[0]
                crop_right_top = (x_right, y_right)
            else:
                crop_w, crop_h = int(frameWidth * 0.3), int(frameHeight * 0.3)
                cx, cy = points[BODY_PARTS["RWrist"]]
                x1 = max(cx - crop_w // 2, 0)
                y1 = max(cy - crop_h // 2, 0)
                x2 = min(cx + crop_w // 2, frameWidth)
                y2 = min(cy + crop_h // 2, frameHeight)
                crop = frame[y1:y2, x1:x2]
                crop_right_top = (x1, y1)

            if crop is not None:
                hsv_crop = cv2.cvtColor(crop, cv2.COLOR_BGR2HSV)
                skinMask = cv2.inRange(hsv_crop, lower, upper)
                skinMask = cv2.erode(skinMask, None, iterations=2)
                skinMask = cv2.dilate(skinMask, None, iterations=2)
                skinMask = cv2.GaussianBlur(skinMask, (3, 3), 0)
                skinMask = cv2.medianBlur(skinMask, 13)
                filtered_crop = cv2.bitwise_and(crop, crop, mask=skinMask)
                filtered_crop = cv2.bilateralFilter(filtered_crop, d=9, sigmaColor=75, sigmaSpace=75)



                cv2.imshow("hand_crop_right", filtered_crop)
                cv2.waitKey(1)
                vector = np.array(points[BODY_PARTS["RWrist"]]) - np.array(points[BODY_PARTS["RElbow"]])

                 

                wrist=points[BODY_PARTS["RWrist"]]
                elbow=points[BODY_PARTS["RElbow"]]
                wrist = (wrist[0] - crop_right_top[0], wrist[1] - crop_right_top[1])
                elbow = (elbow[0] - crop_right_top[0], elbow[1] - crop_right_top[1])
                hand_right_points, hand_right_confidences =hand(
                    filtered_crop, 
                    wrist, 
                    prev_right_hand_points)
                

                prev_right_hand_points = hand_right_points
                for idx in range(len(hand_right_points)):
                    if hand_right_points[idx]:
                        frame_data["hand_right_keypoints_2d"].extend([
                            hand_right_points[idx][0] + crop_right_top[0],
                            hand_right_points[idx][1] + crop_right_top[1],
                            hand_right_confidences[idx]
                        ])
                    else:
                        frame_data["hand_right_keypoints_2d"].extend([0, 0, 0])

                for pair in HAND_PAIRS:
                    partA = HAND_PARTS[pair[0]]
                    partB = HAND_PARTS[pair[1]]
                    if hand_right_points[partA] and hand_right_points[partB]:
                        ptA = (int(hand_right_points[partA][0]) + crop_right_top[0], int(hand_right_points[partA][1]) + crop_right_top[1])
                        ptB = (int(hand_right_points[partB][0]) + crop_right_top[0], int(hand_right_points[partB][1]) + crop_right_top[1])
                        cv2.line(frame, ptA, ptB, (255,255,0), 1)
                for part in HAND_PARTS:
                    if hand_right_points[HAND_PARTS[part]]:
                        pt = (int(hand_right_points[HAND_PARTS[part]][0]) + crop_right_top[0], int(hand_right_points[HAND_PARTS[part]][1]) + crop_right_top[1])
                        cv2.circle(frame, pt, 2, (0, 0,255), -1)






        if len(crop_list) == 3:
            prev_bounding_contours = bounding_contours.copy()
            prev_crop_list = crop_list.copy()                
               
        # body25 키포인트 프레임에 그리기
        for pair in BODY_PAIRS:
            partA = pair[0]            
            partA = BODY_PARTS[partA]  
            partB = pair[1]            
            partB = BODY_PARTS[partB]  
            if points[partA] and points[partB]:
                cv2.line(frame, points[partA], points[partB], (255,0,255), 2)
        # 손목/팔꿈치 좌표 시각화
        if points[BODY_PARTS["LWrist"]] is not None:
            cv2.circle(frame, points[BODY_PARTS["LWrist"]], 8, (0, 0, 255), -1) # 빨간 점
        if points[BODY_PARTS["RWrist"]] is not None:
            cv2.circle(frame, points[BODY_PARTS["RWrist"]], 8, (255, 0, 0), -1)



        # 추출한 좌표를 프레임 번호와 함께 저장
        frames_data.append(frame_data)
        processed_count += 1

        cv.imshow('test', frame)

        # 다음 프레임으로 이동
        frame_count += 1

    # 모든 프레임의 좌표정보를 json 파일에 저장
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump({
            "total_frames": frame_count,
            "processed_frames": processed_count,
            "frames": frames_data
        }, f, ensure_ascii=False, indent=2)

    cv2.destroyAllWindows()

    print(f'키포인트 추출 완료 {json_filename}')

    return json_path






    
if __name__ == "__main__":
    # 입력:영상 출력:신체 키포인트들의 좌표정보 
    # video_path="C:/Users/DS/Desktop/kimsihyun/my/datas/video/KETI_SL_0000000255.avi"
    video_path="C:/Users/DS/Desktop/kimsihyun/my/datas/video/KETI_SL_0000002911.avi"
    json_dir = "C:/Users/DS/Desktop/kimsihyun/my/datas/keypoints"

    filename, _ = os.path.splitext(video_path) #경로에서 파일명만 떼어내기 (json 파일 이름에 활용하기위해)
    cap = cv.VideoCapture(video_path)
    # cap = cv2.VideoCapture(0)

    print(video_to_keypoints(cap, json_dir, filename))