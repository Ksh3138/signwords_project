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
def hand(crop, prev_points=None, threshold=0.1):
    inWidth = 656
    inHeight = 656
    threshold = 0.12 #0.04 -> 0.03으로 신뢰도 높임

    kernel = np.array([[0, -1, 0],
                   [-1, 5, -1],
                   [0, -1, 0]])
    filtered_crop = cv2.filter2D(crop, -1, kernel)


    inpBlob = cv.dnn.blobFromImage(filtered_crop, 1.0/255, (inWidth, inHeight), (0, 0, 0), swapRB=False, crop=False)
    handNet.setInput(inpBlob)
    output = handNet.forward()

    cropHeight, cropWidth = crop.shape[:2] 
    points = [] 
    confidences = [] 
    for i in range(len(HAND_PARTS)):
        probMap = output[0, i, :, :]
        minVal, prob, minLoc, point = cv.minMaxLoc(probMap)

        x = int((cropWidth * point[0]) / output.shape[3])
        y = int((cropHeight * point[1]) / output.shape[2])

        if prob > threshold:
            points.append((x, y))
            confidences.append(float(prob))
        else:
            # 보간: 이전 프레임 값이 있으면 사용
            if prev_points is not None and prev_points[i] is not None:
                points.append(prev_points[i])
            else:
                points.append(None)
            confidences.append(0.0)

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
    prev_hand_points = None

    # 피부색 범위 지정 (예시값, 조정 필요)
    lower = np.array([0, 48, 80], dtype="uint8")
    upper = np.array([20, 255, 255], dtype="uint8")



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



        # 피부색 마스크 생성
        hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
        skinMask = cv2.inRange(hsv, lower, upper)

        # 노이즈 제거 (선택)
        skinMask = cv2.erode(skinMask, None, iterations=2)
        skinMask = cv2.dilate(skinMask, None, iterations=2)
        skinMask = cv2.GaussianBlur(skinMask, (3, 3), 0)

        # 컨투어 검출
        contours, _ = cv2.findContours(skinMask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        h_frame, w_frame = frame.shape[:2] 
        frame_area = h_frame * w_frame # 프레임 크기 

        # 리스트에 저장장
        bounding_contours = []
        for cnt in contours:
            x, y, w, h = cv2.boundingRect(cnt)
            # 전체 프레임의 0.5% 미만인 경우 제외
            if w * h < frame_area * 0.005:
                continue
            bounding_contours.append((x, y, w, h, cnt))

        rwrist = None
        lwrist = None
        if len(bounding_contours) == 3:
            bounding_contours.sort(key=lambda item: item[1])
            hand_bboxes = bounding_contours[1:]
            hand_bboxes.sort(key=lambda item: item[0])
            
            x,y,w,h, _ = bounding_contours[0]
            center_x = x + w // 2
            center_y = y + h 
            neck = (center_x, center_y)
  
            x,y,w,h, _ = hand_bboxes[0]
            center_x = x + w // 2
            center_y = y + h // 2
            rwrist = (center_x, center_y)
     
            x,y,w,h, _ = hand_bboxes[1]
            center_x = x + w // 2
            center_y = y + h // 2
            lwrist =(center_x, center_y)

            if np.linalg.norm(np.array(lwrist) - np.array(rwrist)) < 30:
                rwrist = None
                lwrist = None   
            elif np.linalg.norm(np.array(neck) - np.array(rwrist)) < 30 or rwrist[1] < h_frame/4:
                    rwrist = None 
            elif np.linalg.norm(np.array(neck) - np.array(lwrist)) < 30 or lwrist[1] < h_frame/4:
                    lwrist = None



        # body25 키포인트 추출, 저장
        # ====== 여기서 bilateral filter 적용 ======
        filtered_frame = cv2.bilateralFilter(frame, d=9, sigmaColor=75, sigmaSpace=75)
        # =========================================
        points, body_confidences = body(filtered_frame, prev_points)

        # 컨투어값으로 바꾸기
        if lwrist is not None:
            if points[BODY_PARTS["LWrist"]] is None:
                points[BODY_PARTS["LWrist"]] = lwrist
            else:
                x, y = points[BODY_PARTS["LWrist"]]
                if skinMask[y, x] == 0:
                    points[BODY_PARTS["LWrist"]] = lwrist

        if points[BODY_PARTS["RWrist"]] is None:
            points[BODY_PARTS["RWrist"]] = rwrist
        else:
            x, y = points[BODY_PARTS["RWrist"]]
            if skinMask[y, x] == 0:
                points[BODY_PARTS["RWrist"]] = rwrist
 
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
        #         # 프레임에 그리기
        #         for pair in FACE_PAIRS:
        #             partA = pair[0]            
        #             partA = FACE_PARTS[partA] 
        #             partB = pair[1]            
        #             partB = FACE_PARTS[partB]  
                        
        #             if face_points[partA] and face_points[partB]:
        #                 ptA = (int(face_points[partA][0]) + x, int(face_points[partA][1]) + y)
        #                 ptB = (int(face_points[partB][0]) + x, int(face_points[partB][1]) + y)
        #                 cv2.line(frame, ptA, ptB, (0, 255, 255), 1)
                


        # 왼손 hand 키포인트 추출, 저장
        if points[BODY_PARTS["LWrist"]] is not None:
            frameHeight, frameWidth, _ = frame.shape
            
            crop_w = int(frameWidth * 0.3)
            crop_h = int(frameHeight * 0.3)
            center_x, center_y = points[BODY_PARTS["LWrist"]]
            x1 = center_x - crop_w // 2
            y1 = center_y - crop_h // 2
            x2 = center_x + crop_w // 2
            y2 = center_y + crop_h // 2

            if x1 > 0 and y1 > 0 and x2 < frameWidth and y2 < frameHeight:
                crop = frame[y1:y2, x1:x2]
                
                
                
                # 2. 피부색 마스크 생성
                hsv_crop = cv2.cvtColor(crop, cv2.COLOR_BGR2HSV)
                skinMask = cv2.inRange(hsv_crop, lower, upper)
                skinMask = cv2.erode(skinMask, None, iterations=2)
                skinMask = cv2.dilate(skinMask, None, iterations=2)
                skinMask = cv2.GaussianBlur(skinMask, (3, 3), 0)

                # 3. 컨투어 검출
                contours, _ = cv2.findContours(skinMask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

                # 4. 손목 중심 좌표를 crop 기준으로 변환
                wrist_crop = (center_x - x1, center_y - y1)  # 반드시 crop 좌표계로 변환

                # 5. 손목 중심을 포함하는 컨투어만 선택
                selected_contour = None
                for cnt in contours:
                    if cv2.pointPolygonTest(cnt, wrist_crop, False) >= 0:
                        selected_contour = cnt
                        break

                # 6. 마스크로 컨투어 안만 남기고 바깥은 검정색으로
                mask = np.zeros(crop.shape[:2], dtype=np.uint8)
                if selected_contour is not None:
                    cv2.drawContours(mask, [selected_contour], -1, 255, -1)
                filtered_crop = cv2.bitwise_and(crop, crop, mask=mask)




                # ====== 여기서 bilateral filter 적용 ======
                filtered_crop = cv2.bilateralFilter(filtered_crop, d=9, sigmaColor=75, sigmaSpace=75)
                # =========================================

                # # 디버깅용: 손 크롭 이미지를 화면에 띄우기
                cv2.imshow("hand_crop_left", filtered_crop)
                cv2.waitKey(1)  # 1ms 대기 (필수)

                hand_left_points, hand_left_confidences = hand(filtered_crop, prev_points=prev_hand_points)
                prev_hand_points = hand_left_points

                for idx in range(len(hand_left_points)):
                    if hand_left_points[idx]:
                        frame_data["hand_left_keypoints_2d"].extend([
                            hand_left_points[idx][0] + x,
                            hand_left_points[idx][1] + y,
                            hand_left_confidences[idx]
                        ])
                    else:
                        frame_data["hand_left_keypoints_2d"].extend([0, 0, 0])
                # 프레임에 그리기
                for pair in HAND_PAIRS:
                    partA = pair[0]             
                    partA = HAND_PARTS[partA]   
                    partB = pair[1]              
                    partB = HAND_PARTS[partB]   
                        
                    if hand_left_points[partA] and hand_left_points[partB]:
                        ptA = (int(hand_left_points[partA][0]) + x1, int(hand_left_points[partA][1]) + y1)
                        ptB = (int(hand_left_points[partB][0]) + x1, int(hand_left_points[partB][1]) + y1)
                        cv2.line(frame, ptA, ptB, (255,255,0), 1)
                
                
        # 오른손 hand 키포인트 추출, 저장 (과정은 왼손과 동일)
        if points[BODY_PARTS["RWrist"]] is not None:
            frameHeight, frameWidth, _ = frame.shape

            crop_w = int(frameWidth * 0.3)
            crop_h = int(frameHeight * 0.3)
            center_x, center_y = points[BODY_PARTS["RWrist"]]
            x1 = center_x - crop_w // 2
            y1 = center_y - crop_h // 2
            x2 = center_x + crop_w // 2
            y2 = center_y + crop_h // 2

            if x1 > 0 and y1 > 0 and x2 < frameWidth and y2 < frameHeight:
                crop = frame[y1:y2, x1:x2]



                
                # 2. 피부색 마스크 생성
                hsv_crop = cv2.cvtColor(crop, cv2.COLOR_BGR2HSV)
                skinMask = cv2.inRange(hsv_crop, lower, upper)
                skinMask = cv2.erode(skinMask, None, iterations=2)
                skinMask = cv2.dilate(skinMask, None, iterations=2)
                skinMask = cv2.GaussianBlur(skinMask, (3, 3), 0)

                # 3. 컨투어 검출
                contours, _ = cv2.findContours(skinMask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

                # 4. 손목 중심 좌표를 crop 기준으로 변환
                wrist_crop = (center_x - x1, center_y - y1)  # 반드시 crop 좌표계로 변환

                # 5. 손목 중심을 포함하는 컨투어만 선택
                selected_contour = None
                for cnt in contours:
                    if cv2.pointPolygonTest(cnt, wrist_crop, False) >= 0:
                        selected_contour = cnt
                        break

                # 6. 마스크로 컨투어 안만 남기고 바깥은 검정색으로
                mask = np.zeros(crop.shape[:2], dtype=np.uint8)
                if selected_contour is not None:
                    cv2.drawContours(mask, [selected_contour], -1, 255, -1)
                filtered_crop = cv2.bitwise_and(crop, crop, mask=mask)






                # ====== 여기서 bilateral filter 적용 ======
                filtered_crop = cv2.bilateralFilter(filtered_crop, d=9, sigmaColor=75, sigmaSpace=75)
                # =========================================
                
                # # 디버깅용: 손 크롭 이미지를 화면에 띄우기
                cv2.imshow("hand_crop_right", filtered_crop)
                cv2.waitKey(1)  # 1ms 대기 (필수)

                hand_right_points, hand_right_confidences = hand(filtered_crop, prev_points=prev_hand_points)
                prev_hand_points = hand_right_points
                    
                for idx in range(len(hand_right_points)):
                    if hand_right_points[idx]:
                        frame_data["hand_right_keypoints_2d"].extend([
                            hand_right_points[idx][0] + x,
                            hand_right_points[idx][1] + y,
                            hand_right_confidences[idx]
                        ])
                    else:
                        frame_data["hand_right_keypoints_2d"].extend([0, 0, 0])
                for pair in HAND_PAIRS:
                    partA = pair[0]             
                    partA = HAND_PARTS[partA]   
                    partB = pair[1]              
                    partB = HAND_PARTS[partB]   
                        
                    if hand_right_points[partA] and hand_right_points[partB]:
                        ptA = (int(hand_right_points[partA][0]) + x1, int(hand_right_points[partA][1]) + y1)
                        ptB = (int(hand_right_points[partB][0]) + x1, int(hand_right_points[partB][1]) + y1)
                        cv2.line(frame, ptA, ptB, (255,255,0), 1)
                
                
        # body25 키포인트 프레임에 그리기
        for pair in BODY_PAIRS:
            partA = pair[0]            
            partA = BODY_PARTS[partA]  
            partB = pair[1]            
            partB = BODY_PARTS[partB]  
                
            if points[partA] and points[partB]:
                cv2.line(frame, points[partA], points[partB], (255,0,255), 2)
        ################# 손목 부분을 잘 받는지 좌표로 시각화한 코드 추가################
        # 손목 좌표 시각화 (왼손)
        lwrist_idx = BODY_PARTS["LWrist"]
        if points[lwrist_idx] is not None:
            cv2.circle(frame, points[lwrist_idx], 8, (0, 0, 255), -1) # 빨간 점

        # 손목 좌표 시각화 (오른손)
        rwrist_idx = BODY_PARTS["RWrist"]
        if points[rwrist_idx] is not None:
            cv2.circle(frame, points[rwrist_idx], 8, (255, 0, 0), -1) 

        

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