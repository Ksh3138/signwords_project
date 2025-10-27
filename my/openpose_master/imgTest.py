import cv2
import cv2 as cv

BODY_PARTS = {
    "Head": 0, "Neck": 1, "RShoulder": 2, "RElbow": 3, "RWrist": 4,
    "LShoulder": 5, "LElbow": 6, "LWrist": 7, "Chest": 14
}
BODY_PAIRS = [
    ["Head", "Neck"], ["Neck", "RShoulder"], ["RShoulder", "RElbow"],
    ["RElbow", "RWrist"], ["Neck", "LShoulder"], ["LShoulder", "LElbow"],
    ["LElbow", "LWrist"], ["Neck", "Chest"]
]

HAND_PARTS = {
    "Wrist": 0,
    "ThumbMetacarpal": 1, "ThumbProximal": 2, "ThumbMiddle": 3, "ThumbDistal": 4,
    "IndexFingerMetacarpal": 5, "IndexFingerProximal": 6, "IndexFingerMiddle": 7, "IndexFingerDistal": 8,
    "MiddleFingerMetacarpal": 9, "MiddleFingerProximal": 10, "MiddleFingerMiddle": 11, "MiddleFingerDistal": 12,
    "RingFingerMetacarpal": 13, "RingFingerProximal": 14, "RingFingerMiddle": 15, "RingFingerDistal": 16,
    "LittleFingerMetacarpal": 17, "LittleFingerProximal": 18, "LittleFingerMiddle": 19, "LittleFingerDistal": 20,
}
HAND_PAIRS = [["Wrist", "ThumbMetacarpal"], ["ThumbMetacarpal", "ThumbProximal"],
                ["ThumbProximal", "ThumbMiddle"], ["ThumbMiddle", "ThumbDistal"],
                ["Wrist", "IndexFingerMetacarpal"], ["IndexFingerMetacarpal", "IndexFingerProximal"],
                ["IndexFingerProximal", "IndexFingerMiddle"], ["IndexFingerMiddle", "IndexFingerDistal"],
                ["Wrist", "MiddleFingerMetacarpal"], ["MiddleFingerMetacarpal", "MiddleFingerProximal"],
                ["MiddleFingerProximal", "MiddleFingerMiddle"], ["MiddleFingerMiddle", "MiddleFingerDistal"],
                ["Wrist", "RingFingerMetacarpal"], ["RingFingerMetacarpal", "RingFingerProximal"],
                ["RingFingerProximal", "RingFingerMiddle"], ["RingFingerMiddle", "RingFingerDistal"],
                ["Wrist", "LittleFingerMetacarpal"], ["LittleFingerMetacarpal", "LittleFingerProximal"],
                ["LittleFingerProximal", "LittleFingerMiddle"], ["LittleFingerMiddle", "LittleFingerDistal"]]

bodyProtoFile = "./models/pose/body_25/pose_deploy.prototxt"
bodyWeightsFile = "./models/pose/body_25/pose_iter_584000.caffemodel"

bodyNet = cv2.dnn.readNetFromCaffe(bodyProtoFile, bodyWeightsFile)

handProtoFile = "./models/hand/pose_deploy.prototxt" 
handWeightsFile = "./models/hand/pose_iter_102000.caffemodel" 

handNet = cv.dnn.readNetFromCaffe(handProtoFile, handWeightsFile)






def body(frame):
    # frame.shape = 불러온 이미지에서 height, width, color 받아옴
    frameHeight, frameWidth, _ = frame.shape

    # network에 넣기위해 전처리
    inpBlob = cv2.dnn.blobFromImage(frame, 1.0 / 255, (frameHeight, frameWidth), (0, 0, 0), swapRB=False, crop=False)

    # network에 넣어주기
    bodyNet.setInput(inpBlob)

    # 결과 받아오기
    output = bodyNet.forward()

    # 키포인트 검출시 이미지에 그려줌
    points = [None] * (max(BODY_PARTS.values()) + 1)

    for part, idx in BODY_PARTS.items():
        probMap = output[0, idx, :, :]
        
        minVal, prob, minLoc, point = cv2.minMaxLoc(probMap)

        x = (frameWidth * point[0]) / output.shape[3]
        y = (frameHeight * point[1]) / output.shape[2]

        if prob > 0.1:
            points[idx] = (int(x), int(y))
    
    # 4번이 오른손, 7번이 왼손
    LWrist=points[7]
    RWrist=points[4]

    return frame, LWrist, RWrist, points







def hand(crop):
    threshold = 0.1

    inputHeight = 368
    inputWidth = 368
    inputScale = 1.0/255

    cropWidth = crop.shape[1]
    cropHeight = crop.shape[0]
    inp = cv.dnn.blobFromImage(crop, inputScale, (inputWidth, inputHeight), (0, 0, 0), swapRB=False, crop=False)

    handNet.setInput(inp)
    out = handNet.forward()

    points = []
    points.append(LWrist)
    for i in range(1,len(HAND_PARTS)):
        heatMap = out[0, i, :, :]

        _, conf, _, point = cv.minMaxLoc(heatMap)
        x = int((cropWidth * point[0]) / out.shape[3])
        y = int((cropHeight * point[1]) / out.shape[2])

        if conf > threshold:
            # cv.circle(crop, (x, y), 3, (0, 255, 255), thickness=-1, lineType=cv.FILLED)
            points.append((x, y))
        else:
            points.append(None)


    for pair in HAND_PAIRS:
        partFrom = pair[0]
        partTo = pair[1]

        idFrom = HAND_PARTS[partFrom]
        idTo = HAND_PARTS[partTo]
        if points[idFrom] and points[idTo]:
            cv.line(crop, points[idFrom], points[idTo], (0, 255, 0), 1)
    
    return crop






if __name__ == "__main__":
    # 1. 이미지 불러오기
    frame = cv2.imread("testimg.png")
    
    # 2. 몸 
    frame, LWrist, RWrist, points = body(frame)

    # 3. 왼손
    if LWrist is not None:
        #자르기
        frameHeight, frameWidth, _ = frame.shape
        x, y, w, h = int(LWrist[0]-frameWidth/8), int(LWrist[1]-frameWidth/8), int(frameWidth/4), int(frameHeight/4)
        crop=frame[y:y+h, x:x+w].copy()
        #손 키포인트 추출
        result = hand(crop)
        # 원본 프레임에 붙이기
        frame[y:y+h, x:x+w] = result

    # 3. 오른손 
    if RWrist is not None:
        #자르기
        frameHeight, frameWidth, _ = frame.shape
        x, y, w, h = int(RWrist[0]-frameWidth/16), int(RWrist[1]-frameWidth/8), int(frameWidth/4), int(frameHeight/4)
        crop=frame[y:y+h, x:x+w].copy()
        #손 키포인트 추출
        result = hand(crop)
        # 원본 프레임에 붙이기
        frame[y:y+h, x:x+w] = result


    # 2. 몸
    # 각 POSE_PAIRS별로 선 그어줌 (머리 - 목, 목 - 왼쪽어깨, ...)
    # for point in points:
    #     if(point):
    #         cv2.circle(frame, (int(point[0]), int(point[1])), 3, (255, 0, 255), thickness=-1, lineType=cv2.FILLED)

    for pair in BODY_PAIRS:
        partA = pair[0]             # Head
        partA = BODY_PARTS[partA]   # 0
        partB = pair[1]             # Neck
        partB = BODY_PARTS[partB]   # 1
            
        #partA와 partB 사이에 선을 그어줌 (cv2.line)
        if points[partA] and points[partB]:
            cv2.line(frame, points[partA], points[partB], (255,0,255), 2)

    # 4. 출력
    cv.imshow('test',frame)
    

    cv2.waitKey()
    cv2.destroyAllWindows()