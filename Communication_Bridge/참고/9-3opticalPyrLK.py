import sys

import numpy as np
import cv2

cap=cv2.VideoCapture('KETI_SL_0000000398.avi')
if not cap.isOpened():
    sys.exit('카메라 연결 실패')

ret,prev_frame=cap.read()		# 첫 프레임
prev_gray=cv2.cvtColor(prev_frame,cv2.COLOR_BGR2GRAY)

color=np.random.randint(0,255,(100,3))
feature_params=dict(maxCorners=100,qualityLevel=0.3,minDistance=7,blockSize=7)
lk_params=dict(winSize=(15,15),maxLevel=2,criteria=(cv2.TERM_CRITERIA_EPS|cv2.TERM_CRITERIA_COUNT,10,0.03))
p0=cv2.goodFeaturesToTrack(prev_gray,mask=None,**feature_params)
mask=np.zeros_like(prev_frame)	# 물체의 이동 궤적을 그릴 영상

while(1):
    ret, frame = cap.read()  # 비디오를 구성하는 프레임 획득
    if not ret:
        sys('프레임 획득에 실패하여 루프를 나갑니다.')

    curr_gray=cv2.cvtColor(frame,cv2.COLOR_BGR2GRAY)
    p1,match,err=cv2.calcOpticalFlowPyrLK(prev_gray,curr_gray,p0,None,**lk_params)   # optical flow 계산

    if p1 is not None:		# 양호한 쌍 선택
        good_curr=p1[match==1]
        good_prev=p0[match==1]
        
    for i in range(len(good_curr)): # 이동 궤적 그리기
        a,b=int(good_curr[i][0]),int(good_curr[i][1])
        c,d=int(good_prev[i][0]),int(good_prev[i][1])
        mask=cv2.line(mask,(a,b),(c,d),color[i].tolist(),2)
        frame=cv2.circle(frame,(a,b),5,color[i].tolist(),-1)
        
    img=cv2.add(frame,mask) # 오타
    cv2.imshow('LTK tracker',img)

    prev_gray=curr_gray	    # 현재 프레임을 이전 프레임으로
    p0=good_curr.reshape(-1,1,2)

    key=cv2.waitKey(30)	# 30밀리초 동안 키보드 입력 기다림
    if key==ord('q'):	# 'q' 키가 들어오면 루프를 빠져나감
        break

cap.release()			# 카메라와 연결을 끊음
cv2.destroyAllWindows()