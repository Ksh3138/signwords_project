import numpy as np
import cv2
import sys

cap=cv2.VideoCapture('KETI_SL_0000019552.mp4')
if not cap.isOpened():
    sys.exit('카메라 연결 실패')
    
ret, prev_frame=cap.read()		# 첫 프레임
prev_gray=cv2.cvtColor(prev_frame,cv2.COLOR_BGR2GRAY)

while(1):
    ret,frame=cap.read()	# 비디오를 구성하는 프레임 획득
    if not ret:
        sys.exit('프레임 획득에 실패하여 루프를 나갑니다.')
    
    
    height, width = frame.shape[:2]
    print("Width:", width)
    print("Height:", height)


    curr_gray=cv2.cvtColor(frame,cv2.COLOR_BGR2GRAY)
    flow=cv2.calcOpticalFlowFarneback(prev_gray,curr_gray,None,0.5,3,15,3,5,1.2,0)    # optical flow 계산



    for y in range(16//2,frame.shape[0],16):
        for x in range(16//2,frame.shape[1],16):
            dx,dy = flow[y,x].astype(int)
            if (dx*dx+dy*dy) > 200 :
                cv2.line(frame,(x,y),(x+dx,y+dy),(0,0,255),2) # 큰 모션 있는 곳은 빨간색
            else :
                # cv2.line(frame,(x,y),(x+dx,y+dy),(0,255,0),2)
                pass

    cv2.imshow('Optical flow',frame)

    prev_gray=curr_gray   # 현재 프레임을 이전 프레임으로

    key=cv2.waitKey(30)	# 30밀리초 동안 키보드 입력 기다림
    if key==ord('q'):	# 'q' 키가 들어오면 루프를 빠져나감
        break 
    
cap.release()			# 카메라와 연결을 끊음
cv2.destroyAllWindows()