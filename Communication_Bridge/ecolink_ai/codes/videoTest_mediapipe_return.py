import cv2
import mediapipe as mp
import time
import os
import boto3


#s3에 접근
#pose 전체 list -> return



# 관절(pose, hand, face) 검출을 위한 모델 클래스 (를 포함한 모듈...)
mp_holistic = mp.solutions.holistic
# 검출한 관절을 그리기 위한 함수, 스타일 설정 (를 포함한 모듈22)
mp_drawing = mp.solutions.drawing_utils 
mp_drawing_styles = mp.solutions.drawing_styles


def landmarks_to_dict(landmarks):
    if landmarks is None:
        return None
    return [
        {'x': lm.x, 'y': lm.y, 'z': lm.z, 'visibility': getattr(lm, 'visibility', None)}
        for lm in landmarks.landmark
    ]


# 키포인트 추출하고 리스트 return
def video_to_keypoints(cap):
    prev_time = 0

    # json 저장
    all_frames = []
    frame_idx = 0


    # with문: 종료시 자동으로 자원 해제
    # as문: 생성된 객체의 이름

    # 관절검출을위한 모델객체 생성 - 인자는 차례대로 각 프레임을 독립처리할지 여부, 신뢰도 임곗값, 모델복잡도(높을수록 정확도ㅅ 속도v)
    with mp_holistic.Holistic(
        static_image_mode=True, min_detection_confidence=0.5, model_complexity=2) as holistic:
        
        # 프레임마다 관절 예측, 그리기
        while cap.isOpened():
            success, image = cap.read()
            if not success:
                print("마지막 프레임")
                break
            
            curr_time = time.time()

            # 예측 수행 (cvtColor는 전처리)
            results = holistic.process(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))

            # 프레임에 그리기
            annotated_image = image.copy()
            mp_drawing.draw_landmarks(annotated_image, results.left_hand_landmarks, mp_holistic.HAND_CONNECTIONS)
            mp_drawing.draw_landmarks(annotated_image, results.right_hand_landmarks, mp_holistic.HAND_CONNECTIONS)
            mp_drawing.draw_landmarks(
                annotated_image,
                results.face_landmarks,
                mp_holistic.FACEMESH_TESSELATION,
                landmark_drawing_spec=None,
                connection_drawing_spec=mp_drawing_styles
                .get_default_face_mesh_tesselation_style())
            mp_drawing.draw_landmarks(
                annotated_image,
                results.pose_landmarks,
                mp_holistic.POSE_CONNECTIONS,
                landmark_drawing_spec=mp_drawing_styles.
                get_default_pose_landmarks_style())

            # FPS 계산해서 프레임에 출력 
            sec = curr_time - prev_time
            prev_time = curr_time
            fps = 1/(sec)
            fps_str = "FPS : %0.1f" % fps
            cv2.putText(annotated_image, fps_str, (0, 100), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0))

            # 결과 저장, 표시
            result_dict = {
                'frame': frame_idx,
                'pose_landmarks': landmarks_to_dict(results.pose_landmarks),
                'left_hand_landmarks': landmarks_to_dict(results.left_hand_landmarks),
                'right_hand_landmarks': landmarks_to_dict(results.right_hand_landmarks)
            }
            all_frames.append(result_dict)
            frame_idx += 1

            cv2.imshow('MediaPipe Pose Result', annotated_image)
            if cv2.waitKey(1) & 0xFF == 27:
                break

    cap.release()
    cv2.destroyAllWindows()
            
    return all_frames


if __name__ == "__main__":
    # 접근키, 비밀키
    os.environ['AWS_ACCESS_KEY_ID'] = '접근키'
    os.environ['AWS_SECRET_ACCESS_KEY'] = '비밀키'
    os.environ['AWS_DEFAULT_REGION'] = 'ap-northeast-2' 

    s3 = boto3.client('s3')

    # 파일 다운로드
    bucket_name = 'echolink-dataset-archive-2025'
    s3_file='dataset1/내일/KETI_SL_0000000168.avi'
    local_file='test.avi'
    s3.download_file(bucket_name, s3_file, local_file)
        
    # 다운로드 완료 후 AWS 자격 증명 삭제
    del os.environ['AWS_ACCESS_KEY_ID']
    del os.environ['AWS_SECRET_ACCESS_KEY']



    cap = cv2.VideoCapture(local_file)
    landmarks_list=video_to_keypoints(cap)

    # print(landmarks_list)
    