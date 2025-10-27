import os
import cv2
import numpy as np
import mediapipe as mp

actions = ['계단']
seq_length = 30

mp_holistic = mp.solutions.holistic
holistic = mp_holistic.Holistic(
    min_detection_confidence=0.5,
    min_tracking_confidence=0.5
)


for action in actions:
    action_dir = f'C:/Users/DS/Desktop/kimsihyun/Communication_Bridge/ecolink_ai/datas/video/{action}'
    all_data = []
    for filename in os.listdir(action_dir):
        if filename.endswith('.avi'):
            video_path = os.path.join(action_dir, filename)
            cap = cv2.VideoCapture(video_path)
            while cap.isOpened():
                ret, img = cap.read()
                if not ret:
                    break
                img = cv2.flip(img, 1)
                img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
                results = holistic.process(img_rgb)


                if results.left_hand_landmarks is not None:
                    res = results.left_hand_landmarks
                    joint = np.zeros((21, 4))
                    for j, lm in enumerate(res.landmark):
                        joint[j] = [lm.x, lm.y, lm.z, lm.visibility]
                        v1 = joint[[0,1,2,3,0,5,6,7,0,9,10,11,0,13,14,15,0,17,18,19], :3]
                        v2 = joint[[1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20], :3]
                        v = v2 - v1
                        v = v / np.linalg.norm(v, axis=1)[:, np.newaxis]
                        angle = np.arccos(np.einsum('nt,nt->n', v[[0,1,2,4,5,6,8,9,10,12,13,14,16,17,18],:], v[[1,2,3,5,6,7,9,10,11,13,14,15,17,18,19],:]))
                        angle = np.degrees(angle)
                        angle_label = np.array([angle], dtype=np.float32)
                        angle_label = np.append(angle_label, actions.index(action))
                        d = np.concatenate([joint.flatten(), angle_label])
                        all_data.append(d)

                if results.right_hand_landmarks is not None:
                    res = results.right_hand_landmarks
                    joint = np.zeros((21, 4))
                    for j, lm in enumerate(res.landmark):
                        joint[j] = [lm.x, lm.y, lm.z, lm.visibility]
                        v1 = joint[[0,1,2,3,0,5,6,7,0,9,10,11,0,13,14,15,0,17,18,19], :3]
                        v2 = joint[[1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20], :3]
                        v = v2 - v1
                        v = v / np.linalg.norm(v, axis=1)[:, np.newaxis]
                        angle = np.arccos(np.einsum('nt,nt->n', v[[0,1,2,4,5,6,8,9,10,12,13,14,16,17,18],:], v[[1,2,3,5,6,7,9,10,11,13,14,15,17,18,19],:]))
                        angle = np.degrees(angle)
                        angle_label = np.array([angle], dtype=np.float32)
                        angle_label = np.append(angle_label, actions.index(action))
                        d = np.concatenate([joint.flatten(), angle_label])
                        all_data.append(d)

                if results.pose_landmarks is not None:
                    pose = results.pose_landmarks
                    pose_joint = np.zeros((33, 4))
                    for idx, lm in enumerate(pose.landmark):
                        pose_joint[idx] = [lm.x, lm.y, lm.z, lm.visibility]
                        
                
            cap.release()


    #저장
    all_data = np.array(all_data)
    np.save(f'C:/Users/DS/Desktop/kimsihyun/Communication_Bridge/ecolink_ai/datas/dataset/{action}_raw.npy', all_data)
    
    full_seq_data = []
    for seq in range(len(all_data) - seq_length):
        full_seq_data.append(all_data[seq:seq + seq_length])
    full_seq_data = np.array(full_seq_data)
    np.save(f'C:/Users/DS/Desktop/kimsihyun/Communication_Bridge/ecolink_ai/datas/dataset/{action}_seq.npy', full_seq_data)
