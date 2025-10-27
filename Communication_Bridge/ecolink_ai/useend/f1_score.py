import cv2
import os
from codes.videoTest_mediapipe_return import video_to_keypoints
from codes.normalization_return import normalization
from codes.emedding_test_10_return import classify
from sklearn.metrics import f1_score
import matplotlib.pyplot as plt

label_list=["계단","공원","내일","배고프다","선반","아파트","유리","집","학교","화재"]





model_path = 'C:/Users/DS/Desktop/kimsihyun/Communication_Bridge/ecolink_ai/datas/poseAll10_nor0_augmenGussi5.pth'


video_dir="C:/Users/DS/Desktop/kimsihyun/Communication_Bridge/ecolink_ai/datas/test_video"

y_true= label_list*2    
y_pred=[]

for dirpath, dirnames, filenames in os.walk(video_dir):
    for file in filenames:
        if file.lower().endswith('.mp4'):
            video_path = os.path.join(dirpath, file).replace('\\', '/')
            cap = cv2.VideoCapture(video_path)    
                
            keypoints=video_to_keypoints(cap)
            normalized_keypoints=normalization(keypoints)
            result=classify(model_path, normalized_keypoints)
            y_pred.append(result)
            print(f"{file}: {result}")





# f1 = f1_score(y_true, y_pred, average='macro')
# print(f"f1-score(macro): {f1}")


f1 = f1_score(y_true, y_pred, average=None)
for name, score in zip(label_list, f1):
    print(f"{name}: {score:.2f}")

plt.bar(label_list, f1)
plt.xlabel('class')
plt.ylabel('f1-score')
plt.title('title')
plt.show()

