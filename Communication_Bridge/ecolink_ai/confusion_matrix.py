import cv2
import os
#########확인#########
from codes.videoTest_mediapipe_54node_json import video_to_keypoints
from codes.normalization_return_try import normalization
from codes.emedding_test_n_return_54node import classify
#####################
from sklearn.metrics import f1_score, confusion_matrix, ConfusionMatrixDisplay
import matplotlib.pyplot as plt
plt.rcParams['font.family'] = 'Malgun Gothic'




test_label_list_12 = ['경찰', '교통사고', '구해주세요', '깔리다', 
                      '쓰러지다', '배고프다', '병원', '불나다', 
                      '숨을안쉬다', '아빠', '연락해주세요', '피나다'] 




# model_path = 'C:/Users/DS/Desktop/kimsihyun/Communication_Bridge/ecolink_ai/datas/checkpoint/54node10emer_tryNor_gaussimirrorAugmen.pth'
model_path = 'C:/Users/DS/Desktop/kimsihyun/Communication_Bridge/ecolink_ai/final_datas/checkpoint_12words_54node_trynor_gaussi_notcam.pth'


video_dir='C:/Users/DS/Desktop/kimsihyun/Communication_Bridge/ecolink_ai/final_datas/4_testvideo_12'

json_path = 'C:/Users/DS/Desktop/kimsihyun/Communication_Bridge/ecolink_ai/trashbin/test.json'


y_true= test_label_list_12*4
y_pred=[]

for dirpath, dirnames, filenames in os.walk(video_dir):
    for file in filenames:
        if file.lower().endswith('.mp4'):
            video_path = os.path.join(dirpath, file).replace('\\', '/')
            cap = cv2.VideoCapture(video_path)    
                
            keypoints=video_to_keypoints(cap,json_path)
            normalized_keypoints=normalization(keypoints)
            result=classify(model_path, normalized_keypoints)
            y_pred.append(result)
            print(f"{file}: {result}")


# f1 score
print(f'f1 score: {f1_score(y_true, y_pred, average="macro"):.2f}')



# 혼동행렬 시각화
cm = confusion_matrix(y_true, y_pred, labels=test_label_list_12)

disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=test_label_list_12)
disp.plot(cmap=plt.cm.Blues, xticks_rotation=90)
plt.title('Confusion Matrix')
plt.show()