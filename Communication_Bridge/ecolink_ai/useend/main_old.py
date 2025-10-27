import cv2
import os
from codes.videoTest_mediapipe_1122 import video_to_keypoints
from ecolink_ai.codes.normalization_json import normalization
from codes.emedding_test_10_json import classify




if __name__ == "__main__":
    # cap = cv2.VideoCapture('C:/Users/DS/Desktop/kimsihyun/Communication_Bridge/ecolink_ai/datas/test_datas/12/KETI_SL_0000012156.mp4')
    json_out = 'C:/Users/DS/Desktop/kimsihyun/Communication_Bridge/ecolink_ai/datas/test_datas/test.json'
    model_path = 'C:/Users/DS/Desktop/kimsihyun/Communication_Bridge/ecolink_ai/datas/best_model_checkpoint_10.pth'


    video_dir="C:/Users/DS/Desktop/kimsihyun/Communication_Bridge/ecolink_ai/datas/test_datas"

    

    for dirpath, dirnames, filenames in os.walk(video_dir):
        for file in filenames:
            if file.lower().endswith('.mp4'):
                video_path = os.path.join(dirpath, file).replace('\\', '/')
                cap = cv2.VideoCapture(video_path)    
                
                video_to_keypoints(cap,json_out)
                normalization(json_out,json_out)
                result=classify(model_path, json_out)
                print(f"{file}: {result}")