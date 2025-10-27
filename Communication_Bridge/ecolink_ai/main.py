







# import cv2
# import os
# from codes.videoTest_mediapipe_return import video_to_keypoints
# from codes.normalization_return_moving import normalization
from codes.normalization_json_ import normalization
from codes.emedding_test_10_return import classify
import json




if __name__ == "__main__":
    # cap = cv2.VideoCapture('C:/Users/DS/Desktop/kimsihyun/Communication_Bridge/ecolink_ai/datas/test_datas/12/KETI_SL_0000012156.mp4')
    # json_out = 'C:/Users/DS/Desktop/kimsihyun/Communication_Bridge/ecolink_ai/datas/test_datas/test.json'
    model_path = 'C:/Users/DS/Desktop/kimsihyun/Communication_Bridge/ecolink_ai/datas/75node10_movingNor_augmen0.pth'


    # video_dir="C:/Users/DS/Desktop/kimsihyun/Communication_Bridge/ecolink_ai/datas/test_datas"



    # for dirpath, dirnames, filenames in os.walk(video_dir):
    #     for file in filenames:
    #         if file.lower().endswith('.mp4'):
    #             video_path = os.path.join(dirpath, file).replace('\\', '/')
    #             cap = cv2.VideoCapture(video_path)    
                
    #             keypoints=video_to_keypoints(cap)
    #             normalized_keypoints=normalization(keypoints)
    #             result=classify(model_path, normalized_keypoints)
    #             print(f"{file}: {result}")


    json_file="C:/Users/DS/Downloads/test_집.json"
    output_json="C:/Users/DS/Downloads/test_집_out.json"
    # with open(json_file, 'r') as f:
    #     data = json.load(f)
    # normalized_keypoints=normalization(data)
    # result=classify(model_path, normalized_keypoints)
    # print(f"Result: {result}")


    normalization(json_file,output_json)
