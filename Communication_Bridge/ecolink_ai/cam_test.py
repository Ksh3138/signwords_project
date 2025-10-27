import os
import cv2

from codes.normalization_cam import normalization
from codes.emedding_test_10emer_return_54node import classify





json_dir="C:/Users/DS/Downloads/cam_test_10emer"



output_path="C:/Users/DS/Downloads/test_nor.json"
checkpoint_path = 'C:/Users/DS/Desktop/kimsihyun/Communication_Bridge/ecolink_ai/datas/checkpoint/54node10emer_tryNor_gaussiAugmen.pth'




for dirpath, dirnames, filenames in os.walk(json_dir):
    for file in filenames:
        if file.lower().endswith('.json'):
            json_path = os.path.join(dirpath, file).replace('\\', '/')
            cap = cv2.VideoCapture(json_path)



            normalized_data = normalization(json_path, output_path)
            result=classify(checkpoint_path,normalized_data)

            print(f'{json_path} 결과: {result}')

