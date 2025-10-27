import torch
import cv2

# print(torch.__version__)
# print(torch.cuda.is_available()) # True면 GPU 사용 가능, False면 CPU만 사용 가능
# print(torch.cuda.get_device_name(0) if torch.cuda.is_available() else "No GPU")
# print(cv2.getBuildInformation())
# print(cv2.__version__)
# print(cv2.__file__)

# print(cv2.getBuildInformation())

print(torch.cuda.is_available())  # True면 GPU 사용 가능
print(torch.cuda.device_count())  # 사용 가능한 GPU 개수
print(torch.cuda.get_device_name(0))  # 첫 번째 GPU 이름