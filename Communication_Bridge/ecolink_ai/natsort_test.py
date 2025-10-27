import os
from natsort import natsorted


video_dir='C:/Users/DS/Desktop/kimsihyun/Communication_Bridge/ecolink_ai/final_datas/4_testvideo_12'

video_filepaths = []
for dirpath, dirnames, filenames in os.walk(video_dir):
    mp4_files = [file for file in filenames if file.lower().endswith('.mp4')]
    # 자연 정렬 적용
    for file in natsorted(mp4_files):
        video_path = os.path.join(dirpath, file).replace('\\', '/')
        video_filepaths.append(video_path)
        print(video_path)


pip install natsort
