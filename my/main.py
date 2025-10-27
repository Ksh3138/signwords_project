import os
import cv2 as cv
from openpose_master.videoTest import video_to_keypoints
from vector.vectorTest import vector_normalization
from st_gcn_master.vectorEmbeddingTest import vector_embedding

def main():
    video_path = "C:/Users/DS/Desktop/kimsihyun/my/datas/video/KETI_SL_0000000168.avi"
    json_dir = "C:/Users/DS/Desktop/kimsihyun/my/datas/keypoints"
    vector_dir = "C:/Users/DS/Desktop/kimsihyun/my/datas/vector"
    embedding_dir = "C:/Users/DS/Desktop/kimsihyun/my/datas/embeddings"

    filename, _ = os.path.splitext(video_path)
    cap = cv.VideoCapture(video_path)
    # cap = cv.VideoCapture(0) 

    json_path = video_to_keypoints(cap, json_dir, filename)
    vector_path = vector_normalization(json_path, vector_dir)
    embedding_path = vector_embedding(vector_path, embedding_dir)


if __name__=="__main__":
    main()