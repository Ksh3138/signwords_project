import numpy as np
import mysql.connector
import ast
import json
import os
import cv2 as cv
from openpose_master.videoTest_ksh import video_to_keypoints
from vector.vectorTest import vector_normalization
from st_gcn_master.vectorEmbeddingTest import vector_embedding

db = mysql.connector.connect(
    host="localhost",
    user="root",
    password="0000",
    database="video_vector"
)
cursor = db.cursor()
cursor.execute("select word,vector from signs_ksh")
rows = cursor.fetchall()


video_path = "C:/Users/DS/Desktop/kimsihyun/my/datas/video/KETI_SL_0000002911.avi"
filename, _ = os.path.splitext(video_path)
cap = cv.VideoCapture(video_path)
# cap = cv.VideoCapture(0) 

json_dir = "C:/Users/DS/Desktop/kimsihyun/my/datas/keypoints"
vector_dir = "C:/Users/DS/Desktop/kimsihyun/my/datas/vector"
embedding_dir = "C:/Users/DS/Desktop/kimsihyun/my/datas/embeddings"

# json_path = video_to_keypoints(cap, json_dir, filename)
json_path = "C:/Users/DS/Desktop/kimsihyun/my/datas/keypoints/KETI_SL_0000002911.json"
# json_path = "C:/Users/DS/Desktop/kimsihyun/my/datas/keypoints/KETI_SL_0000004586.json"
vector_path = vector_normalization(json_path, vector_dir)
embedding_path = vector_embedding(vector_path, embedding_dir)

with open(embedding_path, 'r', encoding='utf-8') as file:
    new_vector = json.load(file)
    new_vector = new_vector['embedding']


# 방법1) 코사인 유사도
# max_word=""
# max_cossim=0.7
# for word, vector in rows:
#     vector = np.array(ast.literal_eval(vector))
#     cos_sim = np.dot(vector, new_vector) / (np.linalg.norm(vector) * np.linalg.norm(new_vector))
#     print(f'{word} 코사인 유사도: {cos_sim}')
#     if cos_sim > max_cossim:
#         max_cossim = cos_sim
#         max_word = word
# if max_word=="":
#     print(f'유사한 단어 없음\n')
# else:
#     print(f'\n가장 유사한 단어는 {max_word}\n')



# 방법2) 유클리드 거리
min_word=""
min_eudis=30
# for word, vector in rows:
#     vector = np.array(ast.literal_eval(vector))
#     euclidean_distance = np.linalg.norm(vector - new_vector)
#     print(f'{word} 유클리드 거리: {euclidean_distance}')
#     if euclidean_distance < min_eudis:
#         min_eudis=euclidean_distance
#         min_word=word

word_eudis_list=[]
for word, vector in rows:
    vector = np.array(ast.literal_eval(vector))
    euclidean_distance = np.linalg.norm(vector - new_vector)
    word_eudis_list.append([word, euclidean_distance])
# for word, eudis in word_eudis_list:
#     print(f'{word} 유클리드 거리: {eudis}')
#     if eudis < min_eudis:
#         min_eudis = eudis
#         min_word = word



eudis_values = [eudis for word, eudis in word_eudis_list]
min_eudis = min(eudis_values)
max_eudis = max(eudis_values)

normalized_word_eudis_list = []
for word, eudis in word_eudis_list:
    if max_eudis == min_eudis:
        normalized = 100
    else:
        normalized = (eudis / max_eudis) * 100
    normalized_word_eudis_list.append([word, normalized])
print("유클리드 거리 (0~100)")
for word, eudis in normalized_word_eudis_list:
    print(f'{word}: {eudis}')
    if eudis < min_eudis:
        min_eudis = eudis
        min_word = word


if min_word=="":
    print(f'가까운 단어 없음')
else:
    print(f'\n가장 가까운 단어는 {min_word}')
