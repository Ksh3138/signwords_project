import json
import mysql.connector
import os
import glob

db = mysql.connector.connect(
    host="localhost",
    user="root",
    password="0000",
    database="video_vector"
)
cursor = db.cursor()

json_dir = "C:/Users/DS/Desktop/kimsihyun/my/datas/embeddings/"
# 하위 폴더까지 모든 .json 파일 찾기
json_files = glob.glob(os.path.join(json_dir, "**", "*.json"), recursive=True)

for json_file in json_files:
    with open(json_file, 'r', encoding='utf-8') as file:
        vector_data = json.load(file)
        vector_data = vector_data["embedding"]

    # 파일이 있던 하위폴더명(라벨) 추출
    folder_name = os.path.basename(os.path.dirname(json_file))
    word = folder_name

    # sql = "INSERT INTO signs_ksh (word, vector) VALUES (%s, %s)"
    sql = "INSERT INTO signs_khn (word, vector) VALUES (%s, %s)"
    values = (word, json.dumps(vector_data))
    cursor.execute(sql, values)
    db.commit()
    print(f"'{word}' 데이터가 성공적으로 저장되었습니다.")

cursor.close()
db.close()
print("\n모든 파일 처리가 완료되었습니다.")
