
import sys
import os
import pandas as pd
import shutil

def mkdir(excel_path, base_dir):
    os.makedirs(base_dir, exist_ok=True)

    # 엑셀 파일 읽기 (모든 시트 중 첫 번째 시트가 기본)
    df = pd.read_excel(excel_path, header=None)  # header=None은 헤더 없이 읽기

    # 모든 셀의 값을 순회
    for row in df.itertuples(index=False):
        for cell_value in row:
            if pd.isna(cell_value):
                continue  # 빈 셀 무시
            folder_name = str(cell_value).strip()
            if folder_name:
                folder_path = os.path.join(base_dir, folder_name)
                os.makedirs(folder_path, exist_ok=True)
                print(f'폴더 생성: {folder_path}')

def mvfile(old_dir, new_dir, excel_path):
    # 엑셀 불러오기 (헤더 없는 단일 시트)
    df = pd.read_excel(excel_path, header=None)

    # old_dir 내 영상 파일 리스트 (정렬)
    video_files = sorted(os.listdir(old_dir))

    for i, video_file in enumerate(video_files):
        # 엑셀에서 n번째 셀값 (인덱스 0부터 시작)
        try:
            folder_name = str(df.iat[i, 0]).strip()
        except IndexError:
            print(f'엑셀에 {i+1}번째 셀 데이터 없음. 분류 종료.')
            break
        
        if not folder_name:
            print(f'{i+1}번째 셀 데이터가 비어있음. 건너뜀.')
            continue

        src_path = os.path.join(old_dir, video_file)
        target_folder = os.path.join(new_dir, folder_name)
        os.makedirs(target_folder, exist_ok=True)

        dst_path = os.path.join(target_folder, video_file)
        
        try:
            shutil.move(src_path, dst_path)
            print(f'{video_file} -> {target_folder}')
        except Exception as e:
            print(f'이동 실패 {video_file}: {e}')



if __name__ == "__main__":
    # excel_path = 'C:/Users/DS/Desktop/kimsihyun/Communication_Bridge/ecolink_ai/datas/dirnms.xlsx'
    # base_dir = 'C:/Users/DS/Desktop/kimsihyun/Communication_Bridge/ecolink_ai/datas/keypoints/'
    
    # mkdir(excel_path, base_dir)



    old_dir="C:/Users/DS/Desktop/newfolder/10/"
    new_dir="C:/Users/DS/Desktop/kimsihyun/Communication_Bridge/ecolink_ai/datas/keypoints/"
    excel_path = 'C:/Users/DS/Desktop/kimsihyun/Communication_Bridge/ecolink_ai/datas/dirnms.xlsx'

    mvfile(old_dir, new_dir, excel_path)


