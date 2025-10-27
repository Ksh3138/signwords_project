import os
import shutil

old_dir = "C:/Users/DS/Desktop/kimsihyun/Communication_Bridge/ecolink_ai/datas/keypoints/"
new_dir = "C:/Users/DS/Desktop/newfolder/4"


# old_dir 내부에 있는 폴더들 순회
for folder_name in os.listdir(old_dir):
    folder_path = os.path.join(old_dir, folder_name)

    # 폴더만 처리
    if os.path.isdir(folder_path):
        files = [f for f in os.listdir(folder_path)
                 if os.path.isfile(os.path.join(folder_path, f))]

        # 파일이 3개 이상인 경우에만 진행
        if len(files) >= 4:
            files.sort()  # 오름차순 정렬

            third_file = files[3]  # 3번째 파일
            src = os.path.join(folder_path, third_file)
            dst = os.path.join(new_dir, third_file)

            # 같은 이름의 파일이 존재하면 덮어쓰지 않고 이름 변경도 가능
            if os.path.exists(dst):
                base, ext = os.path.splitext(third_file)
                count = 1
                while os.path.exists(dst):
                    dst = os.path.join(new_dir, f"{base}_{count}{ext}")
                    count += 1

            shutil.move(src, dst)
            print(f"Moved: {src} -> {dst}")
        else:
            print(f"Skip (only {len(files)} files): {folder_path}")
