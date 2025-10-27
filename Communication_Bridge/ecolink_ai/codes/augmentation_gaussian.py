import numpy as np
import os
import json


def add_gaussian_noise_to_landmarks(data, pose_noise=0.01, hand_noise=0.005):
    for frame_data in data:
        for landmark_type in ['pose_landmarks']:
            if landmark_type in frame_data and frame_data[landmark_type] is not None:
                for landmark in frame_data[landmark_type]:
                    landmark['x'] += np.random.normal(0, pose_noise)
                    landmark['y'] += np.random.normal(0, pose_noise)
                    landmark['z'] += np.random.normal(0, pose_noise)

    for frame_data in data:
        for landmark_type in ['left_hand_landmarks', 'right_hand_landmarks']:
            if landmark_type in frame_data and frame_data[landmark_type] is not None:
                for landmark in frame_data[landmark_type]:
                    landmark['x'] += np.random.normal(0, hand_noise)
                    landmark['y'] += np.random.normal(0, hand_noise)
                    landmark['z'] += np.random.normal(0, hand_noise)


    return data





if __name__ == "__main__":
    # 폴더 내 모든 파일 처리
    input_dir = 'C:/Users/DS/Desktop/kimsihyun/Communication_Bridge/ecolink_ai/final_datas/2_normali_12'
    output_dir = 'C:/Users/DS/Desktop/kimsihyun/Communication_Bridge/ecolink_ai/final_datas/3_gaussi_12'

    for root, dirs, files in os.walk(input_dir):
        for file in files:
            if not file.endswith('.json'):
                continue
            input_file = os.path.join(root, file).replace('\\', '/')
            rel_path = os.path.relpath(input_file, input_dir).replace('\\', '/')
            base_name, ext = os.path.splitext(os.path.basename(rel_path))  # 파일명만 분리
            output_folder = os.path.join(output_dir, os.path.dirname(rel_path)).replace('\\', '/')

            # 원본 파일 로드
            with open(input_file, 'r', encoding='utf-8') as f:
                json_data = json.load(f)

            augment_count = 5
            for i in range(1, augment_count + 1):
                augmented_data = add_gaussian_noise_to_landmarks(json_data.copy())
                output_file = os.path.join(output_folder, f"{base_name}_{i}.json").replace('\\', '/')

                # 상위 디렉터리 생성 (output_file의 상위 폴더)
                os.makedirs(os.path.dirname(output_file), exist_ok=True)

                # 저장
                with open(output_file, 'w', encoding='utf-8') as f_out:
                    json.dump(augmented_data, f_out, ensure_ascii=False, indent=2)




    # # 파일 하나 처리
    # input_file="C:/Users/DS/Desktop/kimsihyun/Communication_Bridge/ecolink_ai/trashbin/test_try.json"

    # folder = os.path.dirname(input_file)
    # base_name, ext = os.path.splitext(os.path.basename(input_file))

    # with open(input_file, 'r', encoding='utf-8') as f:
    #     json_data = json.load(f)

    # n=5 #증강 횟수
    # for i in range(1, n+1):
    #     augmented_data = add_gaussian_noise_to_landmarks(json_data.copy())
    #     output_file = os.path.join(folder, f"{base_name}_{i}{ext}")  # 폴더 고정, 파일명만 변화

    #     with open(output_file, 'w', encoding='utf-8') as f_out:
    #         json.dump(augmented_data, f_out, ensure_ascii=False, indent=2)

