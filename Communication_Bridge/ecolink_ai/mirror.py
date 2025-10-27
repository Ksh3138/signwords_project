
import cv2
import os


input_dir="C:/Users/DS/Desktop/kimsihyun/Communication_Bridge/ecolink_ai/datas/test_video_emerplus/old"
output_dir="C:/Users/DS/Desktop/kimsihyun/Communication_Bridge/ecolink_ai/datas/test_video_emerplus/new"

for dirpath, dirnames, filenames in os.walk(input_dir):
    for file in filenames:
        if file.lower().endswith('.mp4') or file.lower().endswith('.avi'):
            input_path = os.path.join(dirpath, file).replace('\\', '/')
            # input_dir 기준 상대 경로 추출
            rel_path = os.path.relpath(input_path, input_dir)
            rel_dir = os.path.dirname(rel_path)
            base, ext = os.path.splitext(os.path.basename(file))
            # output_dir/dir1/asdf_mirrored.mp4 형태로 경로 생성
            output_subdir = os.path.join(output_dir, rel_dir)
            os.makedirs(output_subdir, exist_ok=True)
            output_path = os.path.join(output_subdir, f"{base}_mirrored{ext}").replace('\\', '/')

            cap = cv2.VideoCapture(input_path)
            fourcc = cv2.VideoWriter_fourcc(*'mp4v')
            fps = cap.get(cv2.CAP_PROP_FPS)
            w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
            h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
            out = cv2.VideoWriter(output_path, fourcc, fps, (w, h))

            print(f'{input_path} 처리 시작')
            while True:
                ret, frame = cap.read()
                if not ret:
                    break
                flipped = cv2.flip(frame, 1)
                out.write(flipped)
            print(f'{output_path} 저장 끝')

            cap.release()
            out.release()


