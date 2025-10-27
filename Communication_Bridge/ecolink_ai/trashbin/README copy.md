# Communication_Bridge

videoTest_mediapipe.py
1. pip install mediapipe
2. if __name__ == "__main__": 에서 cap과 video_out, json_out 경로 변경

create_dataset.py
1. 영상 준비 
2. actions, action_dir 변경
3. save 경로 변경

경로예시
datas
  ㄴ video -- 계단 -- (계단 동작 영상파일)
  |        ㄴ 화재 -- (화재 동작 영상파일)
  |        ㄴ ...     
  ㄴ dataset -- (.npy 결과파일)