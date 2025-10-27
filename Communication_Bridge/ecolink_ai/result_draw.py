import json
from matplotlib.pylab import sign
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation


def result_draw(json_file):
    # 데이터 로드(예: JSON 파일에서)
    with open(json_file) as f:
        data = json.load(f)  # data = [{frame:..., pose_landmarks:[...], left_hand_landmarks:[...], right_hand_landmarks:[...]}]

    fig, ax = plt.subplots(figsize=(5, 5))

    def update(frame_idx):
        ax.clear()
        frame_data = data[frame_idx]
        # pose
        pose = frame_data.get("pose_landmarks")
        if pose:
            pose_xy = [(lm['x'], lm['y']) for lm in pose]
            if pose_xy:
                px, py = zip(*pose_xy)
                ax.scatter(px, py, c='b', label="Pose")
        # left hand
        lh = frame_data.get("left_hand_landmarks")
        if lh:
            lh_xy = [(lm['x'], lm['y']) for lm in lh]
            if lh_xy:
                lx, ly = zip(*lh_xy)
                ax.scatter(lx, ly, c='g', label="Left Hand")
        # right hand
        rh = frame_data.get("right_hand_landmarks")
        if rh:
            rh_xy = [(lm['x'], lm['y']) for lm in rh]
            if rh_xy:
                rx, ry = zip(*rh_xy)
                ax.scatter(rx, ry, c='r', label="Right Hand")
        ax.set_xlim(0, 1)
        ax.set_ylim(0, 1)
        ax.invert_yaxis()
        ax.legend()
        ax.set_title(f"Frame {frame_data['frame']}")

    # 전체 프레임 개수만큼 애니메이션 실행
    ani = FuncAnimation(fig, update, frames=len(data), interval=100)

    plt.show()



if __name__ == "__main__":
    json_file ='C:/Users/DS/Desktop/kimsihyun/Communication_Bridge/ecolink_ai/trashbin/test_try_2.json'

    # json_file = 'C:/Users/DS/Downloads/test_내일.json'

    json_file='C:/Users/DS/Desktop/kimsihyun/Communication_Bridge/ecolink_ai/datas/keypoints_18/2_normalization/교통사고/KETI_SL_0000004331.json'
    
    # json_file='C:/Users/DS/Desktop/kimsihyun/Communication_Bridge/ecolink_ai/trashbin/test_moving.json'
    

    json_file="C:/Users/DS/Downloads/test_nor.json"


    json_file='C:/Users/DS/Desktop/test_data/test1_cam.json'

    json_file="C:/Users/DS/Desktop/kimsihyun/Communication_Bridge/ecolink_ai/datas/keypoints_11emer/3_augmentation/숨을안쉬다/sign-data-1760695663711.json"
    result_draw(json_file)