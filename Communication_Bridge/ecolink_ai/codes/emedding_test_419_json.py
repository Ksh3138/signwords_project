import torch
import json
from models.st_gcn_18 import STGCNModel 
import numpy as np

label_list=["0", "1", "2", "3", "4", "5", "6", "7", "8", "9", "10", "11", "12", "13", "14", "15", "16", "17", "18", "19", "20", "21", "22", "23", "24", "25", "26", "27", "28", "29", "30", "31", "32", "33", "34", "35", "36", "37", "38", "39", "40", "41", "42", "43", "44", "45", "46", "47", "48", "49", "50", "51", "52", "53", "54", "55", "56", "57", "58", "59", "60", "61", "62", "63", "64", "65", "66", "67", "68", "69", "70", "71", "72", "73", "74", "75", "76", "77", "78", "79", "80", "81", "82", "83", "84", "85", "86", "87", "88", "89", "90", "91", "92", "93", "94", "95", "96", "97", "98", "99", "100", "112", "119", "1000", "10000", "가렵다", "가스", "가슴", "가시", "각목", "갇히다", "감금", "감전", "강", "강남구", "강동구", "강북구", "강서구", "강풍", "개", "거실", "걸렸다", "결박", "경운기", "경찰", "경찰차", "계곡", "계단", "고속도로", "고압전선", "고열", "고장", "골절", "곰", "공사장", "공원", "공장", "관악구", "광진구", "교통사고", "구급대", "구급대원", "구급차", "구로구", "구청", "구해주세요", "귀", "금가다", "금요일", "금천구", "급류", "기절", "기절하다", "깔리다", "끓는물", "남자친구", "남편", "남학생", "납치", "낫", "낯선남자", "낯선사람", "낯선여자", "내년", "내일", "냄새나다", "노원구", "논", "놀이터", "농약", "누나", "누수", "누전", "누출", "눈", "다리", "다음", "달(월)", "대문앞", "도둑", "도로", "도봉구", "독극물", "독버섯", "독사", "동대문구", "동생", "동작구", "동전", "두드러기생기다", "뒤", "뒤통수", "등", "딸", "떨어지다", "뜨거운물", "마당", "마포구", "말려주세요", "말벌", "맹견", "머리", "멧돼지", "목", "목요일", "무너지다", "무릎", "문틈", "물", "밑에", "바다", "반점생기다", "발", "발가락", "발목", "발작", "방망이", "밭", "배", "배고프다", "뱀", "벌", "범람", "벼락", "병원", "보건소", "보내주세요(경찰)", "보내주세요(구급차)", "복부", "복통", "볼", "부러지다", "부엌", "불", "불나다", "붕괴", "붕대", "비닐하우스", "비상약", "빌라", "뼈", "사이", "산", "살충제", "살해", "삼키다", "서대문구", "서랍", "서울시", "서초구", "선반", "선생님", "성동구", "성북구", "성폭행", "소방관", "소방차", "소화기", "소화전", "손", "손가락", "손목", "송파구", "수영장", "수요일", "술취한 사람", "숨을안쉬다", "시청", "신고하세요(경찰)", "심 장마비", "쓰러지다", "아기", "아내", "아들", "아래", "아빠", "아이들", "아저씨", "아줌마", "아파트", "인대", "안방", "알려주세요", "앞", "앞집", "약국", "약사", "양천구", "어깨", "어린이", "어제", "어지러움", "언니", "얼굴", "엄마", "엘리베이터", "여자친구", "여학생", "연기", "연락해주세요", "열", "열나다", "열어주세요", "엽총", "영등포구", "옆집", "옆집 아저씨", "옆집 할아버지", "옆집사람", "옆쪽", "오늘", "오른쪽", "오른쪽-귀", "오른쪽-눈", "오빠", "옥상", "올해", "왼쪽", "왼쪽-귀", "왼쪽-눈", "욕실", "용산구", "우리집", "운동장", "월요일", "위", "위에", "위협", "윗집", "윗집사람", "유리", "유치원", "유치원 버스", "은평구", "음식물", "응급대원", " 응급처리", "의사", "이마", "이물질", "이번", "이상한사람", "이웃집", "일요일", "임산부", "임신한아내", "자동차", "자살", "자상", "작년", "작은방", "장난감", "장단지", "절단", "절도", "제초제", "조난", "종로구", "주", "중구", "중랑구", "지난", "지혈대", "진통제", "질식", "집", "집단폭행", "차밖", "차안", "창문", "창백하다", "체온계", "총", "추락", "축사", "출산", "출혈", "친구", "침수", "칼", "코", "탈골", "택시", "토요일", "토하다", "통학버스", "트랙터", "트럭", "파도", "파편", "팔", "팔꿈치", "폭발", "폭우", "폭탄", "폭행", "피나다", "학교", "학생", "할머니", "할아버지", "함몰되다", "해(연)", "해독제", "해열제", "허리", "허벅지", "현관", "현관앞", "협박", "형", "호흡곤란", "호흡기", "홍수", "화상", "화약", "화요일", "화장실", "화재"] 


def process_keypoints(frames_json, fixed_length=60):
    num_pose = 33
    num_hand = 21
    num_nodes = num_pose + 2 * num_hand

    keypoints_all = []
    for frame in frames_json:
        pose = frame.get('pose_landmarks') or []
        lhand = frame.get('left_hand_landmarks') or []
        rhand = frame.get('right_hand_landmarks') or []

        frame_kps = []
        for i in range(num_pose):
            if i < len(pose):
                kp = pose[i]
                frame_kps.append([kp['x'], kp['y'], kp['z'], kp['visibility']])
            else:
                frame_kps.append([0, 0, 0, 0])
        for i in range(num_hand):
            if i < len(lhand):
                kp = lhand[i]
                frame_kps.append([kp['x'], kp['y'], kp['z'], kp['visibility']])
            else:
                frame_kps.append([0, 0, 0, 0])
        for i in range(num_hand):
            if i < len(rhand):
                kp = rhand[i]
                frame_kps.append([kp['x'], kp['y'], kp['z'], kp['visibility']])
            else:
                frame_kps.append([0, 0, 0, 0])
        keypoints_all.append(frame_kps)

    T = len(keypoints_all)
    if T < fixed_length:
        padding = [[0, 0, 0, 0]] * num_nodes
        for _ in range(fixed_length - T):
            keypoints_all.append(padding)
    elif T > fixed_length:
        keypoints_all = keypoints_all[:fixed_length]

    arr = np.array(keypoints_all).transpose(2, 0, 1)
    return torch.tensor(arr, dtype=torch.float32).unsqueeze(0)  # (1, C, T, V)

def test(model_path, json_path):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = STGCNModel(in_channels=4, num_class=419)
    checkpoint = torch.load(model_path, map_location=device)
    model.load_state_dict(checkpoint['model_state_dict'])
    model.to(device)
    model.eval()

    with open(json_path, 'r') as f:
        json_data = json.load(f)

    x = process_keypoints(json_data)
    x = x.to(device).contiguous()

    with torch.no_grad():
        output = model(x)
        pred = torch.argmax(output, dim=1).item()
        probs = torch.softmax(output, dim=1)
    print(f"인식된 단어: {label_list[pred]}")






if __name__ == "__main__":
    model_path = "C:/Users/DS/Desktop/kimsihyun/Communication_Bridge/ecolink_ai/datas/best_model_checkpoint_419.pth"
    json_path = "C:/Users/DS/Desktop/kimsihyun/Communication_Bridge/ecolink_ai/datas/KETI_SL_0000010899.json"

    test(model_path, json_path) 
