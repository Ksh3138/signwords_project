import numpy as np
import json
import os
from pathlib import Path
from datetime import datetime
import torch
from st_gcn_master.models.st_gcn_18 import ST_GCN_18 
# from models.st_gcn_18 import ST_GCN_18 

# 1. 파일 오픈
def load_keypoint_data(json_path):
    with open(json_path, 'r') as f:
        data = json.load(f)
    return data['keypoints']  #keypoints만 추출
 
# 2. keypoints 데이터를 모델 입력층에 맞도록 PyTorch 텐서로 만들기
def preprocess_keypoints(normalized_keypoints):
    keypoints_np = np.array(normalized_keypoints)  #numpy배열로 바꾸기
    keypoints_np = keypoints_np.transpose(2, 0, 1) #shape 순서바꾸기: (T, V, C) → (C, T, V) 
    keypoints_tensor = torch.FloatTensor(keypoints_np).unsqueeze(0)  #shape에 배치 차원 추가: (N, C, T, V)
    # return keypoints_tensor

    if keypoints_tensor.shape[-1] < 137:
        pad_size = 137 - keypoints_tensor.shape[-1]
        pad = torch.zeros(keypoints_tensor.shape[0], keypoints_tensor.shape[1], keypoints_tensor.shape[2], pad_size)
        keypoints_tensor = torch.cat([keypoints_tensor, pad], dim=-1)
    
    return keypoints_tensor



# 모델 준비
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model = ST_GCN_18(in_channels=3, num_class=256).to(device)
    
# 체크포인트 (=이전 학습 결과)
checkpoint_path = "C:/Users/DS/Desktop/kimsihyun/my/st_gcn_master/checkpoints/modified_stgcn_init.pth"
checkpoint = torch.load(checkpoint_path, map_location=device)
state_dict = {
    k: v for k, v in checkpoint['state_dict'].items() 
    if 'fcn' not in k  # fcn 레이어 제외
}
model.load_state_dict(state_dict, strict=False) #중에서 가중치만 가져옴

model.eval()

# 3. st-gcn 모델 준비, 임베딩 수행
def create_stgcn_embedding(keypoints):

    with torch.no_grad():
        keypoints = keypoints.to(device)
        embedding = model(keypoints) #모델에 입력
    
    return embedding.cpu().numpy() #numpy 배열로 

# 4. 결과 저장
def save_embedding(embedding, metadata, output_path):
    output_data = {
        "metadata": metadata,
        "embedding": embedding.flatten().tolist(),
        "created_at": datetime.now().isoformat()
    }
    
    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(output_data, f, ensure_ascii=False, indent=2)


# 정규화한 벡터를 256차원의 고정 길이로 변환
def vector_embedding(input_file, output_dir):
    input_file = Path(input_file)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    if not input_file.exists() or input_file.suffix.lower() != '.json':
        return
    
    print(f"\n임베딩 시작 {input_file.name}...")
    
    try:
        keypoint_data = load_keypoint_data(input_file)
        keypoints = preprocess_keypoints(keypoint_data)
        embedding= create_stgcn_embedding(keypoints)

        metadata = {
            "original_file": input_file.name,
            "frame_count": len(keypoint_data),
            "keypoint_info": {
                "body": 25,
                "face": 70,
                "hand_left": 21,
                "hand_right": 21,
                "total": 137
            },
            "channels": 3,
            "embedding_dim": embedding.shape[1],
            "model": "Modified-ST-GCN",
            "model_config": {
                "graph_type": "custom",
                "keypoint_layout": "openpose_extended"
            },
            "device": "cuda" if torch.cuda.is_available() else "cpu"
        }
        output_path = output_dir / f"{input_file.stem}_embedding.json"
        save_embedding(embedding, metadata, output_path)
        print(f"임베딩 완료 {input_file.name} -> {output_path.name}\n")
        
        return str(output_path).replace('\\', '/')

    except Exception as e:
        print(f"create_stgcn_embedding 에러: {e}")
        return None

    

if __name__ == "__main__":  
    input_file = "C:/Users/DS/Desktop/kimsihyun/my/datas/vector/KETI_SL_0000000333_vector.json"
    output_dir = "C:/Users/DS/Desktop/kimsihyun/my/datas/embeddings"
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    # print(vector_embedding(input_file, output_dir))


    input_dir = "C:/Users/DS/Desktop/kimsihyun/my/datas/vector"
    output_root = "C:/Users/DS/Desktop/kimsihyun/my/datas/embeddings"
    for root, dirs, files in os.walk(input_dir):
        for file in files:
            input_file = os.path.join(root, file)
            rel_dir = os.path.relpath(root, input_dir)
            output_dir = os.path.join(output_root, rel_dir)
            print(vector_embedding(input_file, output_dir))
