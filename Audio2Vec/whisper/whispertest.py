import torch
from transformers import AutoModelForSpeechSeq2Seq, AutoProcessor, pipeline
import torchaudio #추가함


# 장치 설정 (GPU가 있는 경우 GPU 사용)
device = "cuda:0" if torch.cuda.is_available() else "cpu"
torch_dtype = torch.float16 if torch.cuda.is_available() else torch.float32

# Whisper 모델과 프로세서 로드
model_id = "C:\\Users\\DS\\Desktop\\kimsihyun\\model"
model = AutoModelForSpeechSeq2Seq.from_pretrained(
    model_id, torch_dtype=torch_dtype, low_cpu_mem_usage=True, use_safetensors=True
)
model.to(device)
processor = AutoProcessor.from_pretrained(model_id)

pipe = pipeline(
    "automatic-speech-recognition",
    model=model,
    tokenizer=processor.tokenizer,
    feature_extractor=processor.feature_extractor,
    torch_dtype=torch_dtype,
    device=device,
)

# 로컬 음성 파일 로드
path_to_audio = "C:\\Users\\DS\\Desktop\\kimsihyun\\audio\\testaudio.wav"
waveform, sample_rate = torchaudio.load(path_to_audio)  # anjdjWjfkrh를 torchaudio로 수정


# Whisper 모델에서 사용할 수 있는 16,000Hz로 샘플링 레이트 변환
if sample_rate != 16000:
    waveform = torchaudio.transforms.Resample(orig_freq=sample_rate, new_freq=16000)(waveform)
    sample_rate = 16000


# 변환한 오디오 데이터 사용
inputs = processor(waveform.squeeze().numpy(), sampling_rate=sample_rate, return_tensors="pt", language="en").to(device)

# 입력 자료형 변환
inputs["input_features"] = inputs["input_features"].to(dtype=torch_dtype)

# 모델 추론
with torch.no_grad():
    generated_tokens = model.generate(inputs["input_features"])

# 텍스트로 변환
transcription = processor.batch_decode(generated_tokens, skip_special_tokens=True)
print("Transcription:", transcription[0])

# 백엔드 수동 설정
torchaudio.set_audio_backend("sox_io")
print("현재 백엔드:", torchaudio.get_audio_backend())

# 오디오 파일 로딩
path_to_audio = r"C:\Users\DS\Desktop\kimsihyun\audio\testaudio.wav"
waveform, sample_rate = torchaudio.load(path_to_audio)
print("로딩 성공:", waveform.shape, sample_rate)
