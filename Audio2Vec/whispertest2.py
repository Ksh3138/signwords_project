from kiwipiepy import Kiwi
from transformers import AutoModelForSpeechSeq2Seq, AutoProcessor, pipeline
import torch
import torchaudio

# Whisper 모델과 프로세서 로드
model_id = "C:\\Users\\DS\\Desktop\\kimsihyun\\model"
model = AutoModelForSpeechSeq2Seq.from_pretrained(model_id)
processor = AutoProcessor.from_pretrained(model_id)

device = "cuda:0" if torch.cuda.is_available() else "cpu"
torch_dtype = torch.float16 if torch.cuda.is_available() else torch.float32
model.to(device)

# Kiwi 객체 생성
kiwi = Kiwi()

# Whisper 파이프라인 설정
pipe = pipeline(
    "automatic-speech-recognition",
    model=model,
    tokenizer=processor.tokenizer,
    feature_extractor=processor.feature_extractor,
    torch_dtype=torch_dtype,
    device=device,
)

# 오디오 파일 로드
path_to_audio = "C:\\Users\\DS\\Desktop\\kimsihyun\\audio\\testaudio.wav"
waveform, sample_rate = torchaudio.load(path_to_audio)

# Whisper 모델에서 사용할 수 있는 16,000Hz로 샘플링 레이트 변환
if sample_rate != 16000:
    waveform = torchaudio.transforms.Resample(orig_freq=sample_rate, new_freq=16000)(waveform)
    sample_rate = 16000

# 오디오 데이터 Whisper 모델로 처리
inputs = processor(waveform.squeeze().numpy(), sampling_rate=sample_rate, return_tensors="pt", language="en").to(device)
inputs["input_features"] = inputs["input_features"].to(dtype=torch_dtype)

# 모델 추론
with torch.no_grad():
    generated_tokens = model.generate(inputs["input_features"])

# 텍스트로 변환
transcription = processor.batch_decode(generated_tokens, skip_special_tokens=True)[0]

# 형태소 분석 (Kiwi 사용)
results = kiwi.analyze(transcription)

# 분석 결과 출력
for sentence in results:
    for token in sentence[0]:
        print(f"{token.form}\t{token.tag}")
