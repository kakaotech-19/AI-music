# app/main.py

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import openai
import musicgen
import numpy as np
import soundfile as sf
import torch
import os
import uuid
from dotenv import load_dotenv
from io import BytesIO

# 환경 변수 로드
load_dotenv()
OPENAI_API_KEY = os.getenv('OPENAI_API_KEY')

if not OPENAI_API_KEY:
    raise ValueError("OPENAI_API_KEY 환경 변수가 설정되지 않았습니다.")

# OpenAI API 키 설정
openai.api_key = OPENAI_API_KEY

app = FastAPI(
    title="MusicGen AI API",
    description="일기 기반 음악 생성 API 서버입니다.",
    version="1.0"
)

# 음악 생성 요청 데이터 모델
class DiaryEntry(BaseModel):
    diary: str

# 음악 생성 응답 데이터 모델
class MusicResponse(BaseModel):
    file_urls: list[str]

def extract_music_prompt(diary_entry: str) -> str:
    """
    사용자의 일기 내용을 기반으로 음악 생성 프롬프트를 추출하는 함수
    """
    try:
        response = openai.ChatCompletion.create(
            model="gpt-4",
            messages=[
                {"role": "system",
                 "content": """당신은 일기 내용을 분석하여 음악 생성에 적합한 프롬프트를 작성하는 전문가입니다.
일기의 내용으로부터 장르와 리듬 및 템포를 추출해냅니다.
(ex, Genre: Rock, EDM, Reggae, Lofi, Classical 등)
(ex, Rhythm & Tempo: Heavy drum break, slow BPM, heavy bang 등)
일기 전반 분위기 및 감정을 읽어냅니다.(ex, Breezy, easygoing, harmonic, organic, energetic 등)
장르, 리듬 및 템포, 분위기 및 감정을 기반으로 어울릴 악기 구성을 추론해냅니다.(ex, Saturated guitars, heavy bass line, electronic guitar solo, ukulele-infused 등)
일기의 전반적인 내용으로부터 특징잡을 수 있는 특징적 요소를 추출해냅니다.(ex, Crazy drum break and fills, environmentally conscious, gentle grooves 등)
추출한 내용들로 프롬프트를 작성합니다."""},
                {"role": "user", "content": f"다음 일기 내용을 기반으로 음악 생성에 사용할 프롬프트를 작성해줘.\n\n일기: {diary_entry}"}
            ],
            max_tokens=200,
            temperature=0.7,
        )
        prompt = response.choices[0].message['content'].strip()
        return prompt
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"프롬프트 생성 오류: {str(e)}")

def generate_and_save_music(prompts: list[str], sample_rate=32000, output_dir='generated_music_files') -> list[str]:
    """
    주어진 프롬프트 리스트를 기반으로 음악을 생성하고 파일로 저장하는 함수
    """
    try:
        # 사전 학습된 모델을 전역으로 불러오기
        device = 'cuda' if torch.cuda.is_available() else 'cpu'
        model = musicgen.MusicGen.get_pretrained('medium', device=device)
        model.set_generation_params(duration=15)  # 필요에 따라 조정 가능

        # 음악 생성
        res = model.generate(prompts, progress=False)

        # 생성된 음악을 파일로 저장하고 파일 URL 리스트를 반환
        file_urls = []
        for idx, audio_tensor in enumerate(res):
            # NumPy 배열로 변환
            audio_np = audio_tensor.cpu().numpy()

            # 데이터 타입과 범위에 맞게 변환
            if audio_np.dtype == np.float32 or audio_np.dtype == np.float64:
                audio_np = np.int16(audio_np * 32767)  # Scale to int16 range

            # Reshape the audio data to a 1D array if it's not already
            if audio_np.ndim > 1:
                audio_np = audio_np.reshape(-1)

            # 파일명 생성
            filename = f"generated_music_{uuid.uuid4()}.wav"
            output_path = os.path.join(output_dir, filename)

            # 디렉토리 생성 (필요 시)
            os.makedirs(output_dir, exist_ok=True)

            # WAV 파일로 저장
            sf.write(output_path, audio_np, sample_rate)
            print(f'음악이 {output_path}에 저장되었습니다.')

            # 파일 URL 생성
            file_url = f"/files/{filename}"
            file_urls.append(file_url)

        return file_urls

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"음악 생성 오류: {str(e)}")

@app.post("/generate_music", response_model=MusicResponse)
def create_music(entries: list[DiaryEntry]):
    """
    여러 개의 일기 내용을 받아 음악을 생성하고 파일 URL 리스트를 반환하는 엔드포인트
    """
    try:
        diary_entries = [entry.diary for entry in entries]
        print(f"받은 일기들: {diary_entries}")

        # 일기에서 프롬프트 추출
        prompts = [extract_music_prompt(entry) for entry in diary_entries]
        print(f"생성된 프롬프트들: {prompts}")

        # 음악 생성 및 파일 저장
        file_urls = generate_and_save_music(prompts, sample_rate=32000, output_dir='generated_music_files')

        return MusicResponse(file_urls=file_urls)

    except HTTPException as http_exc:
        raise http_exc
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

# 정적 파일 서빙 설정 (예: generated_music_files 디렉토리)
from fastapi.staticfiles import StaticFiles
app.mount("/files", StaticFiles(directory="generated_music_files"), name="files")
