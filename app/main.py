from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import openai
from transformers import pipeline
import os
import uuid
from dotenv import load_dotenv
import soundfile as sf
import numpy as np
import logging
import torch

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Load environment variables
load_dotenv()
OPENAI_API_KEY = os.getenv('OPENAI_API_KEY')

if not OPENAI_API_KEY:
    raise ValueError("OPENAI_API_KEY 환경 변수가 설정되지 않았습니다.")


# Set OpenAI API key
openai.api_key = OPENAI_API_KEY

app = FastAPI(
    title="MusicGen AI API",
    description="일기 기반 음악 생성 API 서버입니다.",
    version="1.0"
)


# Define data models
class DiaryEntry(BaseModel):
    diary: str


class MusicResponse(BaseModel):
    file_urls: list[str]


# Load pipeline once at startup
try:
    pipe = pipeline("text-to-audio", model="facebook/musicgen-melody")
    logger.info("MusicGen pipeline loaded successfully.")
except Exception as e:
    logger.error(f"Error loading MusicGen pipeline: {e}")
    raise e

# Test numpy availability
try:
    np.array([1, 2, 3])
    logger.info("Numpy is available.")
except ImportError:
    logger.error("Numpy is not available.")
    raise


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
                                추출한 내용들로 프롬프트를 작성합니다.
                                만약, 우울한 내용이라면 응원하는 느낌의 음원으로 만들어줍니다. 용기를 줄 수 있는 분위기의 음악 혹은 위로해줄 수 있는 느낌의 음악"""},
                {"role": "user", "content": f"다음 일기 내용을 기반으로 음악 생성에 사용할 프롬프트를 작성해줘.\n\n일기: {diary_entry}"}
            ],
            max_tokens=200,
            temperature=0.7,
        )
        prompt = response.choices[0].message['content'].strip()
        return prompt
    except Exception as e:
        logger.error(f"프롬프트 생성 오류: {e}")
        raise HTTPException(status_code=500, detail=f"프롬프트 생성 오류: {str(e)}")


def generate_and_save_music(prompts: list[str], output_dir='generated_music_files') -> list[str]:
    """
    주어진 프롬프트 리스트를 기반으로 음악을 생성하고 파일로 저장하는 함수
    """
    try:
        # Generate audio using the pipeline with duration=15 seconds
        audio_outputs = pipe(prompts)

        # 디버깅을 위해 출력 형식 로그
        logger.info(f"audio_outputs type: {type(audio_outputs)}")
        logger.info(f"audio_outputs content: {audio_outputs}")

        # Create output directory if it doesn't exist
        os.makedirs(output_dir, exist_ok=True)

        # Save each generated audio to a file
        file_urls = []
        for idx, audio in enumerate(audio_outputs):
            logger.info(f"Processing audio {idx + 1}: {audio}")

            # 예상되는 키 확인 및 데이터 추출
            if isinstance(audio, dict):
                if 'audio' in audio and 'sampling_rate' in audio:
                    data = audio['audio']
                    sampling_rate = audio['sampling_rate']
                elif 'array' in audio and 'sampling_rate' in audio:
                    data = audio['array']
                    sampling_rate = audio['sampling_rate']
                elif 'samples' in audio and 'sampling_rate' in audio:
                    data = audio['samples']
                    sampling_rate = audio['sampling_rate']
                else:
                    logger.error("Unexpected audio format: missing 'audio', 'array', or 'samples' key.")
                    raise KeyError("Unexpected audio format: missing 'audio', 'array', or 'samples' key.")
            elif isinstance(audio, (np.ndarray, torch.Tensor)):
                data = audio.cpu().numpy() if isinstance(audio, torch.Tensor) else audio
                sampling_rate = 32000  # 모델에 따라 샘플링 레이트 조정
            else:
                logger.error(f"Unsupported audio format: {type(audio)}")
                raise TypeError(f"Unsupported audio format: {type(audio)}")

            # 데이터의 차원 확인 및 축소
            if data.ndim == 3:
                data = data.squeeze()  # (1, 1, 480000) -> (480000,)
            elif data.ndim == 2 and data.shape[0] == 1:
                data = data.squeeze(0)  # (1, 480000) -> (480000,)
            elif data.ndim == 2:
                pass  # 이미 (채널, 샘플) 형태
            else:
                logger.error(f"Unexpected data shape: {data.shape}")
                raise ValueError(f"Unexpected data shape: {data.shape}")

            # 샘플링 레이트 확인
            if sampling_rate <= 0:
                logger.error(f"Invalid sampling rate: {sampling_rate}")
                raise ValueError(f"Invalid sampling rate: {sampling_rate}")

            # 원하는 길이 설정 (15초)
            desired_duration = 15  # 초
            desired_samples = desired_duration * sampling_rate  # 샘플 수

            # 현재 샘플 수 확인
            current_samples = data.shape[-1]
            if current_samples < desired_samples:
                logger.warning(
                    f"Generated audio is shorter than desired duration: {current_samples / sampling_rate:.2f}s")
                # 필요에 따라 패딩을 추가할 수 있습니다.
            else:
                # 오디오 데이터 슬라이싱 (처음 15초)
                data = data[..., :desired_samples]
                logger.info(f"Trimmed audio to {desired_duration} seconds.")

            # Generate unique filename
            filename = f"generated_music_{uuid.uuid4()}.wav"
            output_path = os.path.join(output_dir, filename)

            # Save audio using soundfile
            try:
                sf.write(output_path, data, sampling_rate)
                logger.info(f"Music saved to {output_path}")
            except Exception as e:
                logger.error(f"Failed to save audio to {output_path}: {e}")
                raise e

            # Append the file URL
            file_url = f"/files/{filename}"
            file_urls.append(file_url)

        return file_urls

    except Exception as e:
        logger.error(f"음악 생성 오류: {e}")
        raise HTTPException(status_code=500, detail=f"음악 생성 오류: {str(e)}")


@app.post("/generate_music", response_model=MusicResponse)
def create_music(entries: list[DiaryEntry]):
    """
    여러 개의 일기 내용을 받아 음악을 생성하고 파일 URL 리스트를 반환하는 엔드포인트
    """
    try:
        diary_entries = [entry.diary for entry in entries]
        logger.info(f"받은 일기들: {diary_entries}")

        # 일기에서 프롬프트 추출
        prompts = [extract_music_prompt(entry) for entry in diary_entries]
        logger.info(f"생성된 프롬프트들: {prompts}")

        # 음악 생성 및 파일 저장
        file_urls = generate_and_save_music(prompts)

        return MusicResponse(file_urls=file_urls)

    except HTTPException as http_exc:
        raise http_exc
    except Exception as e:
        logger.error(f"Unhandled error: {e}")
        raise HTTPException(status_code=500, detail=str(e))


# 정적 파일 서빙 설정 (예: generated_music_files 디렉토리)
from fastapi.staticfiles import StaticFiles
app.mount("/files", StaticFiles(directory="generated_music_files"), name="files")