from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import openai
import replicate
import os
import uuid
from dotenv import load_dotenv
import boto3
from botocore.exceptions import BotoCoreError, NoCredentialsError
import logging
import requests
from datetime import datetime

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Load environment variables
load_dotenv()

# OpenAI API 설정
OPENAI_API_KEY = os.getenv('OPENAI_API_KEY')
if not OPENAI_API_KEY:
    raise ValueError("OPENAI_API_KEY 환경 변수가 설정되지 않았습니다.")
openai.api_key = OPENAI_API_KEY

# Replicate API 설정
REPLICATE_API_TOKEN = os.getenv('REPLICATE_API_TOKEN')
if not REPLICATE_API_TOKEN:
    raise ValueError("REPLICATE_API_TOKEN 환경 변수가 설정되지 않았습니다.")

# AWS S3 설정
AWS_ACCESS_KEY_ID = os.getenv('AWS_ACCESS_KEY_ID')
AWS_SECRET_ACCESS_KEY = os.getenv('AWS_SECRET_ACCESS_KEY')
AWS_S3_BUCKET_NAME = os.getenv('AWS_S3_BUCKET_NAME')
AWS_S3_REGION = os.getenv('AWS_S3_REGION')  # 기본 리전을 설정하세요

if not all([AWS_ACCESS_KEY_ID, AWS_SECRET_ACCESS_KEY, AWS_S3_BUCKET_NAME, AWS_S3_REGION]):
    raise ValueError("AWS S3 관련 환경 변수가 모두 설정되지 않았습니다.")

s3_client = boto3.client(
    's3',
    region_name=AWS_S3_REGION,
    aws_access_key_id=AWS_ACCESS_KEY_ID,
    aws_secret_access_key=AWS_SECRET_ACCESS_KEY
)

app = FastAPI(
    title="MusicGen AI API",
    description="일기 기반 음악 생성 및 S3 업로드 API 서버입니다.",
    version="1.0"
)

# Define data models
class MusicRequest(BaseModel):
    memberId: int        # AI 컨텐츠 생성 완료 시 알림을 보낼 회원 ID
    date: str            # 일기 작성 날짜 (YYYY-MM-DD)
    content: str         # 일기 내용
    emotion: str         # 감정 정보 (BGM 장르 선정에 사용)

class MusicResponse(BaseModel):
    memberId: int
    date: str
    bgmUrl: str

def extract_music_prompt(content: str, emotion: str) -> str:
    """
    사용자의 일기 내용과 감정을 기반으로 음악 생성 프롬프트를 추출하는 함수
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
                {"role": "user", "content": f"다음 일기 내용을 기반으로 음악 생성에 사용할 프롬프트를 작성해줘.\n\n일기: {content}\n감정: {emotion}"}
            ],
            max_tokens=200,
            temperature=0.7,
        )
        prompt = response.choices[0].message['content'].strip()
        logger.info(f"생성된 프롬프트: {prompt}")
        return prompt
    except Exception as e:
        logger.error(f"프롬프트 생성 오류: {e}")
        raise HTTPException(status_code=500, detail=f"프롬프트 생성 오류: {str(e)}")

def generate_music_with_replicate(prompt: str) -> bytes:
    """
    Replicate의 MusicGen 모델을 사용하여 음악을 생성하는 함수
    Returns the bytes of the generated music file.
    """
    try:
        output = replicate.run(
            "meta/musicgen:671ac645ce5e552cc63a54a2bbff63fcf798043055d2dac5fc9e36a837eedcfb",
            input={
                "prompt": prompt,
                "duration": 15,            # 15초
                "output_format": "mp3",    # mp3로 유지
                "normalization_strategy": "peak"
            },
            api_token=REPLICATE_API_TOKEN
        )
        logger.info(f"Replicate output type: {type(output)}")
        logger.info(f"Replicate output: {output}")

        if hasattr(output, 'read'):
            # File-like object, read bytes
            music_bytes = output.read()
            logger.info(f"음악 생성 완료. 파일 데이터 읽음. 크기: {len(music_bytes)} bytes")
            return music_bytes
        elif isinstance(output, str):
            # If output is a URL string, download the file
            logger.info(f"음악 생성 완료. 출력 URL: {output}")
            response = requests.get(output)
            response.raise_for_status()
            logger.info("음악 파일 다운로드 성공.")
            return response.content
        else:
            logger.error(f"Unexpected output type from Replicate: {type(output)}")
            raise HTTPException(status_code=500, detail="Unexpected output type from Replicate")
    except replicate.exceptions.ApiException as e:
        logger.error(f"Replicate API 오류: {e}")
        raise HTTPException(status_code=500, detail=f"Replicate API 오류: {str(e)}")
    except Exception as e:
        logger.error(f"음악 생성 오류: {e}")
        raise HTTPException(status_code=500, detail=f"음악 생성 오류: {str(e)}")

def upload_to_s3(file_content: bytes, member_id: int, date_str: str) -> str:
    """
    파일 내용을 S3에 업로드하고 URL을 반환하는 함수
    """
    if not file_content:
        logger.error("파일 내용이 비어 있습니다.")
        raise HTTPException(status_code=500, detail="파일 내용이 비어 있습니다.")

    try:
        # 날짜 문자열을 datetime 객체로 변환
        try:
            date_obj = datetime.strptime(date_str, "%Y-%m-%d")
        except ValueError:
            logger.error("날짜 형식이 올바르지 않습니다. YYYY-MM-DD 형식을 사용하세요.")
            raise HTTPException(status_code=400, detail="날짜 형식이 올바르지 않습니다. YYYY-MM-DD 형식을 사용하세요.")

        year = date_obj.strftime("%Y")
        month = date_obj.strftime("%m")
        day = date_obj.strftime("%d")

        # 파일 경로 생성: music-ai/memberId/yyyy/MM/dd/bgm.mp3
        file_key = f"music-ai/{member_id}"

        # S3에 파일 업로드 (ACL 제거)
        s3_client.put_object(
            Bucket=AWS_S3_BUCKET_NAME,
            Key=file_key,
            Body=file_content,
            ContentType='audio/mpeg'
            # ACL 제거
        )
        # S3 파일 URL 생성
        bgm_url = f"https://{AWS_S3_BUCKET_NAME}.s3.{AWS_S3_REGION}.amazonaws.com/{file_key}/{year}/{month}/{day}/bgm.mp3"
        logger.info(f"S3 업로드 완료: {bgm_url}")
        return bgm_url

    except (BotoCoreError, NoCredentialsError) as e:
        logger.error(f"S3 업로드 오류: {e}")
        raise HTTPException(status_code=500, detail=f"S3 업로드 오류: {str(e)}")

@app.post("/music-ai", response_model=MusicResponse)
def create_music(request: MusicRequest):
    """
    BGM 생성 요청을 받아 음악을 생성하고 S3에 업로드한 후 URL을 반환하는 엔드포인트
    """
    try:
        member_id = request.memberId
        date_str = request.date
        content = request.content
        emotion = request.emotion

        logger.info(f"Received request: memberId={member_id}, date={date_str}, emotion={emotion}")

        # 프롬프트 생성
        prompt = extract_music_prompt(content, emotion)

        # 음악 생성
        music_content = generate_music_with_replicate(prompt)
        if not music_content:
            logger.error("음악 생성 후 파일 내용이 비어 있습니다.")
            raise HTTPException(status_code=500, detail="음악 생성 후 파일 내용이 비어 있습니다.")

        # S3에 업로드
        bgm_url = upload_to_s3(music_content, member_id, date_str)

        # 응답 생성
        response = MusicResponse(
            memberId=member_id,
            date=date_str,
            bgmUrl=bgm_url
        )

        logger.info(f"Response: {response}")

        return response

    except HTTPException as http_exc:
        raise http_exc
    except Exception as e:
        logger.error(f"Unhandled error: {e}")
        raise HTTPException(status_code=500, detail=str(e))