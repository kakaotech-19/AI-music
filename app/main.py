from fastapi import FastAPI, HTTPException, BackgroundTasks
from pydantic import BaseModel
import openai
import replicate
import os
from dotenv import load_dotenv
import boto3
from botocore.exceptions import BotoCoreError, NoCredentialsError
import logging
import requests
from datetime import datetime
import uuid

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# httpx 로깅 레벨 조정
logging.getLogger("httpx").setLevel(logging.WARNING)


# Load environment variables
load_dotenv()

# OpenAI API 설정
OPENAI_API_KEY = os.getenv('OPENAI_API_KEY')
openai.api_key = OPENAI_API_KEY

# Replicate API 설정
REPLICATE_API_TOKEN = os.getenv('REPLICATE_API_TOKEN')

# AWS S3 설정
AWS_ACCESS_KEY_ID = os.getenv('AWS_ACCESS_KEY_ID')
AWS_SECRET_ACCESS_KEY = os.getenv('AWS_SECRET_ACCESS_KEY')
AWS_S3_BUCKET_NAME = os.getenv('AWS_S3_BUCKET_NAME')
AWS_S3_REGION = os.getenv('AWS_S3_REGION')

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
    emotion: str         # 감정 정보 (BGM 분위기)
    genre: str           # 장르 정보 (BGM 장르 선정)
    apiDomainUrl: str           # Spring에서 전달받은 웹훅 IP 주소

class MusicResponse(BaseModel):
    message: str
    taskId: str

def extract_instruments_by_genre(genre: str) -> list:
    """
    장르에 맞는 악기를 추출하는 함수
    """
    try:
        response = openai.ChatCompletion.create(
            model="gpt-4o-mini",
            messages=[
                {"role": "system",
                 "content": f"""
                 당신은 특정 장르에 맞는 악기를 추천하는 전문가입니다.
                 주어진 장르에 잘 어울리는 악기 3개 이상을 추천하세요.
                 장르: {genre}
                 """},
                {"role": "user", "content": f"장르에 어울리는 악기를 추천해줘: {genre}"}
            ],
            max_tokens=500,
            temperature=0.5,
        )
        instruments_text = response.choices[0].message['content'].strip()
        instruments = instruments_text.split(':')[1].strip().split(', ')
        logger.info(f"장르 기반 추출된 악기: {instruments}")
        return instruments
    except Exception as e:
        logger.error(f"장르 기반 악기 추출 오류: {e}")
        raise HTTPException(status_code=500, detail=f"장르 기반 악기 추출 오류: {str(e)}")

def extract_instruments_by_emotion_and_content(emotion: str, content: str) -> list:
    """
    감정과 일기 내용에 맞는 악기를 추출하는 함수
    """
    try:
        response = openai.ChatCompletion.create(
            model="gpt-4o-mini",
            messages=[
                {"role": "system",
                 "content": f"""
                 당신은 특정 감정과 일기 내용을 기반으로 음악에 사용할 악기를 추천하는 전문가입니다.
                 주어진 감정과 일기 내용에 어울리는 악기 3개 이상을 추천하세요.
                 감정: {emotion}
                 일기 내용: {content}
                 """},
                {"role": "user", "content": f"다음 감정과 일기 내용을 기반으로 악기를 추천해줘:\n감정: {emotion}\n일기 내용: {content}"}
            ],
            max_tokens=500,
            temperature=0.5,
        )
        instruments_text = response.choices[0].message['content'].strip()
        instruments = instruments_text.split(':')[1].strip().split(', ')
        logger.info(f"감정 및 일기 내용 기반 추출된 악기: {instruments}")
        return instruments
    except Exception as e:
        logger.error(f"감정 및 일기 내용 기반 악기 추출 오류: {e}")
        raise HTTPException(status_code=500, detail=f"감정 및 일기 내용 기반 악기 추출 오류: {str(e)}")

def generate_music_with_replicate(prompt: str) -> bytes:
    """
    Replicate API를 사용하여 음악을 생성하는 함수
    """
    try:
        output = replicate.run(
            "meta/musicgen:671ac645ce5e552cc63a54a2bbff63fcf798043055d2dac5fc9e36a837eedcfb",
            input={
                "prompt": prompt,
                "duration": 15,
                "output_format": "mp3"
            }
        )
        response = requests.get(output)  # 생성된 음악 파일 URL에서 파일 다운로드
        response.raise_for_status()
        logger.info("음악 파일 생성 및 다운로드 성공")
        return response.content
    except Exception as e:
        logger.error(f"음악 생성 실패: {e}")
        raise HTTPException(status_code=500, detail="음악 생성 실패")

def upload_to_s3(file_content: bytes, member_id: int, date_str: str) -> str:
    """
    음악 파일을 S3에 업로드하고 파일의 URL을 반환하는 함수
    """
    try:
        date = datetime.strptime(date_str, "%Y-%m-%d")
        file_key = f"music-ai/{member_id}/{date.year}/{date.month}/{date.day}/bgm.mp3"

        s3_client.put_object(
            Bucket=AWS_S3_BUCKET_NAME,
            Key=file_key,
            Body=file_content,
            ContentType='audio/mpeg'
        )
        bgm_url = f"https://{AWS_S3_BUCKET_NAME}.s3.{AWS_S3_REGION}.amazonaws.com/{file_key}"
        logger.info(f"S3 업로드 성공: {bgm_url}")
        return bgm_url
    except Exception as e:
        logger.error(f"S3 업로드 오류: {e}")
        raise HTTPException(status_code=500, detail="S3 업로드 실패")

def background_music_process(request: MusicRequest):
    """
    백그라운드에서 음악을 생성하고 S3에 업로드한 후 웹훅으로 결과를 전송하는 함수
    """
    try:
        member_id = request.memberId
        date_str = request.date
        content = request.content
        webhook_url = f"http://{request.apiDomainUrl}/webhook/ai/bgm"  # apiDomainUrl 기반 동적 웹훅 URL
        genre = request.genre
        emotion = request.emotion

        logger.info(f"백그라운드 작업 시작: memberId={member_id}, date={date_str}, genre={genre}, emotion={emotion}")

        # 장르와 감정 및 일기 내용 기반 악기 추출
        instruments_by_genre = extract_instruments_by_genre(genre)
        instruments_by_emotion_and_content = extract_instruments_by_emotion_and_content(emotion, content)

        # 두 결과 병합 및 중복 제거
        instruments = list(set(instruments_by_genre + instruments_by_emotion_and_content))
        logger.info(f"최종 추출된 악기 목록: {instruments}")

        # 프롬프트 생성
        prompt = f"장르: {genre}, 감정: {emotion}, 일기 내용: {content}, 악기: {', '.join(instruments)}"

        # 음악 생성
        music_content = generate_music_with_replicate(prompt)

        # S3 업로드
        bgm_url = upload_to_s3(music_content, member_id, date_str)

        # 웹훅으로 성공 알림 전송
        send_webhook(webhook_url, {
            "memberId": member_id,
            "date": date_str,
            "bgmUrl": bgm_url
        })

        logger.info(f"작업 완료: {bgm_url}")
    except Exception as e:
        logger.error(f"에러 발생: {e}")
        send_webhook(webhook_url, {
            "memberId": request.memberId,
            "date": request.date,
            "error": str(e)
        })

def send_webhook(webhook_url: str, data: dict):
    """
    웹훅 URL로 데이터를 전송
    """
    try:
        response = requests.post(webhook_url, json=data)
        response.raise_for_status()
        logger.info(f"웹훅 전송 성공: {data}")
    except requests.RequestException as e:
        logger.error(f"웹훅 전송 실패: {e}")

@app.post("/music-ai", response_model=MusicResponse)
def create_music(request: MusicRequest, background_tasks: BackgroundTasks):
    """
    BGM 생성 요청을 받아 백그라운드에서 처리
    """
    try:
        task_id = str(uuid.uuid4())
        background_tasks.add_task(background_music_process, request)
        return MusicResponse(
            message="음악 생성 작업이 시작되었습니다.",
            taskId=task_id
        )
    except Exception as e:
        logger.error(f"오류 발생: {e}")
        raise HTTPException(status_code=500, detail="내부 서버 오류")

@app.post("/webhook")
def test_webhook(data: dict):
    """
    테스트용 웹훅 엔드포인트
    """
    logger.info(f"웹훅 수신 데이터: {data}")
    return {"message": "웹훅 수신 성공", "data": data}
