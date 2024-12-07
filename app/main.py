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

# 고정 웹훅 url 전송
DEFAULT_WEBHOOK_URL = os.getenv('DEFAULT_WEBHOOK_URL')

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
    genre: str         # 장르 정보 (BGM 장르 선정)

class MusicResponse(BaseModel):
    message: str
    taskId: str

def extract_instruments(content: str, genre: str) -> list:
    """
    사용자 선택 장르와 일기 내용을 기반으로 악기를 추출하는 함수
    """
    try:
        response = openai.ChatCompletion.create(
            model="gpt-4o-mini",
            messages=[
                {"role": "system",
                 "content": f"""당신은 일기 내용을 분석하여 주어진 장르와 해당 일기에 어울리는 악기를 추출하는 전문가입니다.
                                장르: {genre}
                                장르에 맞게 잘 쓰이는 악기들로 사용자의 일기 내용에서 음악에 사용할 악기를 3개 이상 추출하세요. 악기 목록은 아래와 같이 작성하세요:
                                악기 목록: 피아노, 기타, 드럼""" },
                {"role": "user", "content": f"다음 일기 내용을 기반으로 생성할 음악에 사용할 악기를 추출해줘.\n\n일기: {content}"}
            ],
            max_tokens=500,
            temperature=0.5,
        )
        instruments_text = response.choices[0].message['content'].strip()
        # "악기 목록: 피아노, 기타, 드럼" 형식으로 응답이 올 것으로 예상
        instruments = instruments_text.split(':')[1].strip().split(', ')
        logger.info(f"추출된 악기: {instruments}")
        return instruments
    except Exception as e:
        logger.error(f"악기 추출 오류: {e}")
        raise HTTPException(status_code=500, detail=f"악기 추출 오류: {str(e)}")

def generate_music_prompt(genre: str, instruments: list) -> str:
    """
    장르와 악기를 기반으로 음악 생성 프롬프트를 작성하는 함수
    """
    try:
        instruments_str = ', '.join(instruments)
        prompt = f"장르: {genre}\n악기: {instruments_str}\n음악의 처음과 끝에 페이드 효과를 1초 넣어주세요."
        logger.info(f"생성된 음악 프롬프트: {prompt}")
        return prompt
    except Exception as e:
        logger.error(f"음악 프롬프트 생성 오류: {e}")
        raise HTTPException(status_code=500, detail=f"음악 프롬프트 생성 오류: {str(e)}")

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
                "duration": 15,
                "output_format": "mp3",
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
        file_key = f"music-ai/{member_id}/{year}/{month}/{day}/bgm.mp3"

        # S3에 파일 업로드
        s3_client.put_object(
            Bucket=AWS_S3_BUCKET_NAME,
            Key=file_key,
            Body=file_content,
            ContentType='audio/mpeg'
        )
        # S3 파일 URL 생성
        bgm_url = f"https://{AWS_S3_BUCKET_NAME}.s3.{AWS_S3_REGION}.amazonaws.com/{file_key}"
        logger.info(f"S3 업로드 완료: {bgm_url}")
        return bgm_url

    except (BotoCoreError, NoCredentialsError) as e:
        logger.error(f"S3 업로드 오류: {e}")
        raise HTTPException(status_code=500, detail=f"S3 업로드 오류: {str(e)}")

def send_webhook(webhook_url: str, data: dict, retries: int = 3):
    """
    클라이언트의 웹훅 URL로 POST 요청을 보내는 함수
    """
    headers = {
        "Content-Type": "application/json"
    }
    for attempt in range(retries):
        try:
            response = requests.post(webhook_url, json=data, headers=headers, timeout=10)
            response.raise_for_status()
            logger.info(f"웹훅 성공적으로 전송: {webhook_url}")
            break
        except requests.exceptions.RequestException as e:
            logger.error(f"웹훅 전송 시도 {attempt + 1} 실패: {e}")
            if attempt < retries - 1:
                logger.info("재시도 중...")
            else:
                logger.error(f"웹훅 전송이 모두 실패했습니다: {webhook_url}")

def background_music_process(request: MusicRequest):
    """
    백그라운드에서 음악을 생성하고 S3에 업로드한 후 웹훅으로 결과를 전송하는 함수
    """
    try:
        member_id = request.memberId
        date_str = request.date
        content = request.content
        webhook_url = DEFAULT_WEBHOOK_URL # 고정된 웹훅 URL 사용
        genre = request.genre
        logger.info(f"백그라운드 작업 시작: memberId={member_id}, date={date_str}")

        # 악기 추출
        instruments = extract_instruments(content, genre)

        # 프롬프트 생성
        prompt = generate_music_prompt(genre, instruments)

        # 음악 생성
        music_content = generate_music_with_replicate(prompt)

        if not music_content:
            logger.error("음악 생성 후 파일 내용이 비어 있습니다.")
            # 웹훅으로 실패 알림 전송
            send_webhook(webhook_url, {
                "memberId": member_id,
                "date": date_str,
                "error": "음악 생성 실패: 파일 내용이 비어 있습니다."
            })
            return

        # S3에 업로드
        bgm_url = upload_to_s3(music_content, member_id, date_str)

        # 웹훅으로 성공 알림 전송
        send_webhook(webhook_url, {
            "memberId": member_id,
            "date": date_str,
            "bgmUrl": bgm_url
        })

        logger.info(f"백그라운드 작업 완료: memberId={member_id}, bgmUrl={bgm_url}")

    except HTTPException as http_exc:
        # 웹훅으로 에러 알림 전송
        send_webhook(DEFAULT_WEBHOOK_URL, {
            "memberId": request.memberId,
            "date": request.date,
            "error": http_exc.detail
        })
    except Exception as e:
        logger.error(f"백그라운드 작업 중 예외 발생: {e}")
        # 웹훅으로 에러 알림 전송
        send_webhook(DEFAULT_WEBHOOK_URL, {
            "memberId": request.memberId,
            "date": request.date,
            "error": f"백그라운드 작업 중 예외 발생: {str(e)}"
        })

@app.post("/music-ai", response_model=MusicResponse)
def create_music(request: MusicRequest, background_tasks: BackgroundTasks):
    """
    BGM 생성 요청을 받아 백그라운드에서 음악을 생성하고 S3에 업로드한 후 웹훅으로 결과를 전달하는 엔드포인트
    """
    try:
        member_id = request.memberId
        date_str = request.date
        genre = request.genre

        logger.info(f"음악 생성 요청 수신: memberId={member_id}, date={date_str}, genre={genre}")

        # 백그라운드 작업 추가
        background_tasks.add_task(background_music_process, request)

        # 즉시 응답
        task_id = str(uuid.uuid4())
        return MusicResponse(
            message="음악 생성 작업이 시작되었습니다. 작업 완료 시 웹훅으로 알림을 받으실 수 있습니다.",
            taskId=task_id
        )

    except Exception as e:
        logger.error(f"Unhandled error: {e}")
        raise HTTPException(status_code=500, detail=str(e))
