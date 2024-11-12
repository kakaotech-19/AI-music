# Python 3.10 슬림 이미지를 사용
FROM python:3.10-slim

# 작업 디렉토리를 설정합니다
WORKDIR /app

# 의존성 파일을 복사하고 설치합니다
COPY requirements.txt .

# 설치 후 정리하여 이미지 크기를 줄입니다
RUN apt-get update && \
    apt-get install -y --no-install-recommends gcc && \
    python3 -m pip install --upgrade pip && \
    pip install --no-cache-dir -r requirements.txt && \
    apt-get purge -y --auto-remove gcc && \
    rm -rf /var/lib/apt/lists/*

# 애플리케이션 소스 코드를 복사합니다
COPY app/ ./app

# 컨테이너 실행 시 Uvicorn을 사용하여 FastAPI 애플리케이션을 실행합니다
CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "5003"]