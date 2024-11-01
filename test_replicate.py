import replicate
import os
from dotenv import load_dotenv
import requests

# Load environment variables
load_dotenv()
REPLICATE_API_TOKEN = os.getenv('REPLICATE_API_TOKEN')

if not REPLICATE_API_TOKEN:
    raise ValueError("REPLICATE_API_TOKEN 환경 변수가 설정되지 않았습니다.")

try:
    output = replicate.run(
        "meta/musicgen:671ac645ce5e552cc63a54a2bbff63fcf798043055d2dac5fc9e36a837eedcfb",
        input={
            "prompt": "Edo25 major g melodies that sound triumphant and cinematic. Leading up to a crescendo that resolves in a 9th harmonic",
            "duration": 15,  # 15초
            "output_format": "mp3",
            "normalization_strategy": "peak"
        },
        api_token=REPLICATE_API_TOKEN
    )
    print(f"Replicate Output Type: {type(output)}")
    print(f"Replicate Output: {output}")

    if isinstance(output, str):
        print(f"음악 생성 완료. 출력 URL: {output}")
        # Optionally, download the file
        response = requests.get(output)
        response.raise_for_status()
        with open("test_output.mp3", "wb") as f:
            f.write(response.content)
        print("음악 파일 다운로드 및 저장 완료: test_output.mp3")
    elif hasattr(output, 'read'):
        music_content = output.read()
        with open("test_output.mp3", "wb") as f:
            f.write(music_content)
        print("음악 파일 다운로드 및 저장 완료: test_output.mp3")
    else:
        print("Unexpected output type from Replicate")
except replicate.exceptions.ApiException as e:
    print(f"Replicate API 오류: {e}")
except Exception as e:
    print(f"음악 생성 오류: {e}")
