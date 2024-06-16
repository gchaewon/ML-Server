import requests
import json
import base64
from io import BytesIO
from pydub import AudioSegment
from pydub.playback import play

def play_audio_base64(base64_audio):
    # Base64 디코딩
    binary_audio = base64.b64decode(base64_audio)

    # 바이너리 데이터를 BytesIO로 읽어오기
    audio_bytes = BytesIO(binary_audio)

    # BytesIO를 AudioSegment로 변환
    audio = AudioSegment.from_file(audio_bytes, format="mp3")

    # 음성 재생
    play(audio)

def test_tts_api():
    # url = "https://도메인/model/tts"  #도메인 변경시 코드 변경
    url = "http://127.0.0.1:5001/model/tts"
    # 테스트할 질문 목록 데이터
    question_list = [
        {"id": 1, "questionText": "첫 번째 질문입니다. 프로세스와 스레드의 차이점에 대해 설명하시오."},
        {"id": 2, "questionText": "두 번째 질문입니다. 멀티 프로세싱과 멀티 프로그래밍의 차이점에 대해 설명하시오."},
        {"id": 3, "questionText": "세 번째 질문입니다. 프로세스의 5가지 상태에 대해 설명하시오."}
    ]

    # POST 요청 데이터 설정
    data = {
        "questionList": question_list
    }

    # GET 요청 보내기
    response = requests.post(url, json=data)

    # 응답 확인
    if response.status_code == 200:
        print("테스트 통과: 응답 코드 200 OK")
        response_data = response.json()
        print("음성 목록:")
        for voice in response_data["voiceList"]:
            print("질문 ID:", voice["id"])
            print("음성 재생 중...")
            play_audio_base64(voice["audio_content"])
            print("음성 재생 완료")
    else:
        print("테스트 실패: 응답 코드", response.status_code)

if __name__ == "__main__":
    test_tts_api()
